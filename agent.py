import os
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import json
from dotenv import load_dotenv

load_dotenv()

# Define the State
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    collected_details: dict

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.0)

# RAG Setup: Load Knowledge Base into an in-memory FAISS VectorStore
current_dir = os.path.dirname(os.path.abspath(__file__))
kb_path = os.path.join(current_dir, "knowledge_base.md")

if os.path.exists(kb_path):
    loader = TextLoader(kb_path)
    docs = loader.load()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
else:
    retriever = None

# --- Intent Classification node ---
class IntentClassification(BaseModel):
    intent: Literal["Greeting", "Pricing Inquiry", "High-Intent Lead"] = Field(
        description="Classify the user intent into one of these three categories"
    )

def intent_classifier_node(state: AgentState):
    messages = state["messages"]
    last_msg = messages[-1].content
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an intent classifier for AutoStream, a SaaS for video editing tools. "
                   "Classify the user intent into exactly one of three categories: "
                   "'Greeting', 'Pricing Inquiry', or 'High-Intent Lead'.\n"
                   "- 'Greeting': Casual greetings like 'hi', 'hello', 'how are you', 'good morning'.\n"
                   "- 'Pricing Inquiry': Asking about plans, pricing, features, or policies. E.g. 'What does it cost?'\n"
                   "- 'High-Intent Lead': Expressions of intent to purchase, sign up, or try a plan. E.g. 'I want to try the Pro plan', 'Sign me up'."),
        ("human", "{user_msg}")
    ])
    
    chain = prompt | llm.with_structured_output(IntentClassification)
    res = chain.invoke({"user_msg": last_msg})
    
    return {"intent": res.intent}

def route_intent(state: AgentState):
    intent = state.get("intent", "Greeting")
    if intent == "Greeting":
        return "greeting_node"
    elif intent == "Pricing Inquiry":
        return "rag_node"
    elif intent == "High-Intent Lead":
        return "lead_node"
    return "greeting_node"

# --- Handler Nodes ---
def greeting_node(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly conversational assistant for AutoStream. Reply casually to the greeting and ask how you can help. Keep it concise."),
        ("human", "{user_msg}")
    ])
    chain = prompt | llm
    res = chain.invoke({"user_msg": state["messages"][-1].content})
    return {"messages": [res]}

def rag_node(state: AgentState):
    user_msg = state["messages"][-1].content
    if retriever:
        docs = retriever.invoke(user_msg)
        context = "\n".join([d.page_content for d in docs])
    else:
        context = "No knowledge base found. Advise user that information is unavailable right now."
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for AutoStream. Answer the user's question about pricing or features strictly based on the provided context.\n\nContext:\n{context}"),
        ("human", "{user_msg}")
    ])
    chain = prompt | llm
    res = chain.invoke({"context": context, "user_msg": user_msg})
    return {"messages": [res]}


# --- Lead Capture Tool ---
def mock_lead_capture(name: str, email: str, platform: str):
    print("\n" + "="*40)
    print(f"LEAD CAPTURED SUCCESSFULLY:")
    print(f"Name: {name}")
    print(f"Email: {email}")
    print(f"Platform: {platform}")
    print("="*40 + "\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"

class LeadDetails(BaseModel):
    name: str = Field(description="Name of the user, leave empty string if not provided", default="")
    email: str = Field(description="Email address of the user, leave empty string if not provided", default="")
    platform: str = Field(description="Creator platform (e.g., YouTube, Instagram, Tiktok), leave empty string if not provided", default="")

def lead_node(state: AgentState):
    details = state.get("collected_details", {})
    if not details:
        details = {"name": "", "email": "", "platform": ""}
        
    messages = state["messages"]
    
    # We pass the conversation history to extract any details provided
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Read the conversation and extract the user's profile details. "
                   "Strictly return empty string for fields that haven't been provided by the user yet."),
        ("human", "Conversation History:\n{conversation}")
    ])
    
    conv_str = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}" for m in messages])
    chain = prompt | llm.with_structured_output(LeadDetails)
    extracted = chain.invoke({"conversation": conv_str})
    
    # Update state collected details (overwrite empty ones with new ones)
    new_details = {
        "name": extracted.name if extracted.name else details.get("name", ""),
        "email": extracted.email if extracted.email else details.get("email", ""),
        "platform": extracted.platform if extracted.platform else details.get("platform", "")
    }
    
    missing = []
    for k in ["name", "email", "platform"]:
        val = new_details.get(k, "")
        if not val or str(val).lower() in ["none", "null", "unknown", "missing"]:
            missing.append(k)
            
    if missing:
        # Prompt user for missing fields
        missing_str = " and ".join([f"**{m}**" for m in missing])
        msg = f"Awesome! I'd love to help you get set up with AutoStream. To proceed, could you please provide your {missing_str}?"
        return {"messages": [AIMessage(content=msg)], "collected_details": new_details}
    else:
        # We have all details assembled, trigger the tool
        result = mock_lead_capture(new_details["name"], new_details["email"], new_details["platform"])
        msg = f"Thanks {new_details['name']}! Everything is set up. ({result}). Our team will be in touch shortly!"
        # Clear details so a new lead isn't accidentally stored, or leave them. We'll leave them.
        return {"messages": [AIMessage(content=msg)], "collected_details": new_details}

# --- Graph Compilation ---
graph_builder = StateGraph(AgentState)
graph_builder.add_node("intent_classifier", intent_classifier_node)
graph_builder.add_node("greeting_node", greeting_node)
graph_builder.add_node("rag_node", rag_node)
graph_builder.add_node("lead_node", lead_node)

graph_builder.add_edge(START, "intent_classifier")
graph_builder.add_conditional_edges("intent_classifier", route_intent)
graph_builder.add_edge("greeting_node", END)
graph_builder.add_edge("rag_node", END)
graph_builder.add_edge("lead_node", END)

# Configure Memory in LangGraph requires using `checkpointer`. 
# For simple testing across turns, MemorySaver can be used.
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

agent_app = graph_builder.compile(checkpointer=memory)
