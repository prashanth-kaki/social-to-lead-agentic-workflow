import gradio as gr
from langchain_core.messages import HumanMessage
from agent import agent_app
import uuid
import os

# Ensure the GOOGLE_API_KEY is present
if not os.environ.get("GOOGLE_API_KEY"):
    print("Warning: GOOGLE_API_KEY environment variable not set. Please set it in a .env file or environment.")

def chat_interface(message, history):
    # We maintain a single session thread for the Gradio user
    # Or ideally create a unique thread id per session.
    # We can use a static thread_id or generate one if we wanted to support multiple tabs.
    # Currently maintaining one main thread.
    thread_id = "gradio-session-1" 
    config = {"configurable": {"thread_id": thread_id}}
    
    # We don't really rely on Gradio's history array here because agent_app tracks its own memory
    # via the checkpointer in LangGraph, but we pass the new user message.
    human_msg = HumanMessage(content=message)
    
    # Invoke the agent
    response = agent_app.invoke(
        {"messages": [human_msg]},
        config=config
    )
    
    # The agent returns the updated state.
    # Output the last message
    ai_response = response["messages"][-1].content
    return ai_response

demo = gr.ChatInterface(
    fn=chat_interface,
    title="AutoStream Social-to-Lead Agent",
    description="Conversational agent for AutoStream that can answer pricing queries and qualify high-intent leads.",
    examples=["Hi, who are you?", "What does the Pro plan cost?", "I want to sign up for the basic plan, my name is John."],
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
