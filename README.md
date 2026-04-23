# AutoStream Conversational Agent

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Enabled-success.svg)](https://python.langchain.com/docs/langgraph)
[![Gemini](https://img.shields.io/badge/Gemini-Flash-orange.svg)](https://ai.google.dev/)

An intelligent, stateful Conversational AI Agent for **AutoStream**—a fictional SaaS platform for content creators. This project implements a sophisticated multi-turn workflow with intent routing, context-aware retrieval, and lead capturing functionalities using LangGraph and Google's Gemini models.

## Key Features

- **Dynamic Intent Classification**: Analyzes the user's input to predict their intent (e.g., *Greeting*, *Pricing Inquiry*, or *High-Intent Lead*) and routes the conversation to the appropriate specialized node.
- **RAG-Powered Knowledge Retrieval**: Intercepts pricing and policy questions, retrieving precise context from an embedded FAISS vector store to generate accurate, grounded responses.
- **Smart Lead Capture System**: Intelligently extracts the user's Name, Email, and Creator Platform (e.g., YouTube, TikTok) from the conversational context. It seamlessly prompts the user for any missing information before triggering a mock tool submission.
- **Robust State Management**: Leverages LangGraph's checkpointer to maintain conversational memory across multiple interaction turns, ensuring a natural and coherent dialogue.

---

## How It Works

The agent is constructed as a stateful graph utilizing **LangChain** and **LangGraph**. Each user message is processed through the following pipeline:

1. **Decision Router (`intent_classifier`)**: The entry point that classifies user intent and dynamically routes the conversation to the most relevant sub-system.
2. **Greeting Handler (`greeting_node`)**: A lightweight node that handles casual conversations and welcomes the user.
3. **RAG Handler (`rag_node`)**: Activated for pricing or feature inquiries. It retrieves relevant documentation from an in-memory `FAISS` vector store (populated by `knowledge_base.md`) and synthesizes an answer using the LLM.
4. **Lead Capture Handler (`lead_node`)**: Activated when a user expresses high intent to purchase or sign up. It scans the entire conversation history, maps it against a Pydantic schema to extract lead data, and determines if fields are missing. If data is missing, it asks the user for it; if complete, it executes the lead capture tool.

---

## Setup and Installation

Follow these steps to run the agent locally.

### 1. Clone the Repository
```bash
git clone https://github.com/prashanth-kaki/social-to-lead-agentic-workflow.git
cd social-to-lead-agentic-workflow
```

### 2. Install Dependencies
Ensure you have Python installed, then set up a virtual environment and install the required packages:
```bash
python -m venv .venv
# On Windows use: .\.venv\Scripts\activate
# On Mac/Linux use: source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Configuration
The agent utilizes Google Gemini. You need an API key from Google AI Studio.
```bash
cp .env.example .env
```
Edit the `.env` file and insert your Google API key:
```env
GOOGLE_API_KEY="your_google_api_key_here"
```

---

## How to Use

To start interacting with the agent, launch the local Gradio application:

```bash
python app.py
```

Once running, the terminal will display a local URL (typically `http://127.0.0.1:7860/`). 
1. Open that URL in your web browser.
2. You will be presented with a chat interface.
3. **Test Greetings:** Try saying "Hello" or "Good morning".
4. **Test RAG:** Ask about AutoStream's pricing, e.g., "How much does the Pro plan cost?"
5. **Test Lead Capture:** Express intent to sign up, e.g., "I'm ready to subscribe." The agent will guide you through providing your name, email, and platform if you haven't mentioned them yet.

---

## Architecture Decisions

### Why LangGraph?
LangGraph was chosen for its ability to provide highly controllable, deterministic cyclic graphs. Unlike fully autonomous agent frameworks (like AutoGen) which can sometimes loop unpredictably, LangGraph allows for explicit structural edges. This makes the router and state checkpoints highly predictable—a critical requirement for enterprise-grade customer-facing chatbots.

### State Management
State is managed across multi-turn interactions utilizing LangGraph's `checkpointer` infrastructure. The current state comprises the entire conversational `messages` list, the currently predicted `intent`, and a `collected_details` dictionary. The Graph engine natively handles appending new messages to the transcript arrays without manual buffer adjustments.

---

## WhatsApp Webhook Deployment Strategy

To deploy this agent to a production WhatsApp environment:
1. **Infrastructure**: Expose a backend service (e.g., FastAPI wrapping `agent_app.invoke()`) using a reverse proxy or serverless endpoints.
2. **WhatsApp Business API**: Register an application on the Meta Developer Portal.
3. **Webhook Setup**: Configure the Webhook URL pointing to your backend endpoint (e.g., `/api/whatsapp/webhook`).
4. **Message Pipeline**: Incoming WhatsApp messages map the sender's phone number to a unique LangGraph `thread_id` to retrieve conversational memory. The generated response is then pushed back via the WhatsApp Graph API.
