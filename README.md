# AutoStream Conversational Agent

A Conversational AI Agent for **AutoStream** (a fictional SaaS for content creators) developed as a Machine Learning Assignment.

## Features
- **Intent Classification**: Predicts intent as Greeting, Pricing Inquiry, or High-Intent Lead.
- **RAG-Powered Knowledge Base**: Answers pricing and policy questions utilizing a local FAISS-based vector store.
- **Lead Capture Tool Execution**: Intelligently extracts user name, email, and platform across conversations, querying missing fields before executing a mock tool submission.
- **State Management**: Robust memory using LangGraph's checkpointer across multiple interaction turns.

## Setup & Running Locally

1. **Clone the repository** and navigate to the root directory `cd Inflx`
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set your Environment Variables**:
   Copy the example file and add your OpenAI API key:
   ```bash
   cp .env.example .env
   # Edit .env and enter your OPENAI_API_KEY
   ```
4. **Run the App**:
   ```bash
   python app.py
   ```
   *This will launch a local Gradio Web UI on http://127.0.0.1:7860/ where you can chat with the assistant.*

## Architecture Summary

The agent is built using **LangChain** and **LangGraph**. Every user message passes through an architecture consisting of distinct conceptual nodes:
1. **Decision/Router Node (`intent_classifier`)**: Extracts the context and classifies user intent to dynamically route the conversation to the most relevant sub-system.
2. **Greeting Handler (`greeting_node`)**: Forwards the prompt to a fast, zero-shot conversational agent.
3. **RAG Handler (`rag_node`)**: When a pricing inquiry is detected, this agent retrieves context from an in-memory `FAISS` vector store populated by `knowledge_base.md`. 
4. **Lead Capture Handler (`lead_node`)**: Manages complex high-intent workflows. It scans the entire conversational transcript, maps it against a Pydantic schema using structured outputs, and identifies missing data parameters. If fields (Name, Email, Platform) are missing, it asks the user for them. Otherwise, it issues a function call to complete the capture. 

### Why LangGraph?
LangGraph was chosen over AutoGen because it provides highly controllable and deterministic cyclic graphs natively suited originally for stateful flow engineering. LangGraph allows explicit structural edges, making the router and state checkpoints very predictable, which is essential for deterministic enterprise chatbots.

### State Management
State is managed across multi-turn interactions utilizing LangGraph's `checkpointer` infrastructure (implemented via `MemorySaver()`). The current state comprises the entire conversational `messages` list, the currently predicted `intent`, and a `collected_details` dictionary. The Graph engine natively handles appending new messages to the transcript arrays without manual buffer adjustments during sequential calls to the `.invoke()` method.

## WhatsApp Webhook Deployment Strategy

To deploy this setup as a WhatsApp agent:
1. **Infrastructure**: Expose a backend service (e.g., FastAPI or Flask server wrapping `agent_app.invoke()`) using a reverse proxy (like Nginx) or Serverless endpoints (like AWS Lambda or Google Cloud Run).
2. **WhatsApp Business API**: Register an application on the Meta Developer Portal and setup the WhatsApp Business Platform API.
3. **Webhook Setup**: Configure the app's Webhook URL (with a verification token) pointing to your backend endpoint (e.g., `/api/whatsapp/webhook`).
4. **Message Handling Pipeline**:
   - Meta posts incoming user messages as JSON to the webhook.
   - The backend retrieves the sender's phone number and the message text.
   - The phone number maps to a unique LangGraph `thread_id` to retrieve state and conversational memory.
   - The message is processed by `agent_app.invoke()`.
   - The generated response is formatted and pushed back to the user via an outgoing HTTPS POST request to the WhatsApp Graph API `/messages` endpoint.
