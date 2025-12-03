
import gradio as gr
import requests
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import messages_to_dict
import json

API_URL = "http://localhost:8000//mcp/invoke"

chat_history = []

def chat(user_input):
    global chat_history

    # Serialize history to send as JSON
    serialized_history = messages_to_dict(chat_history)

    payload = {
        "input": user_input,
        "chat_history": serialized_history
    }

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()["output"]
    except Exception as e:
        result = f"Error: {str(e)}"

    # Append messages after response
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=result))

    return result

def reset():
    global chat_history
    chat_history = []
    return "History cleared. Ready for a new conversation!"

initial_message = [
    (
        "👋", 
        "Hello! I'm your **Telecom MCP Assistant**. I have access to historical Telecom **KPI data**.\n\nYou can ask me questions like:\n- *Were there anomalies in DL_Throughput?*\n- *On which day SINR to drop at SITE_005?*\n- *Which site had the highest packet loss last week?*"
    )
]

with gr.Blocks() as demo:
    gr.Markdown("## 📡 Telecom KPI Analysis Assistant")

    chatbot = gr.Chatbot(value=initial_message)
    user_input = gr.Textbox(label="Ask something...")
    send_btn = gr.Button("Send")
    clear_btn = gr.Button("🔄 Clear Conversation")

    def respond(message, history):
        reply = chat(message)
        history.append((message, reply))
        return "", history

    send_btn.click(respond, [user_input, chatbot], [user_input, chatbot])
    clear_btn.click(lambda: ([], reset()), None, [chatbot])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
