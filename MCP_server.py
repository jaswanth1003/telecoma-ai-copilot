from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage
from Agent import agent_executor  # Import from agent.py
from langchain_core.messages import messages_from_dict, messages_to_dict
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
from fastapi_mcp import FastApiMCP
import uvicorn

app = FastAPI(title="Telecom MCP Server", version="0.1")

class ChatInput(BaseModel):
    input: str
    chat_history: List[Dict] = Field(default=[])

class ChatOutput(BaseModel):
    output: str

@app.post("/mcp/invoke", response_model=ChatOutput, operation_id="Get_Agent")
def invoke_agent(input: ChatInput):
    result = agent_executor.invoke({
        "input": input.input,
        "chat_history": input.chat_history
    })

    messages = result.get("messages", [])
    last_ai = next((msg for msg in reversed(messages) if msg.type == "ai"), None)

    return {"output": last_ai.content if last_ai else "❌ No response generated."}

@app.post("/mcp/stream")
async def stream_agent(input: ChatInput):
    async def event_stream() -> AsyncGenerator[str, None]:
        async for chunk in agent_executor.astream({
            "input": input.input,
            "chat_history": input.chat_history
        }):
            if isinstance(chunk, dict) and "output" in chunk:
                yield chunk["output"]
            else:
                yield str(chunk)
    
    return StreamingResponse(event_stream(), media_type="text/plain")

# @app.get("/mcp")
# def mcp_root():
#     return {
#         "message": "Welcome to Telecom MCP. Use /mcp/invoke, or /mcp/stream."
#     }

mcp = FastApiMCP(app, "Get_Agent")
mcp.mount()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)