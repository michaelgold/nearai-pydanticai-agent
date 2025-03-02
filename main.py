from dataclasses import dataclass
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from duckduckgo_search import DDGS
import openai
import time
import tiktoken
from fastapi.responses import StreamingResponse
import json
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


config_path = Path.home() / '.nearai' / 'config.json'
if not config_path.exists():
    config_path = Path('config.json')  # Fallback to local config.json

with open(config_path) as f:
    config = json.load(f)
    signature = json.dumps(config['auth'])

near_hub_url = "https://api.near.ai/v1"

near_client = openai.AsyncOpenAI(base_url=near_hub_url, api_key=signature)


model_name = "fireworks::accounts/fireworks/models/llama-v3p1-405b-instruct"

near_model = OpenAIModel(
    model_name=model_name,
    openai_client=near_client
)


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@dataclass
class UserDependencies:
    name: str
    ddgs: DDGS
    openai_client: openai.OpenAI

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = Field(default=model_name)
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2000
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    system_fingerprint: str = Field(default="fp_44709d6fcb")
    choices: List[dict]
    usage: dict = Field(default_factory=dict)

class AIResponse(BaseModel):
    response: str = Field(description='Response to the user query')
    context_used: bool = Field(description='Whether user context was used')
    news_added: bool = Field(description='Whether news context was added')

class PromptRequest(BaseModel):
    prompt: str = Field(description="User's prompt or question")

ai_agent = Agent(
    model=near_model,
    deps_type=UserDependencies,
    result_type=AIResponse,
    system_prompt=(
        'You are an AI assistant that can impersonate users and provide information about recent events. You must always use the name of the user you are impersonating. When you respond, you must use the style of voice of the user you are impersonating. You are not allowed to say that you are an AI assistant. Respond briefly and concisely in a conversational tone with just one or two sentences. When answering questions, do not start with \'As <user_name>\' or \'As the user <user_name>\' or anything similar. Just answer the question directly.'
    
    ),
)

@ai_agent.system_prompt
async def add_user_context(ctx: RunContext[UserDependencies]) -> str:
    if ctx.deps.name:
        return f"You are now impersonating {ctx.deps.name}."
    return ""

@ai_agent.tool
async def search_recent_news(
    ctx: RunContext[UserDependencies],
    query: str,
    max_results: int = 10
) -> str:
    """Search for recent news articles related to the query."""
    with ctx.deps.ddgs as ddgs:
        results = list(ddgs.news(query, max_results=max_results))
    
    if not results:
        return f"No recent news found for: {query}"
    
    news = "Recent news:\n"
    for result in results:
        # Use 'body' instead of 'snippet' or handle missing keys
        title = result.get('title', 'No title')
        body = result.get('body', result.get('snippet', 'No content available'))
        news += f"- {title}: {body}\n"
    return news

@ai_agent.tool
async def search_general_info(
    ctx: RunContext[UserDependencies],
    query: str,
    max_results: int = 10
) -> str:
    """Search for general information related to the query."""
    with ctx.deps.ddgs as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    
    if not results:
        return f"No general information found for: {query}"
    
    info = "General information:\n"
    for result in results:
        title = result.get('title', 'No title')
        body = result.get('body', result.get('snippet', 'No content available'))
        info += f"- {title}: {body}\n"
    return info

def count_tokens(text: str, model: str = model_name) -> int:
    """Count the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

async def create_stream_response(response_text: str, request: ChatCompletionRequest):
    """Generate streaming response chunks."""
    # Split response into words/chunks for streaming
    chunks = response_text.split()
    
    for i, chunk in enumerate(chunks):
        is_last = i == len(chunks) - 1
        chunk_data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant" if i == 0 else None,
                    "content": chunk + (" " if not is_last else "")
                },
                "finish_reason": "stop" if is_last else None
            }]
        }
        
        yield f"data: {json.dumps(chunk_data)}\n\n"
        
        if not is_last:
            await asyncio.sleep(0.02)  # Add small delay between chunks
    
    # Send the final [DONE] message
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    logger.debug(f"Received request: {request}")
    try:
        # Create dependencies
        deps = UserDependencies(
            name=user_context.get("name", ""),
            ddgs=DDGS(),
            openai_client=near_client
        )

        # Get the last user message
        last_message = request.messages[-1].content
        
        # Run the agent
        result = await ai_agent.run(last_message, deps=deps)

        logger.debug(f"Agent Result: {result.data}")
        
        # Handle streaming response
        if request.stream:
            return StreamingResponse(
                create_stream_response(result.data.response, request),
                media_type="text/event-stream"
            )
        
        # Count tokens for non-streaming response
        prompt_tokens = sum(count_tokens(msg.content, request.model) for msg in request.messages)
        completion_tokens = count_tokens(result.data.response, request.model)
        
        # Format regular response
        response = ChatCompletionResponse(
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.data.response
                },
                "logprobs": None,
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0
                }
            }
        )

        logger.debug(f"Response: {response}")
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set-user")
async def set_user(name: str):
    """Set the system prompt with user's name and background."""
    user_context["name"] = name
    
    # Use the news search tool to get background
    deps = UserDependencies(name=name, ddgs=DDGS(), openai_client=near_client)
    
    # Create a proper RunContext with required parameters
    ctx = RunContext(
        deps=deps,
        model=near_model,  # This can be any model identifier
        usage={},       # Empty usage dictionary
        prompt=""       # Empty prompt
    )
    
    news_background = await search_recent_news(ctx, name)
    general_background = await search_general_info(ctx, name)
    
    user_context["background"] = f"{general_background}\n\n{news_background}"
    
    return {
        "message": f"User context set for {name}",
        "background": user_context["background"]
    }


# Store the system prompt/user context
user_context = {
    "name": "",
    "background": ""
}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)