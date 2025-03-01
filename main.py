from dataclasses import dataclass
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from duckduckgo_search import DDGS
import openai
import time
import tiktoken
from fastapi.responses import StreamingResponse
import json
import asyncio

app = FastAPI()

@dataclass
class UserDependencies:
    name: str
    ddgs: DDGS
    openai_client: openai.OpenAI

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-4")
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
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
    'openai:gpt-4',
    deps_type=UserDependencies,
    result_type=AIResponse,
    system_prompt=(
        'You are an AI assistant that can impersonate users and provide information about recent events. You must always use the name of the user you are impersonating. When you respond, you must use the style of voice of the user you are impersonating. You are not allowed to say that you are an AI assistant.'
    
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

def count_tokens(text: str, model: str = "gpt-4") -> int:
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
    print(request)
    try:
        # Create dependencies
        deps = UserDependencies(
            name=user_context.get("name", ""),
            ddgs=DDGS(),
            openai_client=openai.OpenAI()
        )

        # Get the last user message
        last_message = request.messages[-1].content
        
        # Run the agent
        result = await ai_agent.run(last_message, deps=deps)
        
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

        print(response)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set-user")
async def set_user(name: str):
    """Set the system prompt with user's name and background."""
    user_context["name"] = name
    
    # Use the news search tool to get background
    deps = UserDependencies(name=name, ddgs=DDGS(), openai_client=openai.OpenAI())
    
    # Create a proper RunContext with required parameters
    ctx = RunContext(
        deps=deps,
        model="gpt-4",  # This can be any model identifier
        usage={},       # Empty usage dictionary
        prompt=""       # Empty prompt
    )
    
    background = await search_recent_news(ctx, name)
    user_context["background"] = background
    
    return {
        "message": f"User context set for {name}",
        "background": background
    }

@app.post("/prompt")
async def process_prompt(request: PromptRequest):
    try:
        # Create dependencies
        deps = UserDependencies(
            name=user_context.get("name", ""),
            ddgs=DDGS(),
            openai_client=openai.OpenAI()
        )

        # Run the agent with the user's prompt
        result = await ai_agent.run(request.prompt, deps=deps)

        # Return a simplified response
        return {
            "response": result.data.response,  # Access through result.data
            "context_used": result.data.context_used,
            "news_added": result.data.news_added
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Store the system prompt/user context
user_context = {
    "name": "",
    "background": ""
}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)