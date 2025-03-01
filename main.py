from dataclasses import dataclass
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from duckduckgo_search import DDGS
import openai
import time

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

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[dict]
    usage: dict

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

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
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

        # Format response like OpenAI's API
        response = ChatCompletionResponse(
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.data.response  # Access through result.data
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 0,  # You might want to implement token counting
                "completion_tokens": 0,
                "total_tokens": 0
            }
        )

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