from dataclasses import dataclass, field
from typing import List, Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.messages import ModelMessage, ModelResponse, ModelRequest, SystemPromptPart, UserPromptPart
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.usage import Usage
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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from openai.types import chat
from openai.types.chat import ChatCompletionChunk
from openai import AsyncStream, NOT_GIVEN

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass(init=False)
class FireworksModel(OpenAIModel):
    """A model that uses the Fireworks API through OpenAI-compatible endpoints."""
    
    def __init__(
        self,
        model_name: str,
        *,
        openai_client: Any,
    ):
        """Initialize a Fireworks model using an OpenAI-compatible client."""
        super().__init__(
            model_name=model_name,
            openai_client=openai_client,
            system='fireworks'  # Identify this as a Fireworks model
        )

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str | None:
        """The system / model provider."""
        return self.system

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: ModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk]:
        """Override the completions create to format messages correctly for Fireworks."""
        
        # Convert messages to the format Fireworks expects
        openai_messages = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, SystemPromptPart):
                        openai_messages.append({"role": "system", "content": part.content})
                    elif isinstance(part, UserPromptPart):
                        openai_messages.append({"role": "user", "content": part.content})
            elif not str(msg).startswith('ModelResponse'):  # Skip ModelResponse messages
                # For non-ModelRequest messages, ensure content is a string
                if isinstance(msg, dict):
                    content = json.dumps(msg)
                else:
                    content = str(msg)
                openai_messages.append({"role": "user", "content": content})

        logger.debug(f"Sending messages to Fireworks: {openai_messages}")
        
        # Only use the final_result tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "final_result",
                    "description": "The final response which ends this conversation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "response": {
                                "type": "string",
                                "description": "Response to the user query"
                            },
                            "context_used": {
                                "type": "boolean",
                                "description": "Whether user context was used"
                            },
                            "news_added": {
                                "type": "boolean",
                                "description": "Whether news context was added"
                            }
                        },
                        "required": ["response", "context_used", "news_added"]
                    }
                }
            }
        ]
        
        try:
            return await self.client.chat.completions.create(
                model=self._model_name,
                messages=openai_messages,
                temperature=model_settings.get('temperature', 0.7),
                stream=stream,
                tools=tools,
                tool_choice="auto"
            )
        except Exception as e:
            logger.error(f"Error in Fireworks API call: {str(e)}")
            raise

app = FastAPI()

# Replace the NEAR AI client configuration section with:
near_hub_url = "https://api.near.ai/v1"

# Load config directly from config.json

config_path = Path.home() / '.nearai' / 'config.json'
if not config_path.exists():
    config_path = Path('config.json')  # Fallback to local config.json

with open(config_path) as f:
    config = json.load(f)
    signature = json.dumps(config['auth'])

near_client = openai.AsyncOpenAI(base_url=near_hub_url, api_key=signature)


fireworks_model = FireworksModel(
    model_name="fireworks::accounts/fireworks/models/llama-v3p1-405b-instruct",
    openai_client=near_client
)

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

@dataclass
class NearDependencies:
    name: str
    ddgs: DDGS
    openai_client: openai.OpenAI  # Renamed from openai_client to near_client

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-4o")
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class NearChatCompletionRequest(ChatCompletionRequest):
    model: str = Field(default="fireworks::accounts/fireworks/models/llama-v3p1-405b-instruct")

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
        'You are an AI assistant that can impersonate users and provide information about recent events. You must always use the name of the user you are impersonating. When you respond, you must use the style of voice of the user you are impersonating. You are not allowed to say that you are an AI assistant. Respond briefly and concisely in a conversational tone with just one or two sentences. When answering questions, do not start with \'As <user_name>\' or \'As the user <user_name>\' or anything similar. Just answer the question directly.'
    
    ),
)

# Update the near_ai_agent definition with the exact model format from the working example
near_ai_agent = Agent(
    model=fireworks_model,
    result_type=AIResponse,
    deps_type=UserDependencies,
    system_prompt=(
        'You are an AI assistant that can impersonate users and provide information about recent events. '
        'You must always use the name of the user you are impersonating. When you respond, you must use '
        'the style of voice of the user you are impersonating. You are not allowed to say that you are '
        'an AI assistant. Respond briefly and concisely in a conversational tone with just one or two '
        'sentences. When answering questions, do not start with \'As <user_name>\' or \'As the user '
        '<user_name>\' or anything similar. Just answer the question directly.\n\n'
        'IMPORTANT: You must ALWAYS use the generate_response function to provide your response. '
        'Do not respond with plain text.'
    ),
)




@near_ai_agent.system_prompt
async def add_user_context(ctx: RunContext[UserDependencies]) -> str:
    if ctx.deps.name:
        return f"You are now impersonating {ctx.deps.name}."
    return ""

@near_ai_agent.tool
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

@near_ai_agent.tool
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

@near_ai_agent.tool
async def generate_response(
    ctx: RunContext[UserDependencies],
    response: str = Field(description="Complete response to the user query"),
    context_used: bool = Field(description="Whether user context was used in generating the response"),
    news_added: bool = Field(description="Whether news context was added to the response")
) -> AIResponse:
    """Generate a response in the style and voice of the user being impersonated."""
    return AIResponse(
        response=response,
        context_used=context_used,
        news_added=news_added
    )

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
    
    news_background = await search_recent_news(ctx, name)
    general_background = await search_general_info(ctx, name)
    
    user_context["background"] = f"{general_background}\n\n{news_background}"
    
    return {
        "message": f"User context set for {name}",
        "background": user_context["background"]
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



# @near_ai_agent.tool
# async def search_knowledge_base(
#     ctx: RunContext[NearDependencies],
#     query: str,
#     max_results: int = 5
# ) -> str:
#     """Search for relevant information in knowledge base."""
#     # Use ctx.deps.near_client for API calls if needed
#     with ctx.deps.ddgs as ddgs:
#         results = list(ddgs.text(query, max_results=max_results))
    
#     if not results:
#         return ""
    
#     knowledge = "Relevant information:\n"
#     for result in results[:3]:
#         title = result.get('title', 'No title')
#         body = result.get('body', result.get('snippet', 'No content available'))
#         knowledge += f"- {title}: {body}\n"
#     return knowledge

# @near_ai_agent.tool
# async def get_conversation_history(
#     ctx: RunContext[NearDependencies],
#     messages: List[ChatMessage]
# ) -> str:
#     """Get formatted conversation history."""
#     if len(messages) <= 1:
#         return ""
    
#     history = "Previous conversation:\n"
#     for msg in messages[:-1]:
#         history += f"{msg.role}: {msg.content}\n"
#     return history

@app.post("/v1/near/chat/completions")
async def create_near_chat_completion(request: NearChatCompletionRequest):
    try:
        # Add debugging
        print("Request:", request.dict())
        print(f"Requested model: {request.model}")
        print(f"Request configuration: {request.dict()}")
        
        # Create dependencies with the near client
        deps = NearDependencies(
            name=user_context.get("name", ""),
            ddgs=DDGS(),
            openai_client=near_client
        )

        # Get the last message and format messages for context
        messages = request.messages
        last_message = messages[-1].content
        
        # Run the agent with async_api=False to match our client configuration
        result = await near_ai_agent.run(
            last_message,
            deps=deps
        )
        
        print("Raw result:", result)
        print("Result data:", result.data)
        
        # Handle streaming response
        if request.stream:
            return StreamingResponse(
                create_stream_response(result.data.response, request),
                media_type="text/event-stream"
            )


        print(f"trying to format response")
        # Format response
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
            max_tokens=request.max_tokens,
            usage={
                "prompt_tokens": sum(count_tokens(msg.content, request.model) for msg in messages),
                "completion_tokens": count_tokens(result.data.response, request.model),
                "total_tokens": sum(count_tokens(msg.content, request.model) for msg in messages) + 
                               count_tokens(result.data.response, request.model)
            }
        )
        return response

    except Exception as e:
        print(f"Error in chat completion: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/near/models")
async def list_near_models():
    """List all available models from NEAR AI."""
    try:
        # Get the list of models from NEAR AI
        models = await near_client.models.list()
        
        # Format the response
        model_list = []
        for model in models.data:
            model_list.append({
                "id": model.id,
                "created": model.created,
                "object": model.object,
                "owned_by": getattr(model, "owned_by", "unknown")
            })
        
        return {
            "object": "list",
            "data": model_list
        }
    except Exception as e:
        print(f"Error listing NEAR AI models: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)