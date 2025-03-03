# NearAI PydanticAI Agent

## Overview
This repository contains a FastAPI-based Agent that integrates with **Near AI's** inference API using `pydantic-ai` and `OpenAIModel`. It leverages **FastAPI**, **DuckDuckGo Search**, and **NEAR AI's** Llama-based models for enhanced AI interactions. The chatbot provides real-time AI-generated responses, impersonation capabilities, and tools for retrieving recent news or general information.

## Features
- 🚀 **FastAPI-powered**: High-performance, asynchronous chatbot API.
- 🤖 **NEAR AI Integration**: Uses **Fireworks Llama-v3p1-405B** or **Llama-v3p3-70B** models.
- 🔍 **Live Information Retrieval**: Retrieves recent news and general info using **DuckDuckGo Search**.
- 🔄 **Streaming Responses**: Supports real-time token streaming for chat completions.
- 🛠 **Custom AI Agent**: Built with `pydantic-ai` for structured AI interactions.
- 🏦 **Multi-User Context**: Impersonates users and tailors responses based on context.

## Installation
### Prerequisites
Ensure you have Python 3.11 installed.

```sh
pip install -r ./requirements.txt
```

### Configuration
A `config.json` file is required to authenticate with **NEAR AI**.
Generate it with `nearai login`
Then `cp ~/.nearai/config.json .`


## Running the API
Start the FastAPI server with:

```sh
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

You can view all endpoints at http://localhost:8000/docs

### 1. Chat Completion
**Endpoint:** `/v1/chat/completions`  
**Method:** `POST`

**Request:**
```json
{
  "model": "fireworks::accounts/fireworks/models/llama-v3p1-405b-instruct",
  "messages": [
    {"role": "user", "content": "Tell me about NEAR AI"}
  ],
  "temperature": 0.7,
  "max_tokens": 2000,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-12345",
  "model": "fireworks::llama-v3p1-405b-instruct",
  "choices": [
    { "index": 0, "message": { "role": "assistant", "content": "NEAR AI is..." } }
  ],
  "usage": { "total_tokens": 250 }
}
```

### 2. Set User Context
**Endpoint:** `/set-user`  
**Method:** `POST`

**Request:**
```json
{
  "name": "Michael Gold SVA"
}
```

**Response:**
```json
{
  "message": "User context set for Michael Gold",
  "background": "Michael Gold is an entrepreneur and professor at SVA..."
}
```

### 3. List NEAR AI Models
**Endpoint:** `/v1/near/models`  
**Method:** `GET`

**Response:**
```json
{
  "object": "list",
  "data": [
    { "id": "fireworks::llama-v3p1-405b-instruct", "created": 1678901234 },
    { "id": "fireworks::llama-v3p3-70b-instruct", "created": 1678905678 }
  ]
}
```

## Streaming Responses
If `stream: true` is set in chat completion requests, the server will return chunked responses via `text/event-stream`.

Example stream output:
```
data: {"id":"chatcmpl-12345", "content": "Hello "}

data: {"id":"chatcmpl-12345", "content": "world!"}

data: [DONE]
```

## Deployment
Deploy using Docker:
```sh
docker build -t nearai-chatbot .
docker run -p 8000:8000 nearai-chatbot
```

## Contributing
- Submit PRs for bug fixes or new features.
- Open issues for discussions or questions.

## License
Apache License

---

🔥 **Built with FastAPI, NEAR AI, and love!**

