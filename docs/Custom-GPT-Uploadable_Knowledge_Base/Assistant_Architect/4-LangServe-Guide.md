# LangServe Guide - Efficient LLM deployment

## Overview: Creating Server & Client Python modules with LangServe
`LangServe` helps developers deploy `LangChain` [runnables and chains](https://python.langchain.com/docs/expression_language/) as a REST API.

This library is integrated with [FastAPI](https://fastapi.tiangolo.com/) and uses [pydantic](https://docs.pydantic.dev/latest/) for data validation.

In addition, it provides a client that can be used to call into runnables deployed on a server. A javascript client is available in [LangChainJS](https://js.langchain.com/docs/api/runnables_remote/classes/RemoteRunnable).

### Features
- Input and Output schemas automatically inferred from your LangChain object, and enforced on every API call, with rich error messages
- API docs page with JSONSchema and Swagger (insert example link)
- Efficient `/invoke`, `/batch` and `/stream` endpoints with support for many concurrent requests on a single server
- `/stream_log` endpoint for streaming all (or some) intermediate steps from your chain/agent
- Playground page at `/playground` with streaming output and intermediate steps
- Built-in (optional) tracing to [LangSmith](https://www.langchain.com/langsmith), just add your API key (see [Instructions](https://docs.smith.langchain.com/)
- All built with battle-tested open-source Python libraries like FastAPI, Pydantic, uvloop and asyncio.
- Use the client SDK to call a LangServe server as if it was a Runnable running locally (or call the HTTP API directly)
- [LangServe Hub](https://github.com/langchain-ai/langchain/blob/master/templates/README.md)

### Installation:
For both client and server:
`pip install langserve[all]`
or `pip install "langserve[client]"` for client code, and `pip install "langserve[server]"` for server code.

## LangChain CLI 🛠️
Use the `LangChain` CLI to bootstrap a `LangServe` project quickly.

To use the langchain CLI make sure that you have a recent version of `langchain-cli` installed. You can install it with `pip install -U langchain-cli.`

`langchain app new ../path/to/directory`

#### Examples
Get your LangServe instance started quickly with [LangChain Templates](https://github.com/langchain-ai/langchain/blob/master/templates/README.md "LangChain templates").

For more examples, see the templates [index](https://github.com/langchain-ai/langchain/blob/master/templates/docs/INDEX.md) or the [examples](https://github.com/langchain-ai/langserve/tree/main/examples "Examples") directory.

#### Server
Here's a server that deploys an OpenAI chat model, an Anthropic chat model, and a chain that uses the Anthropic model to tell a joke about a topic.

```
#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langserve import add_routes


app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai",
)

add_routes(
    app,
    ChatAnthropic(),
    path="/anthropic",
)

model = ChatAnthropic()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(
    app,
    prompt | model,
    path="/joke",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
```

#### Docs
If you've deployed the server above, you can view the generated OpenAPI docs using:

⚠️ If using pydantic v2, docs will not be generated for invoke/batch/stream/stream_log. See Pydantic section below for more details.

`curl localhost:8000/docs`

make sure to add the /docs suffix.

⚠️ Index page / is not defined by design, so curl localhost:8000 or visiting the URL will return a 404. If you want content at / define an endpoint @app.get("/").

#### Client
Python SDK

```
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langserve import RemoteRunnable

openai = RemoteRunnable("http://localhost:8000/openai/")
anthropic = RemoteRunnable("http://localhost:8000/anthropic/")
joke_chain = RemoteRunnable("http://localhost:8000/joke/")

joke_chain.invoke({"topic": "parrots"})

# or async
await joke_chain.ainvoke({"topic": "parrots"})

prompt = [
    SystemMessage(content='Act like either a cat or a parrot.'),
    HumanMessage(content='Hello!')
]

# Supports astream
async for msg in anthropic.astream(prompt):
    print(msg, end="", flush=True)

prompt = ChatPromptTemplate.from_messages(
    [("system", "Tell me a long story about {topic}")]
)

# Can define custom chains
chain = prompt | RunnableMap({
    "openai": openai,
    "anthropic": anthropic,
})

chain.batch([{ "topic": "parrots" }, { "topic": "cats" }])
```

In TypeScript (requires LangChain.js version 0.0.166 or later):

```
import { RemoteRunnable } from "langchain/runnables/remote";

const chain = new RemoteRunnable({
  url: `http://localhost:8000/joke/`,
});
const result = await chain.invoke({
  topic: "cats",
});
```

Python using requests:

```
import requests
response = requests.post(
    "http://localhost:8000/joke/invoke/",
    json={'input': {'topic': 'cats'}}
)
response.json()
```

You can also use curl:

```
curl --location --request POST 'http://localhost:8000/joke/invoke/' \
    --header 'Content-Type: application/json' \
    --data-raw '{
        "input": {
            "topic": "cats"
        }
    }'
```

#### Endpoints:
The following code:

```
...
add_routes(
  app,
  runnable,
  path="/my_runnable",
)
```

adds of these endpoints to the server:

- `POST /my_runnable/invoke` - invoke the runnable on a single input
- `POST /my_runnable/batch` - invoke the runnable on a batch of inputs
- `POST /my_runnable/stream` - invoke on a single input and stream the output
- `POST /my_runnable/stream_log` - invoke on a single input and stream the output, including output of intermediate steps as it's generated
- `GET /my_runnable/input_schema` - json schema for input to the runnable
- `GET /my_runnable/output_schema` - json schema for output of the runnable
- `GET /my_runnable/config_schema` - json schema for config of the runnable
These endpoints match the LangChain Expression Language interface -- please reference this documentation for more details.

### Basic Deployment and Querying with GPT-3.5-Turbo
- **Example**: Deploying and querying the GPT-3.5-Turbo model using LangServe.
- **Objective**: To illustrate the use of LangServe within the LangChain ecosystem. LangServe is designed to facilitate server-side functionalities for managing and deploying language models, making it an essential tool for scalable and efficient AI applications.
```python
from langserve import LangServeClient

# Initialize the LangServe client
langserve_client = LangServeClient(api_url="https://api.langserve.com")

# Deploying the GPT-3.5-Turbo model
model_config = {
    "model_name": "gpt-3.5-turbo-1106",
    "description": "GPT-3.5 Turbo model for general-purpose use"
}
deployment_response = langserve_client.deploy_model(model_config)
print("Deployment Status:", deployment_response.status)

# Sending a query to the deployed model
query = "Explain the concept of machine learning in simple terms."
response = langserve_client.query_model(model_name="gpt-3.5-turbo-1106", query=query)
print("Model Response:", response.content)
```

#### Advanced Deployment and Custom Configuration
- **Example**: Utilizing LangServe for deploying custom-configured models for specialized tasks.
```python
# Custom deployment with specific parameters
advanced_model_config = {
    "model_name": "custom-gpt-model",
    "description": "A custom-configured GPT model for specialized tasks",
    "parameters": {
        "temperature": 0.7,
        "max_tokens": 150
    }
}
langserve_client.deploy_model(advanced_model_config)

# Querying the custom model
custom_query = "Generate a technical summary of quantum computing."
custom_response = langserve_client.query_model(model_name="custom-gpt-model", query=custom_query)
print("Custom Model Response:", custom_response.content)
```

#### Model Management and Analytics
- **Example**: Managing deployed models and accessing detailed analytics.
```python
# Fetching model analytics
model_analytics = langserve_client.get_model_analytics(model_name="gpt-3.5-turbo-1106")
print("Model Usage Analytics:", model_analytics)

# Updating a deployed model's configuration
update_config = {
    "temperature": 0.5,
    "max_tokens": 200
}
langserve_client.update_model_config(model_name="gpt-3.5-turbo-1106", new_config=update_config)

# Retrieving updated model details
updated_model_details = langserve_client.get_model_details(model_name="gpt-3.5-turbo-1106")
print("Updated Model Details:", updated_model_details)
```

#### Integration with LangChain Applications
- **Example**: Demonstrating seamless integration of LangServe with LangChain.
```python
from langchain.chains import SimpleChain

# Building a SimpleChain with a LangServe deployed model
chain = SimpleChain(model_name="gpt-3.5-turbo-1106", langserve_client=langserve_client)

# Executing the chain with a user query
chain_response = chain.execute("What are the latest trends in AI?")
print("Chain Response using LangServe Model:", chain_response)
```

---
