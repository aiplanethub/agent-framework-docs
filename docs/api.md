# API Reference

Complete API documentation for the Agent Workflow Framework. This reference covers all core classes, decorators, and methods.

## Core Classes

### ChatAgent

The primary agent implementation that uses a chat client to interact with language models.

#### Constructor

```python
ChatAgent(
    chat_client: ChatClientProtocol,
    instructions: str | None = None,
    *,
    id: str | None = None,
    name: str | None = None,
    description: str | None = None,
    chat_message_store_factory: Callable[[], ChatMessageStoreProtocol] | None = None,
    conversation_id: str | None = None,
    context_providers: ContextProvider | list[ContextProvider] | None = None,
    middleware: list[AgentMiddleware | FunctionMiddleware | ChatMiddleware] | None = None,
    frequency_penalty: float | None = None,
    logit_bias: dict[str | int, float] | None = None,
    max_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
    model_id: str | None = None,
    presence_penalty: float | None = None,
    response_format: type[BaseModel] | None = None,
    seed: int | None = None,
    stop: str | Sequence[str] | None = None,
    store: bool | None = None,
    temperature: float | None = None,
    tool_choice: Literal['auto', 'required', 'none'] | dict[str, Any] | None = 'auto',
    tools: ToolProtocol | Callable | Sequence[ToolProtocol | Callable] | None = None,
    top_p: float | None = None,
    user: str | None = None,
    additional_chat_options: dict[str, Any] | None = None,
    **kwargs: Any
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `chat_client` | `ChatClientProtocol` | **Required.** The chat client to use for the agent |
| `instructions` | `str \| None` | Optional instructions for the agent. These will be added as a system message |
| `id` | `str \| None` | Unique identifier for the agent. Auto-generated if not provided |
| `name` | `str \| None` | The name of the agent |
| `description` | `str \| None` | Brief description of the agent's purpose |
| `chat_message_store_factory` | `Callable \| None` | Factory function to create a ChatMessageStoreProtocol instance |
| `conversation_id` | `str \| None` | Conversation ID for service-managed threads |
| `context_providers` | `ContextProvider \| list[ContextProvider] \| None` | Context providers for agent invocation |
| `middleware` | `list \| None` | Middleware to intercept agent and function invocations |
| `frequency_penalty` | `float \| None` | Penalize frequent tokens (range: -2.0 to 2.0) |
| `logit_bias` | `dict[str \| int, float] \| None` | Modify likelihood of specific tokens |
| `max_tokens` | `int \| None` | Maximum number of tokens to generate |
| `metadata` | `dict[str, Any] \| None` | Additional metadata for the request |
| `model_id` | `str \| None` | The model ID to use for the agent |
| `presence_penalty` | `float \| None` | Penalize tokens based on presence (range: -2.0 to 2.0) |
| `response_format` | `type[BaseModel] \| None` | Structured output format |
| `seed` | `int \| None` | Random seed for deterministic responses |
| `stop` | `str \| Sequence[str] \| None` | Stop sequences for generation |
| `store` | `bool \| None` | Whether to store the response |
| `temperature` | `float \| None` | Sampling temperature (range: 0.0 to 2.0) |
| `tool_choice` | `Literal['auto', 'required', 'none'] \| dict \| None` | Tool selection mode. Default: `'auto'` |
| `tools` | `ToolProtocol \| Callable \| Sequence \| None` | Tools the agent can use |
| `top_p` | `float \| None` | Nucleus sampling probability |
| `user` | `str \| None` | User identifier for tracking |
| `additional_chat_options` | `dict[str, Any] \| None` | Provider-specific parameters |

**Raises:**

- `AgentInitializationError`: If both `conversation_id` and `chat_message_store_factory` are provided

**Example:**

```python
from agent_workflow_framework import ChatAgent
from agent_workflow_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential

client = AzureAIAgentClient(
    async_credential=AzureCliCredential(),
    endpoint="https://your-resource.openai.azure.com/",
    deployment_name="gpt-4"
)

agent = ChatAgent(
    chat_client=client,
    name="assistant",
    description="A helpful assistant",
    instructions="You are a friendly assistant.",
    temperature=0.7,
    max_tokens=500
)
```

#### Methods

##### `run()`

Run the agent with the given messages and options.

```python
async def run(
    messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
    *,
    thread: AgentThread | None = None,
    frequency_penalty: float | None = None,
    logit_bias: dict[str | int, float] | None = None,
    max_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
    model_id: str | None = None,
    presence_penalty: float | None = None,
    response_format: type[BaseModel] | None = None,
    seed: int | None = None,
    stop: str | Sequence[str] | None = None,
    store: bool | None = None,
    temperature: float | None = None,
    tool_choice: Literal['auto', 'required', 'none'] | dict[str, Any] | None = None,
    tools: ToolProtocol | Callable | list[ToolProtocol | Callable] | None = None,
    top_p: float | None = None,
    user: str | None = None,
    additional_chat_options: dict[str, Any] | None = None,
    **kwargs: Any
) -> AgentRunResponse
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | `str \| ChatMessage \| list \| None` | Messages to process. Can be a string, ChatMessage, or list |
| `thread` | `AgentThread \| None` | The thread to use for the agent |

All other parameters override the constructor defaults when provided.

**Returns:**

- `AgentRunResponse`: Contains the agent's response with `.text` property

**Example:**

```python
# Simple string input
response = await agent.run("Hello, how are you?")
print(response.text)

# With ChatMessage objects
from agent_workflow_framework import ChatMessage, ChatRole
message = ChatMessage(ChatRole.User, "Tell me a joke")
response = await agent.run(message)
print(response.text)

# With thread for conversation history
thread = agent.get_new_thread()
response1 = await agent.run("What is Python?", thread=thread)
response2 = await agent.run("Tell me more", thread=thread)
```

##### `run_stream()`

Stream the agent's response with the given messages.

```python
def run_stream(
    messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
    *,
    thread: AgentThread | None = None,
    # ... same parameters as run()
) -> AsyncIterable[AgentRunResponseUpdate]
```

**Returns:**

- `AsyncIterable[AgentRunResponseUpdate]`: Async iterator yielding response updates

**Example:**

```python
async for update in agent.run_stream("Write a story"):
    print(update.text, end="", flush=True)
```

##### `get_new_thread()`

Get a new conversation thread for the agent.

```python
def get_new_thread(
    *,
    service_thread_id: str | None = None,
    **kwargs: Any
) -> AgentThread
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `service_thread_id` | `str \| None` | Optional service-managed thread ID |

**Returns:**

- `AgentThread`: A new thread instance for conversation history

**Example:**

```python
thread = agent.get_new_thread()
response1 = await agent.run("Hello", thread=thread)
response2 = await agent.run("How are you?", thread=thread)
```

##### `as_tool()`

Convert an agent into a tool that can be used by other agents.

```python
def as_tool(
    *,
    name: str | None = None,
    description: str | None = None,
    arg_name: str = 'task',
    arg_description: str | None = None,
    stream_callback: Callable[[AgentRunResponseUpdate], None | Awaitable[None]] | None = None
) -> AIFunction[BaseModel, str]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str \| None` | Tool name. Uses agent's name if None |
| `description` | `str \| None` | Tool description. Uses agent's description if None |
| `arg_name` | `str` | Name of the function argument. Default: `"task"` |
| `arg_description` | `str \| None` | Description for the argument |
| `stream_callback` | `Callable \| None` | Optional callback for streaming responses |

**Returns:**

- `AIFunction[BaseModel, str]`: A tool that other agents can call

**Raises:**

- `TypeError`: If the agent doesn't implement AgentProtocol
- `ValueError`: If the agent tool name cannot be determined

**Example:**

```python
# Create a research agent
research_agent = ChatAgent(
    chat_client=client,
    name="research-agent",
    description="Performs research tasks"
)

# Convert to tool
research_tool = research_agent.as_tool()

# Use in another agent
coordinator = ChatAgent(
    chat_client=client,
    name="coordinator",
    tools=[research_tool]
)
```

##### `as_mcp_server()`

Create an MCP (Model Context Protocol) server from an agent instance.

```python
def as_mcp_server(
    *,
    server_name: str = 'Agent',
    version: str | None = None,
    instructions: str | None = None,
    lifespan: Callable[[Server], AbstractAsyncContextManager] | None = None,
    **kwargs: Any
) -> Server
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `server_name` | `str` | Name of the MCP server. Default: `'Agent'` |
| `version` | `str \| None` | Version of the server |
| `instructions` | `str \| None` | Instructions for the server |
| `lifespan` | `Callable \| None` | Lifespan context manager for the server |

**Returns:**

- `Server`: The MCP server instance

##### `deserialize_thread()`

Deserialize a thread from its serialized state.

```python
async def deserialize_thread(
    serialized_thread: Any,
    **kwargs: Any
) -> AgentThread
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `serialized_thread` | `Any` | The serialized thread data |

**Returns:**

- `AgentThread`: Restored thread instance

##### `to_dict()` / `to_json()`

Serialize the agent to a dictionary or JSON string.

```python
def to_dict(
    *,
    exclude: set[str] | None = None,
    exclude_none: bool = True
) -> dict[str, Any]

def to_json(
    *,
    exclude: set[str] | None = None,
    exclude_none: bool = True
) -> str
```

##### `from_dict()` / `from_json()`

Deserialize an agent from a dictionary or JSON string.

```python
@classmethod
def from_dict(
    value: MutableMapping[str, Any],
    *,
    dependencies: MutableMapping[str, Any] | None = None
) -> ChatAgent

@classmethod
def from_json(
    value: str,
    *,
    dependencies: MutableMapping[str, Any] | None = None
) -> ChatAgent
```

#### Attributes

##### `display_name`

Returns the display name of the agent (name if present, otherwise id).

```python
@property
def display_name(self) -> str
```

---

## Tools and Functions

### @ai_function

Decorator that transforms a Python function into an AI-callable tool.

**Usage:**

```python
from agent_workflow_framework import ai_function
from typing import Annotated
from pydantic import Field

@ai_function
def function_name(
    param: Annotated[type, Field(description="Parameter description")]
) -> return_type:
    """
    Function description that the LLM uses to decide when to call this tool.
    """
    # Implementation
    return result
```

**Features:**

- Automatically generates JSON schema for the LLM
- Uses function name, docstring, and parameter annotations
- Validates inputs and outputs
- Provides error handling context to the LLM

**Example:**

```python
@ai_function
def get_weather(
    location: Annotated[str, Field(description="The city name")],
    unit: Annotated[str, Field(description="Temperature unit")] = "celsius"
) -> str:
    """Get the weather for a location."""
    return f"Weather in {location}: 22°{unit.upper()}"

# Use in agent
agent = ChatAgent(
    chat_client=client,
    tools=[get_weather]
)
```

### AIFunction

A class that wraps a Python function to make it callable by AI models.

#### Constructor

```python
AIFunction(
    *,
    name: str,
    description: str = '',
    approval_mode: Literal['always_require', 'never_require'] | None = None,
    additional_properties: dict[str, Any] | None = None,
    func: Callable[[...], Awaitable[ReturnT] | ReturnT],
    input_model: type[ArgsT],
    **kwargs: Any
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | **Required.** The name of the function |
| `description` | `str` | Description of the function. Default: `""` |
| `approval_mode` | `Literal['always_require', 'never_require'] \| None` | Whether approval is required to run this tool |
| `additional_properties` | `dict[str, Any] \| None` | Additional properties for the function |
| `func` | `Callable` | **Required.** The function to wrap |
| `input_model` | `type[BaseModel]` | **Required.** Pydantic model defining input parameters |

**Example:**

```python
from pydantic import BaseModel, Field
from agent_workflow_framework import AIFunction

class WeatherArgs(BaseModel):
    location: Annotated[str, Field(description="The city name")]
    unit: Annotated[str, Field(description="Temperature unit")] = "celsius"

weather_func = AIFunction(
    name="get_weather",
    description="Get the weather for a location",
    func=lambda location, unit="celsius": f"Weather in {location}: 22°{unit.upper()}",
    approval_mode="never_require",
    input_model=WeatherArgs
)
```

#### Methods

##### `invoke()`

Run the AI function with provided arguments.

```python
async def invoke(
    *,
    arguments: ArgsT | None = None,
    **kwargs: Any
) -> ReturnT
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `arguments` | `BaseModel \| None` | Pydantic model instance with arguments |
| `**kwargs` | `Any` | Keyword arguments (used if `arguments` is None) |

**Returns:**

- `ReturnT`: The result of the function execution

**Raises:**

- `TypeError`: If arguments is not an instance of the expected input model

**Example:**

```python
result = await weather_func.invoke(
    arguments=WeatherArgs(location="Seattle", unit="fahrenheit")
)
```

##### `parameters()`

Get the JSON schema of the function parameters.

```python
def parameters() -> dict[str, Any]
```

**Returns:**

- `dict[str, Any]`: JSON schema for the function's parameters

##### `to_json_schema_spec()`

Convert the function to JSON Schema function specification format.

```python
def to_json_schema_spec() -> dict[str, Any]
```

**Returns:**

- `dict[str, Any]`: Function specification in JSON Schema format

---

## Workflows

### WorkflowBuilder

Creates directed acyclic graphs (DAGs) for multi-agent orchestration.

**Usage:**

```python
from agent_workflow_framework import WorkflowBuilder, executor

workflow = (WorkflowBuilder()
    .add_edge(executor1, executor2)
    .add_edge(executor2, executor3)
    .set_start_executor(executor1)
    .build())
```

#### Methods

##### `add_edge()`

Define data flow from one executor to another.

```python
def add_edge(
    from_executor: Callable,
    to_executor: Callable
) -> WorkflowBuilder
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `from_executor` | `Callable` | Source executor (output provider) |
| `to_executor` | `Callable` | Target executor (input receiver) |

**Returns:**

- `WorkflowBuilder`: Self for method chaining

**Example:**

```python
workflow = (WorkflowBuilder()
    .add_edge(step1, step2)  # step1's output → step2's input
    .add_edge(step2, step3)  # step2's output → step3's input
)
```

##### `set_start_executor()`

Define the entry point of the workflow.

```python
def set_start_executor(
    executor: Callable
) -> WorkflowBuilder
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `executor` | `Callable` | The first executor to run |

**Returns:**

- `WorkflowBuilder`: Self for method chaining

##### `build()`

Compile the workflow into an executable graph.

```python
def build() -> Workflow
```

**Returns:**

- `Workflow`: Compiled workflow ready to run

**Example:**

```python
workflow = (WorkflowBuilder()
    .add_edge(step1, step2)
    .set_start_executor(step1)
    .build())
```

### @executor

Decorator that defines a workflow node (executor).

**Usage:**

```python
from agent_workflow_framework import executor, WorkflowContext
from typing import Annotated
from pydantic import Field

@executor(id="unique_step_id")
async def executor_name(
    input_param: Annotated[type, Field(description="Input description")]
) -> output_type:
    """
    Description of what this executor does.
    """
    # Call agents, process data
    result = await some_agent.run(input_param)
    return result
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `str` | **Required.** Unique identifier for this executor |

**Requirements:**

- Must be an `async` function
- Must use `Annotated[type, Field(description="...")]` for parameters
- Return value is passed to the next executor in the workflow

**Example:**

```python
@executor(id="research_step")
async def research_executor(
    topic: Annotated[str, Field(description="The topic to research")]
) -> dict:
    """Research a topic and gather information."""
    research_notes = await research_agent.run(f"Research: {topic}")
    return {"topic": topic, "notes": research_notes}

@executor(id="writing_step")
async def writing_executor(
    data: Annotated[dict, Field(description="Research data from previous step")]
) -> str:
    """Write a blog post from research notes."""
    blog_post = await writer_agent.run(data["topic"], data["notes"])
    return blog_post
```

### WorkflowContext

Context object for advanced workflow state management.

**Usage in Executors:**

```python
@executor(id="stateful_step")
async def stateful_executor(
    input_data: Annotated[str, Field(description="Input data")],
    ctx: WorkflowContext[str]
) -> str:
    """Executor with workflow context access."""

    # Send messages
    await ctx.send_message("Progress update")

    # Yield intermediate outputs
    await ctx.yield_output("Intermediate result")

    # Return final result
    return "Final result"
```

#### Methods

##### `send_message()`

Send a message to other parts of the workflow.

```python
async def send_message(message: Any) -> None
```

##### `yield_output()`

Yield an intermediate output from the workflow.

```python
async def yield_output(output: Any) -> None
```

### Workflow

The compiled workflow returned by `WorkflowBuilder.build()`.

#### Methods

##### `run()`

Execute the workflow with the given input.

```python
async def run(input_data: Any) -> Any
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_data` | `Any` | Input for the start executor |

**Returns:**

- `Any`: The output from the final executor

**Example:**

```python
workflow = (WorkflowBuilder()
    .add_edge(research_executor, writing_executor)
    .set_start_executor(research_executor)
    .build())

result = await workflow.run("AI trends in 2025")
print(result)  # Final blog post
```

---

## Azure Integration

### AzureAIAgentClient

Client for Azure AI services with Agent Framework.

**Constructor:**

```python
from agent_workflow_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential

client = AzureAIAgentClient(
    async_credential: AsyncTokenCredential,
    endpoint: str,
    deployment_name: str
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `async_credential` | `AsyncTokenCredential` | Azure credential (e.g., `AzureCliCredential()`) |
| `endpoint` | `str` | Azure AI project endpoint URL |
| `deployment_name` | `str` | Model deployment name |

**Example:**

```python
import os
from agent_workflow_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential

client = AzureAIAgentClient(
    async_credential=AzureCliCredential(),
    endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
    deployment_name=os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")
)
```

### AzureOpenAIResponsesClient

Client for Azure OpenAI Responses API.

```python
from agent_workflow_framework.azure import AzureOpenAIResponsesClient

agent = AzureOpenAIResponsesClient(
    endpoint: str | None = None,
    deployment_name: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    credential: AsyncTokenCredential | None = None
).create_agent(
    name: str,
    instructions: str
)
```

**Example:**

```python
from agent_workflow_framework.azure import AzureOpenAIResponsesClient
from azure.identity.aio import AzureCliCredential

agent = AzureOpenAIResponsesClient(
    credential=AzureCliCredential()
).create_agent(
    name="HaikuBot",
    instructions="You are an upbeat assistant that writes beautifully."
)

print(await agent.run("Write a haiku about AI."))
```

---

## Response Objects

### AgentRunResponse

Response object returned by `ChatAgent.run()`.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `text` | `str` | The text response from the agent |
| `thread_id` | `str \| None` | The thread ID if conversation was stored |
| `messages` | `list[ChatMessage]` | All messages in the conversation |

**Example:**

```python
response = await agent.run("Hello!")
print(response.text)
print(f"Thread ID: {response.thread_id}")
```

### AgentRunResponseUpdate

Update object yielded by `ChatAgent.run_stream()`.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Incremental text update |
| `is_complete` | `bool` | Whether this is the final update |

**Example:**

```python
async for update in agent.run_stream("Tell me a story"):
    print(update.text, end="", flush=True)
    if update.is_complete:
        print("\n[Done]")
```

### AgentThread

Represents a conversation thread for maintaining history.

**Creation:**

```python
thread = agent.get_new_thread()
```

**Usage:**

```python
# First message
response1 = await agent.run("What is Python?", thread=thread)

# Follow-up message (has context)
response2 = await agent.run("Tell me more", thread=thread)
```

---

## Exception Classes

### AgentInitializationError

Raised when there's an error initializing an agent.

**Common causes:**

- Both `conversation_id` and `chat_message_store_factory` provided
- Invalid configuration parameters

**Example:**

```python
try:
    agent = ChatAgent(
        chat_client=client,
        conversation_id="conv-123",
        chat_message_store_factory=my_factory  # ❌ Can't use both
    )
except AgentInitializationError as e:
    print(f"Initialization failed: {e}")
```

---

## Type Aliases and Enums

### ChatRole

Enum for message roles.

```python
from agent_workflow_framework import ChatRole

ChatRole.User       # User message
ChatRole.Assistant  # Assistant message
ChatRole.System     # System message
ChatRole.Tool       # Tool response message
```

### ChatMessage

Represents a chat message.

```python
from agent_workflow_framework import ChatMessage, ChatRole

message = ChatMessage(
    role: ChatRole,
    content: str | list[ContentPart]
)
```

**Example:**

```python
user_message = ChatMessage(ChatRole.User, "Hello!")
system_message = ChatMessage(ChatRole.System, "You are a helpful assistant.")
```

---

## Environment Variables

The framework uses these environment variables for Azure integration:

| Variable | Description | Example |
|----------|-------------|---------|
| `AZURE_AI_PROJECT_ENDPOINT` | Azure AI project endpoint URL | `https://my-project.openai.azure.com/` |
| `AZURE_AI_MODEL_DEPLOYMENT_NAME` | Model deployment name | `gpt-4` or `gpt-4o-mini` |
| `AZURE_OPENAI_API_VERSION` | API version (optional) | `2024-10-01-preview` |
| `AZURE_OPENAI_API_KEY` | API key (if not using Azure CLI auth) | `your-api-key` |

**Setup:**

Create a `.env` file:

```
AZURE_AI_PROJECT_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_AI_MODEL_DEPLOYMENT_NAME=gpt-4
```

Load in code:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Complete Example

```python
import asyncio
import os
from dotenv import load_dotenv
from agent_workflow_framework import ChatAgent, ai_function, WorkflowBuilder, executor
from agent_workflow_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from typing import Annotated
from pydantic import Field

# Load environment
load_dotenv()

# Define a tool
@ai_function
def get_weather(
    location: Annotated[str, Field(description="City name")]
) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

# Create client
client = AzureAIAgentClient(
    async_credential=AzureCliCredential(),
    endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
    deployment_name=os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")
)

# Create agent
agent = ChatAgent(
    chat_client=client,
    name="weather-assistant",
    instructions="You are a helpful weather assistant.",
    tools=[get_weather],
    temperature=0.7
)

# Define workflow executors
@executor(id="get_weather_info")
async def weather_executor(
    city: Annotated[str, Field(description="City to check")]
) -> str:
    """Get weather for a city."""
    response = await agent.run(f"What's the weather in {city}?")
    return response.text

# Build workflow
workflow = (WorkflowBuilder()
    .set_start_executor(weather_executor)
    .build())

# Run
async def main():
    result = await workflow.run("Seattle")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Version Information

This API reference is based on **Agent Framework version 1.0.0b251016** (Python).

---

## See Also

- [Core Concepts](guide/concepts.md) - Framework architecture and design
- [Creating Agents](guide/agents.md) - Agent development guide
- [Working with Tools](guide/tools.md) - Tool creation patterns
- [Building Workflows](guide/workflows.md) - Workflow orchestration
- [Examples](examples.md) - Real-world usage examples
