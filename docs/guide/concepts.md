# Core Concepts

The Agent Workflow Framework is built on three foundational pillars that work together to enable sophisticated AI agent applications.

## Architecture Overview

The framework uses a **composition-based architecture** that separates concerns into three distinct layers:

1. **Tools Layer**: Self-documenting functions that provide capabilities to agents
2. **Agents Layer**: Specialized AI components that orchestrate tool usage and decision-making
3. **Workflows Layer**: Graph-based orchestration for coordinating multiple agents

## Design Philosophy

### Composition Over Inheritance

Unlike traditional frameworks that require inheriting from base classes, this framework uses composition:

```python
class MyAgent:
    def __init__(self):
        # Your agent HAS-A ChatAgent (composition)
        self._agent = ChatAgent(...)

    async def run(self, query: str) -> str:
        return await self._agent.run(query)
```

**Benefits:**
- Cleaner, more maintainable code
- Easier testing and mocking
- Flexible agent design without framework coupling
- Clear separation of business logic from framework internals

### Explicit Over Implicit

The framework prioritizes explicitness:

- Tools are regular Python functions with clear type annotations
- Agent instructions are written as plain strings
- Workflow graphs are built explicitly with `.add_edge()`
- No hidden magic or auto-discovery mechanisms

### Azure-First Design

Built specifically for Azure AI services:

- Native integration with Azure OpenAI
- Seamless authentication via Azure CLI
- Environment-based configuration
- Enterprise-ready security patterns

## Key Components

### ChatAgent

The core AI component that handles LLM interactions:

- Manages conversation context and history
- Executes tool calls and processes results
- Handles the request/response cycle with the LLM
- Processes structured outputs and streaming

### @ai_function Decorator

Transforms Python functions into AI-callable tools:

- Automatically generates JSON schemas for the LLM
- Uses type annotations and docstrings for schema generation
- Validates inputs and outputs
- Provides error handling context to the LLM

### WorkflowBuilder

Creates directed acyclic graphs (DAGs) for multi-agent orchestration:

- Defines data flow between executors
- Manages state across workflow steps
- Handles parallel and sequential execution
- Enables complex multi-agent collaboration patterns

## Data Flow

```
User Input
    ↓
Workflow.run()
    ↓
Executor 1 → Agent 1 → ChatAgent → LLM
    ↓                       ↓
    |                   Tool Calls
    ↓                       ↓
Executor 2 → Agent 2 → ChatAgent → LLM
    ↓
Final Output
```

## Prerequisites

Before using this framework, ensure you have:

### Required Knowledge
- **Python 3.10+**: Intermediate to advanced proficiency
- **Async/await**: Understanding of Python's asyncio patterns
- **Type annotations**: Familiarity with `typing` module and `Annotated`
- **Environment variables**: Basic knowledge of `.env` files

### Azure Setup
- Active Azure subscription
- Azure OpenAI resource provisioned
- Azure CLI installed and authenticated (`az login`)
- Required environment variables configured:
  - `AZURE_AI_PROJECT_ENDPOINT`
  - `AZURE_AI_MODEL_DEPLOYMENT_NAME`

### Installation
Complete the [Installation & Setup](../installation.md) guide before proceeding.

## Next Steps

Now that you understand the core concepts, proceed to:

- [Working with Tools](tools.md) - Learn to create AI-callable functions
- [Creating Agents](agents.md) - Build specialized AI agents
- [Building Workflows](workflows.md) - Orchestrate multi-agent systems
