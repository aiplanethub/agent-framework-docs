# Agent Workflow Framework

A sophisticated agent workflow framework built on Microsoft's Agent Framework, offering enterprise-grade orchestration with advanced RAG capabilities and comprehensive tool integration.

## Features

- **Advanced Agent Orchestration**: Built on Microsoft's latest Agent Framework
- **Event-Driven Workflow Engine**: Sophisticated workflow execution with state management
- **Azure-First Architecture**: First-class support for Azure OpenAI
- **Enhanced Tool System**: Extensible tool framework with built-in enterprise tools
- **Advanced RAG Integration**: Robust knowledge management with hybrid search capabilities

## Quick Links

- [Installation & Setup](installation.md) - Get started with the framework
- [Quick Start](quickstart.md) - Build your first agent in 5 minutes
- [API Reference](api.md) - Complete API documentation
- [Examples](examples.md) - Real-world usage examples

## Getting Started

Install from Azure Artifacts:

```python
pip install --index-url "https://pkgs.dev.azure.com/AIPlanetFramework/agent_framework/_packaging/FEED/pypi/simple/" --extra-index-url "https://pypi.org/simple" agent_workflow
```

Create your first agent:

```sh
from agent_workflow_framework import WorkflowEngine, create_assistant

engine = WorkflowEngine()
agent = create_assistant("my_assistant", description="A helpful assistant")
```

Check out the [Installation & Setup Guide](installation.md) for detailed instructions.

