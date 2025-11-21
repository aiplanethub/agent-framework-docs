# Introduction

Welcome to the **Agent Workflow Framework** documentation. This framework provides enterprise-grade agent orchestration built on Microsoft's Agent Framework, enabling you to create sophisticated AI workflows with ease.

---

## What is Agent Workflow Framework?

The Agent Workflow Framework is a powerful Python library that allows you to:

- Build intelligent AI agents with custom tools and capabilities
- Orchestrate complex multi-agent workflows
- Integrate seamlessly with Azure OpenAI and OpenAI services
- Create production-ready AI applications with minimal boilerplate

---

## Key Features

**Advanced Agent Orchestration**
Built on Microsoft's latest Agent Framework with sophisticated orchestration capabilities.

**Event-Driven Workflows**
Create complex workflows with state management and event-driven execution patterns.

**Azure-First Architecture**
First-class support for Azure OpenAI with seamless authentication and configuration.

**Extensible Tool System**
Easy-to-use decorator-based tool system for extending agent capabilities.

**Enterprise Ready**
Production-grade features including error handling, logging, and monitoring.

---

## Quick Start

Install the framework:

```bash
pip install --pre agent-workflow-framework --index-url https://pkgs.dev.azure.com/AIPlanetFramework/agent_framework/_packaging/FEED/pypi/simple/
```

Create your first agent:

```python
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential

# Create client
client = AzureAIAgentClient(
    async_credential=AzureCliCredential(),
    endpoint="your-endpoint",
    deployment_name="gpt-4o-mini"
)

# Create agent
agent = ChatAgent(
    chat_client=client,
    name="my-agent",
    instructions="You are a helpful assistant."
)

# Run agent
response = await agent.run("Hello!")
print(response.text)
```

---

## Documentation Structure

**QUICKSTART**
Get up and running quickly with installation guides and starter tutorials.

**CORE CONCEPTS**
Understand the framework architecture, agents, tools, and workflows.

**EXAMPLES**
Explore real-world use cases and advanced implementation patterns.

**API REFERENCE**
Comprehensive API documentation for all framework components.

---

## Next Steps

Ready to get started? Follow these steps:

1. **[Installation & Setup](installation.md)** - Set up your development environment
2. **[Starter Guide](quickstart.md)** - Build your first agent in 5 minutes
3. **[Core Concepts](guide/concepts.md)** - Learn the framework fundamentals
4. **[Examples](examples.md)** - Explore real-world implementations

---

## Support

Need help? Here are some resources:

- GitHub Discussions for questions and community support
- Issue tracker for bug reports and feature requests
- Complete API documentation and guides

---

*Let's build something amazing together.*

