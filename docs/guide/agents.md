# Creating Agents

Agents are specialized AI components that use tools to accomplish tasks. This framework uses a **composition pattern** to make agents flexible, testable, and maintainable.

## Understanding the Composition Pattern

### Traditional Inheritance (Not Used Here)

```python
# ❌ NOT how this framework works
class MyAgent(BaseAgent):  # Inheriting from framework
    def __init__(self):
        super().__init__()
```

### Composition Pattern (Framework Approach)

```python
# ✅ How this framework works
class MyAgent:  # Regular Python class
    def __init__(self):
        self._agent = ChatAgent(...)  # HAS-A ChatAgent
```

**Benefits:**
- Your agent class is just a container for logic
- No framework coupling in your business code
- Easy to test and mock
- Clear separation of concerns

## Creating Your First Agent

### Basic Structure

Every agent follows this pattern:

```python
from agent_workflow_framework import ChatAgent
from agent_workflow_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
import os

class MyAgent:
    def __init__(self):
        self.name = "MyAgent"

        # 1. Define instructions (the agent's "personality")
        instructions = """Clear instructions for the agent."""

        # 2. Define tools the agent can use
        self.tools = [tool1, tool2]

        # 3. Set up Azure AI client
        self.client = AzureAIAgentClient(
            async_credential=AzureCliCredential(),
            endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
            deployment_name=os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")
        )

        # 4. Compose the internal ChatAgent
        self._agent = ChatAgent(
            chat_client=self.client,
            instructions=instructions,
            tools=self.tools
        )

    async def run(self, query: str) -> str:
        """Public interface for running the agent."""
        response = await self._agent.run(query)
        return response.text
```

## Complete Example: Research Agent

Create a file `agents.py`:

```python
from agent_workflow_framework import ChatAgent
from agent_workflow_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
import os

# Import the tools we defined earlier
from src.agent_workflow_framework.tools.web_tools import web_search, read_url

class ResearchAgent:
    """
    A specialized agent for performing web research and summarizing content.
    Its public interface is a single `run` method.
    """

    def __init__(self):
        self.name = "ResearchAgent"

        # 1. Define the agent's "personality" and purpose.
        # This is a critical piece of prompt engineering.
        instructions = """You are a world-class research assistant. 
        Your job is to find information using the 'web_search' tool and then 
        read promising URLs using the 'read_url' tool.

        Your research process:
        1. Use web_search to find relevant sources
        2. Use read_url to read the most promising URLs
        3. Synthesize the information into a clear, concise summary

        You must be concise and factual. Do not make up information.
        If you cannot find an answer, state that clearly.
        Always cite your sources by mentioning the URLs you read."""

        # 2. Define the list of tool functions this agent can use
        self.tools = [web_search, read_url]

        # 3. Set up the LLM client
        # AzureCliCredential() automatically and securely finds your
        # logged-in Azure credentials.
        # AzureAIAgentClient reads endpoint/deployment from .env variables.
        self.client = AzureAIAgentClient(
            async_credential=AzureCliCredential(),
            endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
            deployment_name=os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")
        )

        # 4. Compose the internal ChatAgent
        # This is the "brain" of our agent. We pass it the client,
        # our instructions, and the tools it's allowed to use.
        self._agent = ChatAgent(
            chat_client=self.client,
            instructions=instructions,
            tools=self.tools
        )

    async def run(self, query: str) -> str:
        """
        Runs the agent with a specific query. This is the single, 
        clean entry point for our agent.
        """
        print(f"\n--- [{self.name}] Received query: {query} ---")

        # 5. Delegate the work to the internal ChatAgent
        # The .run() method handles the entire conversation loop:
        # -> sending the query,
        # -> deciding to call a tool,
        # -> executing the tool,
        # -> sending the tool's result back,
        # -> and generating a final text response.
        response = await self._agent.run(query)

        print(f"--- [{self.name}] Responding ---")
        return response.text
```

## Key Components Explained

### 1. Instructions (Prompt Engineering)

The `instructions` parameter is the most critical part of your agent. It defines:

- **Role**: What is the agent's expertise?
- **Capabilities**: What tools can it use?
- **Process**: How should it approach tasks?
- **Constraints**: What should it avoid?
- **Output format**: How should it respond?

#### Example: Structured Instructions

```python
instructions = """You are a professional data analyst specializing in financial reports.

**Capabilities:**
- Analyze CSV files using the analyze_csv tool
- Calculate statistics using built-in Python functions
- Generate insights from numerical data

**Process:**
1. Load the data using the provided file path
2. Perform statistical analysis
3. Identify key trends and anomalies
4. Present findings in a clear, bullet-point format

**Constraints:**
- Never make up data or statistics
- Always cite the source file in your analysis
- If data is incomplete, clearly state the limitations

**Output Format:**
- Start with a brief summary (2-3 sentences)
- List key findings as bullet points
- End with actionable recommendations"""
```

### 2. Tools Selection

Choose tools that match your agent's purpose:

```python
# Research agent: web tools
self.tools = [web_search, read_url]

# Data agent: file and analysis tools
self.tools = [read_file, analyze_csv, write_report]

# General assistant: no tools (pure LLM reasoning)
self.tools = []
```

### 3. Azure Client Setup

The client handles authentication and API calls:

```python
from azure.identity.aio import AzureCliCredential
from agent_workflow_framework.azure import AzureAIAgentClient

# Automatically uses your `az login` credentials
client = AzureAIAgentClient(
    async_credential=AzureCliCredential(),
    endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
    deployment_name=os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")
)
```

**Environment variables required:**
```bash
AZURE_AI_PROJECT_ENDPOINT=https://your-project.openai.azure.com/
AZURE_AI_MODEL_DEPLOYMENT_NAME=gpt-4
```

### 4. The `run()` Method

Your agent's public interface should be clean and simple:

```python
async def run(self, query: str) -> str:
    """
    Processes a user query and returns the agent's response.

    Args:
        query: The user's input or question

    Returns:
        The agent's text response after using tools and reasoning
    """
    response = await self._agent.run(query)
    return response.text
```

## Creating Specialized Agents

### Example: Writer Agent (No Tools)

Some agents don't need tools—they use pure LLM reasoning:

```python
class WriterAgent:
    """
    A specialized agent for writing high-quality blog posts.
    No tools needed—uses LLM reasoning and generation only.
    """
    def __init__(self):
        self.name = "WriterAgent"

        instructions = """You are a professional blog post writer with expertise 
        in technical content.

        When given a topic and research notes:
        1. Analyze the key points from the research
        2. Structure a compelling narrative
        3. Write a 3-paragraph blog post using Markdown formatting
        4. Use engaging language while staying factual
        5. Include a catchy opening and strong conclusion

        Output Format:
        - Use ## for the title
        - Use **bold** for key terms
        - Keep paragraphs concise (3-4 sentences each)"""

        self.client = AzureAIAgentClient(
            async_credential=AzureCliCredential(),
            endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
            deployment_name=os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")
        )

        # No tools needed—pure LLM generation
        self._agent = ChatAgent(
            chat_client=self.client,
            instructions=instructions,
            tools=[]
        )

    async def run(self, topic: str, research_notes: str) -> str:
        """
        Generates a blog post from a topic and research notes.
        """
        prompt = f"""Topic: {topic}

Research Notes:
{research_notes}

Write the blog post:"""

        print(f"\n--- [{self.name}] Received prompt ---")
        response = await self._agent.run(prompt)
        print(f"--- [{self.name}] Responding ---")

        return response.text
```

### Example: Data Analyst Agent

```python
from src.agent_workflow_framework.tools.data_tools import read_file, analyze_csv, write_file

class DataAnalystAgent:
    """Agent specialized in analyzing CSV data and generating reports."""

    def __init__(self):
        self.name = "DataAnalystAgent"

        instructions = """You are a data analyst specializing in CSV analysis.

        **Process:**
        1. Read the CSV file using read_file tool
        2. Analyze it using analyze_csv tool
        3. Generate insights about trends, averages, and anomalies
        4. Write a report using write_file tool

        Always include:
        - Summary statistics
        - Key trends
        - Notable outliers
        - Actionable recommendations"""

        self.tools = [read_file, analyze_csv, write_file]

        self.client = AzureAIAgentClient(
            async_credential=AzureCliCredential(),
            endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
            deployment_name=os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")
        )

        self._agent = ChatAgent(
            chat_client=self.client,
            instructions=instructions,
            tools=self.tools
        )

    async def run(self, csv_path: str) -> str:
        query = f"Analyze the CSV file at {csv_path} and generate a report."
        response = await self._agent.run(query)
        return response.text
```

## Testing Your Agent

### Simple Test Script

```python
import asyncio
import os
from dotenv import load_dotenv
from agents import ResearchAgent

async def test_research_agent():
    load_dotenv()

    agent = ResearchAgent()
    result = await agent.run("What is the Agent Framework?")

    print("\n=== Agent Response ===")
    print(result)

if __name__ == "__main__":
    asyncio.run(test_research_agent())
```

### Expected Output

```
--- [ResearchAgent] Received query: What is the Agent Framework? ---
[Tool Call] Searching for: Agent Framework
[Tool Call] Reading URL: https://example.com/agent-framework
--- [ResearchAgent] Responding ---

=== Agent Response ===
The Agent Framework is a comprehensive platform for building
enterprise-grade AI agents. It provides advanced orchestration capabilities,
robust tool integration, and Azure-native deployment options.

Sources:
- https://example.com/agent-framework
```

## Best Practices

### 1. Single Responsibility

Each agent should have one clear purpose:

❌ **Bad:** Generic "do everything" agent
```python
class UniversalAgent:
    def __init__(self):
        self.tools = [web_search, read_file, analyze_csv, write_code, ...]
```

✅ **Good:** Specialized agents
```python
class ResearchAgent:  # Web research only
class DataAgent:      # Data analysis only
class WriterAgent:    # Content generation only
```

### 2. Clear Instructions

Be explicit about the agent's behavior:

❌ **Bad:** Vague instructions
```python
instructions = "You are a helpful assistant."
```

✅ **Good:** Detailed instructions
```python
instructions = """You are a research assistant specializing in academic papers.

Process:
1. Search for relevant papers
2. Read abstracts
3. Summarize key findings
4. Cite all sources

Never make up citations or fabricate research."""
```

### 3. Resource Management

For production use, implement proper resource cleanup:

```python
class ResearchAgent:
    def __init__(self):
        self._agent_instance = None
        self._credential = None

    async def _get_agent(self):
        if self._agent_instance is None:
            self._credential = AzureCliCredential()
            client = AzureAIAgentClient(
                async_credential=self._credential,
                endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
                deployment_name=os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")
            )
            self._agent_instance = ChatAgent(
                chat_client=client,
                instructions=self.instructions,
                tools=self.tools
            )
        return self._agent_instance

    async def run(self, query: str) -> str:
        agent = await self._get_agent()
        response = await agent.run(query)
        return response.text

    async def cleanup(self):
        """Clean up resources."""
        if self._credential:
            await self._credential.close()
```

### 4. Error Handling

Handle failures gracefully:

```python
async def run(self, query: str) -> str:
    try:
        response = await self._agent.run(query)
        return response.text
    except Exception as e:
        error_msg = f"Agent error: {str(e)}"
        print(error_msg)
        return f"I encountered an error: {error_msg}"
```

## Common Issues & Troubleshooting

### Agent Not Using Tools

**Problem:** Agent responds without calling any tools.

**Solutions:**
- Make instructions more explicit about when to use tools
- Ensure tools are properly registered in `self.tools`
- Check that tool docstrings clearly describe their purpose

### Authentication Errors

**Problem:** `AzureCliCredential` fails.

**Solutions:**
```bash
# Re-authenticate with Azure CLI
az login
az account set --subscription <your-subscription-id>

# Verify environment variables
echo $AZURE_AI_PROJECT_ENDPOINT
echo $AZURE_AI_MODEL_DEPLOYMENT_NAME
```

### Incomplete Responses

**Problem:** Agent stops mid-response or doesn't finish tasks.

**Solutions:**
- Increase token limits if available
- Simplify the task or break it into steps
- Make instructions more focused

## Next Steps

Now that you understand agents, proceed to:

- [Building Workflows](workflows.md) - Orchestrate multiple agents together
- [API Reference](../api-reference.md) - Detailed API documentation
- [Examples](../examples.md) - More complex agent patterns
