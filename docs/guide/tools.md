# Working with Tools

Tools are the foundation of an agent's capabilities. In this framework, tools are simple Python functions enhanced with the `@ai_function` decorator.

## What Are Tools?

Tools are Python functions that agents can call to perform specific tasks:

- **Web searches**: Find information online
- **File operations**: Read, write, or process files
- **API calls**: Interact with external services
- **Data processing**: Transform or analyze data
- **Custom logic**: Any Python function you can write

## Creating Your First Tool

### Basic Structure

Every tool follows this pattern:

```python
from agent_workflow_framework import ai_function
from typing import Annotated
from pydantic import Field

@ai_function
def tool_name(
    param: Annotated[type, Field(description="Clear description")]
) -> return_type:
    """
    Describe what this tool does and when the agent should use it.
    """
    # Implementation
    return result
```

### Example: Web Search Tool

Create a file at `src/agent_workflow_framework/tools/web_tools.py`:

```python
from agent_workflow_framework import ai_function
from pydantic import Field
from typing import Annotated
import httpx

@ai_function
def web_search(
    query: Annotated[str, Field(description="The precise search query for Google.")]
) -> str:
    """
    Performs a web search using the provided query and returns a list of top results.
    Use this tool when the user asks for current information or facts from the internet.
    """
    print(f"[Tool Call] Searching for: {query}")

    # This is a mock search for demonstration.
    # In a real app, you would use an actual search API.
    mock_results = {
        "AI framework": "Result: Agent Framework is a new platform...",
        "python": "Result: Python is an interpreted, high-level programming language..."
    }

    return mock_results.get(query, f"No results found for '{query}'.")
```

### Example: URL Reader Tool

```python
@ai_function
def read_url(
    url: Annotated[str, Field(description="The full URL of the website to read.")]
) -> str:
    """
    Reads the full content from a specific URL and returns it as a string.
    Use this tool to fetch and read web page content when given a URL.
    Includes error handling for bad requests.
    """
    print(f"[Tool Call] Reading URL: {url}")
    try:
        response = httpx.get(url, follow_redirects=True, timeout=10.0)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx responses
        # Return first 1000 chars to avoid oversized context
        return response.text[:1000] + "..."
    except httpx.RequestError as e:
        return f"Error reading URL: {e}"
    except httpx.HTTPStatusError as e:
        return f"Error: Received status {e.response.status_code} from {e.request.url}"
```

## How Tools Work

### Schema Generation

The framework automatically generates a JSON schema for the LLM based on:

1. **Function name**: `web_search` becomes the tool identifier
2. **Docstring**: Tells the LLM when to use the tool
3. **Parameter annotations**: `Annotated[str, Field(description="...")]` describes each parameter
4. **Return type**: Defines what the tool returns

### Execution Flow

```
1. Agent receives user query
2. LLM decides which tool(s) to call
3. Framework validates parameters
4. Tool function executes
5. Result returns to LLM as context
6. LLM generates final response
```

## Best Practices

### 1. Descriptions Are Critical

❌ **Bad:**
```python
def search(q: str) -> str:
    """Search."""
    pass
```

✅ **Good:**
```python
@ai_function
def web_search(
    query: Annotated[str, Field(description="The precise search query including relevant keywords")]
) -> str:
    """
    Performs a web search using the provided query and returns top results.
    Use this when the user asks for current information or facts.
    """
    pass
```

**Why:** The LLM uses descriptions to decide what data to pass and when to call the tool.

### 2. Write Clear Docstrings

The docstring should answer:
- **What does this tool do?**
- **When should the agent use it?**
- **What kind of results does it return?**

✅ **Good example:**
```python
"""
Fetches the current stock price for a given ticker symbol.
Use this tool when the user asks for real-time stock market data.
Returns the price as a decimal number with currency symbol.
"""
```

### 3. Single Responsibility Principle

Keep tools focused on one task:

❌ **Bad:** One giant "research" tool
```python
@ai_function
def research(query: str, read_urls: bool, summarize: bool) -> str:
    # Too many responsibilities
    pass
```

✅ **Good:** Separate, focused tools
```python
@ai_function
def web_search(query: str) -> str:
    """Find relevant URLs."""
    pass

@ai_function
def read_url(url: str) -> str:
    """Read content from a URL."""
    pass

@ai_function
def summarize_text(text: str) -> str:
    """Summarize long text."""
    pass
```

**Why:** This gives the LLM more flexibility and control over the research process.

### 4. Handle Errors Gracefully

Always return descriptive strings on errors:

❌ **Bad:**
```python
@ai_function
def read_file(path: str) -> str:
    return open(path).read()  # Crashes on error
```

✅ **Good:**
```python
@ai_function
def read_file(
    path: Annotated[str, Field(description="Full path to the file to read")]
) -> str:
    """Reads and returns the contents of a file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found at path '{path}'"
    except PermissionError:
        return f"Error: Permission denied reading file '{path}'"
    except Exception as e:
        return f"Error reading file: {str(e)}"
```

**Why:** Returning error strings keeps the agent running and gives the LLM context to adapt its strategy.

### 5. Use Type Annotations

Always use `Annotated` with `Field`:

```python
from typing import Annotated
from pydantic import Field

@ai_function
def calculate_distance(
    lat1: Annotated[float, Field(description="Latitude of first location")],
    lon1: Annotated[float, Field(description="Longitude of first location")],
    lat2: Annotated[float, Field(description="Latitude of second location")],
    lon2: Annotated[float, Field(description="Longitude of second location")]
) -> float:
    """Calculates distance between two geographic coordinates."""
    pass
```

## Common Tool Patterns

### File Operations

```python
@ai_function
def write_file(
    path: Annotated[str, Field(description="Full path where file should be written")],
    content: Annotated[str, Field(description="Content to write to the file")]
) -> str:
    """Writes content to a file at the specified path."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"
```

### API Calls

```python
@ai_function
def get_weather(
    city: Annotated[str, Field(description="City name to get weather for")]
) -> str:
    """Fetches current weather information for a city."""
    try:
        response = httpx.get(f"https://api.weather.com/v1/weather/{city}")
        response.raise_for_status()
        data = response.json()
        return f"Temperature: {data['temp']}°C, Conditions: {data['conditions']}"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"
```

### Data Processing

```python
@ai_function
def analyze_csv(
    file_path: Annotated[str, Field(description="Path to CSV file to analyze")]
) -> str:
    """Analyzes a CSV file and returns summary statistics."""
    try:
        import pandas as pd
        df = pd.read_csv(file_path)
        summary = df.describe().to_string()
        return f"CSV Analysis:\n{summary}"
    except Exception as e:
        return f"Error analyzing CSV: {str(e)}"
```

## Testing Tools

Test tools independently before using them in agents:

```python
import asyncio

async def test_tool():
    result = web_search("Python programming")
    print(result)

    result = read_url("https://www.python.org")
    print(result[:200])

if __name__ == "__main__":
    asyncio.run(test_tool())
```

## Common Issues & Troubleshooting

### Tool Not Being Called

**Problem:** Agent doesn't use your tool even though it should.

**Solutions:**
- Improve the docstring to be more explicit about when to use the tool
- Add more specific keywords to parameter descriptions
- Ensure the tool name clearly indicates its purpose

### Invalid Parameters

**Problem:** LLM passes wrong data types or missing parameters.

**Solutions:**
- Make descriptions more specific about expected format
- Add examples in the docstring
- Validate and return helpful error messages

### Tool Execution Fails

**Problem:** Tool crashes during execution.

**Solutions:**
- Add comprehensive error handling with try/except
- Return descriptive error strings instead of raising exceptions
- Log errors for debugging while returning user-friendly messages

## Next Steps

Now that you understand tools, proceed to:

- [Creating Agents](agents.md) - Learn how to build agents that use these tools
- [Building Workflows](workflows.md) - Orchestrate multiple agents with different tools
