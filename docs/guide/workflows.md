# Building Workflows

Workflows enable multi-agent orchestration by defining how agents collaborate. The `WorkflowBuilder` creates directed acyclic graphs (DAGs) where data flows from one executor to the next.

## What Are Workflows?

A workflow is a **graph of executors** where:

- **Nodes (Executors)**: Functions that wrap agent calls or other logic
- **Edges**: Define data flow from one executor to another
- **Start Executor**: The entry point of the workflow
- **Output**: The final result from the last executor

```
User Input → Executor 1 → Executor 2 → Executor 3 → Final Output
                ↓            ↓            ↓
             Agent 1      Agent 2      Agent 3
```

## Creating Your First Workflow

### Basic Structure

Every workflow follows this pattern:

```python
from agent_workflow_framework import WorkflowBuilder, WorkflowContext, executor
from typing import Annotated
from pydantic import Field

# 1. Define executors (workflow nodes)
@executor(id="step1")
async def first_step(
    input_data: Annotated[str, Field(description="Input description")]
) -> dict:
    """First step logic."""
    result = await agent1.run(input_data)
    return {"data": result}

@executor(id="step2")
async def second_step(
    input_data: Annotated[dict, Field(description="Output from step1")]
) -> str:
    """Second step logic."""
    result = await agent2.run(input_data["data"])
    return result

# 2. Build the workflow
workflow = (WorkflowBuilder()
    .add_edge(first_step, second_step)  # Define data flow
    .set_start_executor(first_step)      # Set entry point
    .build())                            # Compile

# 3. Run the workflow
result = await workflow.run("initial input")
```

## Complete Example: Research + Writing Workflow

Let's create a workflow where:
1. **ResearchAgent** gathers information about a topic
2. **WriterAgent** uses that information to write a blog post

Create `run_workflow.py`:

```python
import asyncio
import os
from dotenv import load_dotenv
from agent_workflow_framework import WorkflowBuilder, WorkflowContext, executor
from pydantic import Field
from typing import Annotated

# Import our agents
from agents import ResearchAgent, WriterAgent

# --- 1. Instantiate Agents ---
# Create global instances for the workflow to use.
# This ensures they are initialized once.
print("Initializing agents...")
try:
    research_agent = ResearchAgent()
    writer_agent = WriterAgent()
except Exception as e:
    print(f"Error initializing agents. Is your .env file set up? Error: {e}")
    exit()

# --- 2. Define Workflow Executors ---
# Executors are the "nodes" in our workflow graph.
# They are async functions that call our agents.

@executor(id="run_research")
async def research_executor(
    topic: Annotated[str, Field(description="The topic to research")]
) -> dict:
    """
    First step: Run the ResearchAgent to gather information.
    This executor's output (a dict) will be the input for the next step.
    """
    print(f"--- [Executor: run_research] START ---")
    research_notes = await research_agent.run(f"Find information on: {topic}")
    return {"topic": topic, "research_notes": research_notes}

@executor(id="run_writing")
async def writing_executor(
    inputs: Annotated[dict, Field(description="The research notes from the previous step")]
) -> str:
    """
    Second step: Run the WriterAgent to write the blog post.
    It takes the dictionary from `research_executor` as its input.
    """
    print(f"--- [Executor: run_writing] START ---")
    topic = inputs["topic"]
    research_notes = inputs["research_notes"]

    blog_post = await writer_agent.run(topic, research_notes)
    return blog_post  # This is the final output of the workflow

# --- 3. Build the Workflow ---

print("Building workflow...")
workflow = (WorkflowBuilder()
    # Define the data flow: output of `research_executor`
    # goes to the input of `writing_executor`.
    .add_edge(research_executor, writing_executor)

    # Define the entry point of the workflow
    .set_start_executor(research_executor)

    # Compile the workflow
    .build())

# --- 4. Run the Workflow ---

async def main():
    print("--- Starting Workflow ---")
    load_dotenv()  # Load the .env file

    initial_topic = "the future of AI in 2025"

    # The .run() method takes the input for the start_executor
    final_result = await workflow.run(initial_topic)

    print("\n--- Workflow Complete ---")
    print("Final Blog Post:")
    print("=======================")
    print(final_result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Understanding Executors

### What Is an Executor?

An executor is a **decorated async function** that represents one step in your workflow:

```python
@executor(id="unique_step_id")
async def my_executor(
    input_param: Annotated[type, Field(description="What this input is")]
) -> output_type:
    """
    Description of what this step does.
    """
    # Call agents, process data, make decisions
    result = await some_agent.run(input_param)
    return result
```

**Key points:**
- Must be decorated with `@executor(id="unique_id")`
- Must be an `async` function
- Must use `Annotated[type, Field(description="...")]` for parameters
- Return value is passed to the next executor

### Executor Patterns

#### 1. Agent Wrapper Pattern

```python
@executor(id="analyze")
async def analysis_executor(
    data: Annotated[str, Field(description="Data to analyze")]
) -> dict:
    """Wraps a data analyst agent."""
    result = await data_analyst_agent.run(data)
    return {"analysis": result, "timestamp": datetime.now().isoformat()}
```

#### 2. Data Transformation Pattern

```python
@executor(id="transform")
async def transform_executor(
    raw_data: Annotated[dict, Field(description="Raw data from previous step")]
) -> str:
    """Transforms data without calling an agent."""
    # Pure Python logic
    processed = raw_data["analysis"].upper()
    return processed
```

#### 3. Conditional Logic Pattern

```python
@executor(id="decide")
async def decision_executor(
    input_data: Annotated[dict, Field(description="Data to evaluate")]
) -> str:
    """Makes decisions based on data."""
    if input_data["confidence"] > 0.8:
        return await high_confidence_agent.run(input_data["text"])
    else:
        return await verification_agent.run(input_data["text"])
```

## Building Complex Workflows

### Multiple Edges (Sequential Chain)

Create a linear pipeline:

```python
workflow = (WorkflowBuilder()
    .add_edge(step1, step2)
    .add_edge(step2, step3)
    .add_edge(step3, step4)
    .set_start_executor(step1)
    .build())
```

Data flows: `step1 → step2 → step3 → step4`

### Parallel Execution

**Note:** The current framework executes sequentially based on edges. For true parallel execution, you'll need to manage that within a single executor:

```python
@executor(id="parallel_step")
async def parallel_executor(
    input_data: Annotated[str, Field(description="Input for multiple agents")]
) -> dict:
    """Calls multiple agents in parallel."""
    # Run multiple agents concurrently
    results = await asyncio.gather(
        agent1.run(input_data),
        agent2.run(input_data),
        agent3.run(input_data)
    )

    return {
        "agent1_result": results[0],
        "agent2_result": results[1],
        "agent3_result": results[2]
    }
```

### Branching Logic

```python
@executor(id="branch")
async def branching_executor(
    input_data: Annotated[str, Field(description="Input to evaluate")]
) -> str:
    """Routes to different agents based on input."""
    if "technical" in input_data.lower():
        return await technical_agent.run(input_data)
    elif "creative" in input_data.lower():
        return await creative_agent.run(input_data)
    else:
        return await general_agent.run(input_data)
```

## Data Flow Between Executors

### How Data Passes Through Edges

When you define `.add_edge(executor_a, executor_b)`:

1. `executor_a` returns a value
2. That value becomes the **first parameter** of `executor_b`
3. The type must match the parameter annotation

**Example:**

```python
@executor(id="step1")
async def first_step(query: str) -> dict:
    return {"result": "processed", "count": 42}

@executor(id="step2")
async def second_step(
    data: Annotated[dict, Field(description="Output from step1")]
) -> str:
    # `data` receives {"result": "processed", "count": 42}
    return f"Got {data['count']} results: {data['result']}"
```

### Multiple Edges from One Executor

You can send one executor's output to multiple downstream executors:

```python
@executor(id="source")
async def source_executor(input: str) -> dict:
    return {"data": input}

@executor(id="process_a")
async def process_a_executor(data: dict) -> str:
    return f"Process A: {data['data']}"

@executor(id="process_b")
async def process_b_executor(data: dict) -> str:
    return f"Process B: {data['data']}"

# Both process_a and process_b receive the same output from source
workflow = (WorkflowBuilder()
    .add_edge(source_executor, process_a_executor)
    .add_edge(source_executor, process_b_executor)
    .set_start_executor(source_executor)
    .build())
```

**Note:** Currently, this creates a fork where both execute with the same input. The workflow will complete when all branches finish.

## Using WorkflowContext

For advanced state management, use `WorkflowContext`:

```python
@executor(id="stateful_step")
async def stateful_executor(
    input_data: Annotated[str, Field(description="Input data")],
    ctx: WorkflowContext[str]  # Access workflow context
) -> str:
    """Executor with access to workflow context."""
    # Send messages to other parts of the workflow
    await ctx.send_message("Progress update")

    # Yield intermediate outputs
    await ctx.yield_output("Intermediate result")

    # Final return
    return "Final result"
```

## Testing Workflows

### Test Individual Executors First

```python
async def test_research_executor():
    result = await research_executor("AI trends 2025")
    assert "research_notes" in result
    assert "topic" in result
    print(f"✓ Research executor works: {result}")

async def test_writing_executor():
    mock_input = {
        "topic": "Test Topic",
        "research_notes": "Test notes about the topic."
    }
    result = await writing_executor(mock_input)
    assert len(result) > 0
    print(f"✓ Writing executor works: {len(result)} chars")

# Run tests
asyncio.run(test_research_executor())
asyncio.run(test_writing_executor())
```

### Test the Complete Workflow

```python
async def test_workflow():
    load_dotenv()

    # Test with known input
    result = await workflow.run("Python programming")

    # Validate output
    assert len(result) > 100
    assert "Python" in result
    print("✓ Workflow test passed")

asyncio.run(test_workflow())
```

## Expected Output

When running the complete example:

```
Initializing agents...
Building workflow...
--- Starting Workflow ---
--- [Executor: run_research] START ---

--- [ResearchAgent] Received query: Find information on: the future of AI in 2025 ---
[Tool Call] Searching for: AI trends 2025
[Tool Call] Reading URL: https://example.com/ai-2025
--- [ResearchAgent] Responding ---
--- [Executor: run_writing] START ---

--- [WriterAgent] Received prompt ---
--- [WriterAgent] Responding ---

--- Workflow Complete ---
Final Blog Post:
=======================
## The Future of AI in 2025

The artificial intelligence landscape in 2025 is poised for transformative 
breakthroughs. Enterprise adoption of AI agents will accelerate...

[Blog post continues...]
```

## Best Practices

### 1. Keep Executors Focused

❌ **Bad:** One executor doing everything
```python
@executor(id="do_everything")
async def everything_executor(input: str) -> str:
    research = await research_agent.run(input)
    analysis = await analysis_agent.run(research)
    writing = await writer_agent.run(analysis)
    return writing
```

✅ **Good:** Separate executors for each step
```python
@executor(id="research")
async def research_exec(input: str) -> str: ...

@executor(id="analyze")
async def analyze_exec(research: str) -> str: ...

@executor(id="write")
async def write_exec(analysis: str) -> str: ...
```

### 2. Use Type Hints and Descriptions

Always annotate parameters clearly:

```python
@executor(id="process")
async def process_executor(
    # ✅ Good: Clear type and description
    data: Annotated[dict, Field(description="Dictionary containing 'text' and 'metadata' keys")]
) -> str:
    ...
```

### 3. Handle Errors in Executors

```python
@executor(id="safe_step")
async def safe_executor(
    input_data: Annotated[str, Field(description="Input to process")]
) -> str:
    try:
        result = await agent.run(input_data)
        return result
    except Exception as e:
        print(f"Error in executor: {e}")
        return f"Error occurred: {str(e)}"
```

### 4. Log Execution Flow

Add logging to track workflow progress:

```python
@executor(id="logged_step")
async def logged_executor(
    input_data: Annotated[str, Field(description="Input")]
) -> str:
    print(f"[{datetime.now()}] Starting logged_step with input: {input_data[:50]}")
    result = await agent.run(input_data)
    print(f"[{datetime.now()}] Completed logged_step: {len(result)} chars")
    return result
```

## Common Issues & Troubleshooting

### Workflow Doesn't Start

**Problem:** Workflow hangs or doesn't execute.

**Solutions:**
- Ensure `.set_start_executor()` is called
- Verify the start executor is included in an edge
- Check that `.build()` is called before `.run()`

### Data Type Mismatches

**Problem:** "Type mismatch" or "unexpected parameter" errors.

**Solutions:**
- Ensure return type of executor A matches parameter type of executor B
- Use `dict` for complex data passing
- Verify `Annotated` types are correct

### Executors Not Chaining

**Problem:** Only first executor runs, rest are skipped.

**Solutions:**
- Check that edges are defined correctly
- Ensure return values are not `None`
- Verify executor IDs are unique

### Workflow Times Out

**Problem:** Workflow runs but never completes.

**Solutions:**
- Check for infinite loops in executor logic
- Ensure all async calls use `await`
- Verify agents complete and return values

## Advanced Patterns

### Retry Logic

```python
@executor(id="retry_step")
async def retry_executor(
    input_data: Annotated[str, Field(description="Input")]
) -> str:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = await agent.run(input_data)
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Failed after {max_retries} attempts: {e}"
            print(f"Attempt {attempt + 1} failed, retrying...")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### Caching Results

```python
_cache = {}

@executor(id="cached_step")
async def cached_executor(
    input_data: Annotated[str, Field(description="Input")]
) -> str:
    if input_data in _cache:
        print("Returning cached result")
        return _cache[input_data]

    result = await expensive_agent.run(input_data)
    _cache[input_data] = result
    return result
```

## Next Steps

Now that you understand workflows, explore:

- [API Reference](../api-reference.md) - Detailed API documentation
- [Examples](../examples.md) - Complex workflow patterns
- [Core Concepts](concepts.md) - Deeper understanding of the framework architecture
