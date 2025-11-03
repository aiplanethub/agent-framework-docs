# Examples

Practical examples demonstrating how to build production-ready agent workflows using the Agent Workflow Framework.

## Overview

This guide provides complete, working examples that demonstrate:

- **Real-world use cases** for agent workflows
- **Best practices** for code organization
- **Tool creation** and integration patterns
- **Multi-agent orchestration** strategies
- **State management** and persistence
- **Error handling** and monitoring

## Prerequisites

Before working with these examples, ensure you have:

- **Framework installed**: See [Installation Guide](installation.md)
- **Azure configuration**: Environment variables set up
- **Python knowledge**: Comfortable with async/await patterns
- **Understanding of core concepts**: Reviewed [Core Concepts](guide/concepts.md)

### Required Environment Variables

```bash
# Azure AI Configuration (recommended)
AZURE_AI_PROJECT_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_AI_MODEL_DEPLOYMENT_NAME=gpt-4o-mini

# OR OpenAI Configuration (alternative)
OPENAI_API_KEY=your-api-key
LLM_MODEL=gpt-4o-mini
```

---

## Example 1: Fraud Detection Workflow

A comprehensive enterprise-grade fraud detection system that compares CSV files, identifies discrepancies, and generates detailed risk assessments.

### Use Case

Financial institutions and payment processors need to verify transaction integrity by comparing reported transactions against actual records. This workflow automates:

- **Data comparison**: Record-by-record analysis of CSV files
- **Pattern detection**: Identification of sophisticated fraud schemes
- **Risk scoring**: Quantitative assessment of fraud indicators
- **Alert generation**: Automated notifications for high-risk cases
- **Report generation**: Comprehensive documentation for investigators

### Real-World Applications

- **Banking**: Verify transaction logs across systems
- **E-commerce**: Detect payment fraud and chargebacks
- **Insurance**: Identify claims fraud patterns
- **Accounting**: Audit financial records for discrepancies
- **Healthcare**: Validate billing and claims data

---

### Architecture Overview

The workflow uses a **multi-agent pipeline** with specialized agents for each analysis stage:

```
┌─────────────────┐     ┌─────────────────┐
│  Reported CSV   │     │   Actual CSV    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
            ┌────────▼────────┐
            │  CSV Comparison │ (Tool)
            │      Tool       │
            └────────┬────────┘
                     │
            ┌────────▼────────┐
            │ Fraud Analyst   │ (Agent)
            │   Initial       │
            │   Analysis      │
            └────────┬────────┘
                     │
            ┌────────▼────────┐
            │ Pattern         │ (Agent)
            │   Detector      │
            └────────┬────────┘
                     │
            ┌────────▼────────┐
            │  Risk Scorer    │ (Agent)
            └────────┬────────┘
                     │
            ┌────────▼────────┐
            │    Fraud        │ (Agent)
            │  Classifier     │
            └─────┬──────┬────┘
                  │      │
         ┌────────▼──┐ ┌▼────────────┐
         │  Report   │ │    Alert    │ (Agents)
         │ Generator │ │  Generator  │
         └───────────┘ └─────────────┘
```

### Key Components

1. **Custom CSV Comparison Tool**: Performs detailed file comparison
2. **7 Specialized Agents**: Each handles specific analysis stages
3. **Workflow Engine**: Orchestrates execution with state management
4. **Monitoring System**: Tracks performance and generates alerts
5. **Persistence Layer**: Saves workflow state and results

---

### Project Structure

```
fraud_detection_workflow/
├── fraud_detection_workflow.py    # Main workflow implementation
├── streamlit_fraud_detection_app.py  # Web UI
├── setup_env.py                   # Environment configuration helper
├── run_streamlit.py              # UI launcher script
├── requirements.txt              # Dependencies
├── FRAUD_DETECTION_README.md     # Detailed documentation
├── STREAMLIT_README.md           # UI guide
├── test_data/                    # Generated test CSV files
└── workflow_data/                # Workflow state persistence
```

---

### Step-by-Step Walkthrough

#### 1. Custom Tool Creation

First, we create a custom tool for CSV comparison:

```python
from agent_workflow_framework.tools import BaseTool
from agent_workflow_framework.tools.base import ToolSchema, ToolParameter, ToolResult
import pandas as pd
import io

class CSVComparisonTool(BaseTool):
    """Custom tool for comparing two CSV files and identifying discrepancies."""

    def __init__(self):
        super().__init__(name="csv_comparator")

    @property
    def schema(self) -> ToolSchema:
        """Define the tool's schema for the LLM."""
        return ToolSchema(
            name="csv_comparator",
            description="Compare two CSV files and identify mismatches, discrepancies, and potential fraud indicators",
            parameters=[
                ToolParameter(
                    name="file1_content",
                    type="string",
                    description="Content of the first CSV file (reported/expected data)",
                    required=True
                ),
                ToolParameter(
                    name="file2_content",
                    type="string",
                    description="Content of the second CSV file (actual/received data)",
                    required=True
                ),
                ToolParameter(
                    name="key_column",
                    type="string",
                    description="Primary key column name for matching records",
                    required=False,
                    default=None
                )
            ],
            returns="Dictionary containing detailed comparison results"
        )

    async def execute(self, file1_content: str, file2_content: str, key_column: str = None) -> ToolResult:
        """
        Compare two CSV files and return detailed comparison results.

        Returns ToolResult with:
        - success: bool indicating if comparison succeeded
        - result: dict with comparison statistics and mismatches
        - error: str with error message if failed
        """
        try:
            # Parse CSV content using pandas
            df1 = pd.read_csv(io.StringIO(file1_content))
            df2 = pd.read_csv(io.StringIO(file2_content))

            # Initialize comparison results
            comparison_results = {
                "file1_records": len(df1),
                "file2_records": len(df2),
                "columns_file1": list(df1.columns),
                "columns_file2": list(df2.columns),
                "mismatches": [],
                "missing_in_file2": [],
                "extra_in_file2": [],
                "value_mismatches": [],
                "statistics": {}
            }

            # Identify common columns
            cols1 = set(df1.columns)
            cols2 = set(df2.columns)
            common_cols = cols1.intersection(cols2)

            comparison_results["column_differences"] = {
                "only_in_file1": list(cols1 - cols2),
                "only_in_file2": list(cols2 - cols1),
                "common_columns": list(common_cols)
            }

            # Compare records by key column if specified
            if key_column and key_column in common_cols:
                keys1 = set(df1[key_column].astype(str))
                keys2 = set(df2[key_column].astype(str))

                # Identify missing/extra records
                comparison_results["missing_in_file2"] = list(keys1 - keys2)
                comparison_results["extra_in_file2"] = list(keys2 - keys1)

                # Compare matching records
                common_keys = keys1.intersection(keys2)
                for key in common_keys:
                    row1 = df1[df1[key_column].astype(str) == str(key)].iloc[0]
                    row2 = df2[df2[key_column].astype(str) == str(key)].iloc[0]

                    # Find value mismatches
                    mismatches = {}
                    for col in common_cols:
                        if col != key_column:
                            val1 = str(row1[col])
                            val2 = str(row2[col])
                            if val1 != val2:
                                mismatches[col] = {"file1": val1, "file2": val2}

                    if mismatches:
                        comparison_results["value_mismatches"].append({
                            "key": key,
                            "mismatches": mismatches
                        })

            # Calculate statistics
            total_cells = len(common_cols) * min(len(df1), len(df2))
            mismatched_cells = sum(len(m["mismatches"]) for m in comparison_results["value_mismatches"])

            comparison_results["statistics"] = {
                "total_records_compared": min(len(df1), len(df2)),
                "total_mismatches": len(comparison_results["value_mismatches"]),
                "mismatch_percentage": (len(comparison_results["value_mismatches"]) / min(len(df1), len(df2)) * 100) if min(len(df1), len(df2)) > 0 else 0,
                "mismatched_cells": mismatched_cells,
                "cell_mismatch_percentage": (mismatched_cells / total_cells * 100) if total_cells > 0 else 0
            }

            return ToolResult(success=True, result=comparison_results)

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"CSV comparison failed: {str(e)}",
                result={"error": str(e), "status": "failed"}
            )
```

**Key Points:**

- **BaseTool inheritance**: Provides framework integration
- **Schema definition**: Tells the LLM what parameters the tool accepts
- **Error handling**: Returns ToolResult with success/error information
- **Detailed results**: Provides comprehensive comparison data for agents

#### 2. Creating Specialized Agents

Each agent has a specific role in the fraud detection pipeline:

```python
from agent_workflow_framework import create_specialist, create_assistant

# Fraud Analyst - Initial discrepancy identification
fraud_analyst = create_specialist(
    name="fraud_analyst",
    domain="fraud_detection",
    expertise_areas=["fraud detection", "transaction analysis", "anomaly detection"],
    description="Initial fraud analysis specialist",
    tools=["text_analysis"],  # Built-in tool
    system_prompt="""You are a senior fraud analyst specializing in transaction fraud detection.

Your expertise includes:
- Identifying transaction discrepancies and their implications
- Recognizing common fraud patterns and schemes
- Analyzing data integrity issues
- Detecting unauthorized modifications
- Assessing fraud risk levels

Always provide specific, detailed analysis with clear fraud indicators and evidence."""
)

# Pattern Detection Agent - Complex pattern recognition
pattern_detector = create_specialist(
    name="pattern_detector",
    domain="pattern_analysis",
    expertise_areas=["pattern recognition", "behavioral analysis", "fraud schemes"],
    description="Fraud pattern detection specialist",
    system_prompt="""You are a fraud pattern detection expert specializing in identifying complex fraud schemes.

Your expertise includes:
- Time-based fraud patterns
- Amount manipulation patterns
- Entity relationship patterns
- Velocity and frequency analysis
- Modus operandi identification

Focus on identifying sophisticated fraud patterns that might not be immediately obvious."""
)

# Risk Scoring Agent - Quantitative assessment
risk_scorer = create_specialist(
    name="risk_scorer",
    domain="risk_assessment",
    expertise_areas=["risk assessment", "statistical analysis", "scoring models"],
    description="Fraud risk scoring specialist",
    system_prompt="""You are a risk scoring specialist focused on quantifying fraud risk.

Your responsibilities:
- Calculate precise risk scores (0-100 scale)
- Apply weighted scoring methodologies
- Consider multiple risk factors
- Provide score breakdowns and justifications
- Ensure consistent scoring criteria

Always provide numerical scores with clear explanations of the scoring methodology."""
)

# Fraud Classifier - Final determination
fraud_classifier = create_specialist(
    name="fraud_classifier",
    domain="fraud_classification",
    expertise_areas=["fraud classification", "decision making", "compliance"],
    description="Fraud classification and decision specialist",
    system_prompt="""You are a fraud classification specialist responsible for final fraud determinations.

Your responsibilities:
- Classify fraud risk levels (HIGH/MEDIUM/LOW)
- Make clear, actionable recommendations
- Prioritize investigation efforts
- Ensure compliance with policies
- Balance false positives with fraud prevention

Provide clear classifications with specific action items."""
)

# Report Generator - Documentation
report_generator = create_assistant(
    name="report_generator",
    description="Fraud report generation specialist",
    system_prompt="""You are a fraud reporting specialist who creates comprehensive fraud detection reports.

Your responsibilities:
- Generate clear, professional fraud reports
- Include executive summaries and detailed findings
- Present evidence and risk scores clearly
- Provide actionable recommendations
- Ensure reports are suitable for fraud investigators and management

Create well-structured reports that facilitate quick decision-making."""
)
```

**Agent Design Principles:**

1. **Single Responsibility**: Each agent has one clear purpose
2. **Domain Expertise**: System prompts define specialized knowledge
3. **Clear Instructions**: Explicit guidance on responsibilities
4. **Composition Pattern**: Uses framework functions, not inheritance
5. **Prompt Engineering**: Detailed context for consistent output

#### 3. Workflow Definition

Define the workflow nodes with proper dependencies:

```python
from agent_workflow_framework import WorkflowDefinition, WorkflowNode, WorkflowVariable, NodeInput
from agent_workflow_framework.workflow.definition import NodeOutput
from agent_workflow_framework.workflow import NodeType, ExecutionStrategy, RetryPolicy

async def create_fraud_detection_workflow(reported_file: str, actual_file: str) -> WorkflowDefinition:
    """Create a comprehensive fraud detection workflow."""

    # Define workflow variables (configuration)
    variables = [
        WorkflowVariable(
            name="reported_file",
            type="string",
            default_value=reported_file,
            description="Path to reported transactions CSV"
        ),
        WorkflowVariable(
            name="actual_file",
            type="string",
            default_value=actual_file,
            description="Path to actual transactions CSV"
        ),
        WorkflowVariable(
            name="key_column",
            type="string",
            default_value="transaction_id",
            description="Primary key column for matching records"
        ),
        WorkflowVariable(
            name="fraud_threshold",
            type="number",
            default_value=70,
            description="Fraud risk score threshold (0-100)"
        )
    ]

    # Define workflow nodes (execution steps)
    nodes = [
        # Node 1: Read Reported File (Tool execution)
        WorkflowNode(
            id="read_reported_file",
            name="Read Reported File",
            description="Read the reported transactions CSV file",
            type=NodeType.TOOL,  # Tool execution node
            config={
                "tool_name": "file_read",  # Built-in tool
                "parameter_mapping": {
                    "file_path": "file_path"
                }
            },
            depends_on=[],  # No dependencies - can run first
            inputs={
                "file_path": NodeInput(variable="reported_file")
            },
            outputs=[NodeOutput(name="result", type="string", description="Content of reported file")],
            timeout_seconds=30
        ),

        # Node 2: Read Actual File (Parallel with Node 1)
        WorkflowNode(
            id="read_actual_file",
            name="Read Actual File",
            description="Read the actual transactions CSV file",
            type=NodeType.TOOL,
            config={
                "tool_name": "file_read",
                "parameter_mapping": {
                    "file_path": "file_path"
                }
            },
            depends_on=[],  # No dependencies - can run in parallel
            inputs={
                "file_path": NodeInput(variable="actual_file")
            },
            outputs=[NodeOutput(name="result", type="string", description="Content of actual file")],
            timeout_seconds=30
        ),

        # Node 3: CSV Comparison (Depends on Nodes 1 & 2)
        WorkflowNode(
            id="compare_files",
            name="CSV Comparison",
            description="Compare the two CSV files for discrepancies",
            type=NodeType.TOOL,
            config={
                "tool_name": "csv_comparator",  # Our custom tool
                "parameter_mapping": {
                    "file1_content": "reported_content",
                    "file2_content": "actual_content",
                    "key_column": "key_col"
                }
            },
            depends_on=["read_reported_file", "read_actual_file"],  # Wait for file reads
            inputs={
                "reported_content": NodeInput(
                    source_node="read_reported_file",
                    source_output="result"
                ),
                "actual_content": NodeInput(
                    source_node="read_actual_file",
                    source_output="result"
                ),
                "key_col": NodeInput(variable="key_column")
            },
            outputs=[NodeOutput(name="comparison_results", type="dict")],
            timeout_seconds=60
        ),

        # Node 4: Initial Fraud Analysis (Agent execution)
        WorkflowNode(
            id="initial_analysis",
            name="Initial Fraud Analysis",
            description="Analyze comparison results for fraud indicators",
            type=NodeType.AGENT,  # Agent execution node
            config={
                "agent_name": "fraud_analyst"  # Use our fraud_analyst agent
            },
            depends_on=["compare_files"],
            inputs={
                "message": NodeInput(
                    value="You are a fraud analyst. Analyze the CSV comparison results provided below for fraud indicators. Provide: 1) Summary of discrepancies, 2) Fraud indicators, 3) Risk assessment, 4) Recommended next steps."
                ),
                "comparison_results": NodeInput(
                    source_node="compare_files",
                    source_output="result"
                )
            },
            outputs=[NodeOutput(name="fraud_analysis", type="string")],
            timeout_seconds=120
        ),

        # Nodes 5-10: Additional analysis stages
        # (Pattern Detection, Risk Scoring, Classification, Reporting, Alerts, Action Planning)
        # ... (see full implementation in fraud_detection_workflow.py)
    ]

    # Create workflow definition
    workflow = WorkflowDefinition(
        name="Fraud Detection Analysis",
        description="Comprehensive fraud detection through CSV comparison and pattern analysis",
        version="1.0",
        variables=variables,
        nodes=nodes,
        execution_strategy=ExecutionStrategy.SEQUENTIAL,  # Execute nodes in order
        max_parallel_nodes=3,  # Allow up to 3 parallel executions
        global_retry_policy=RetryPolicy(
            max_attempts=2,
            delay_seconds=5.0,
            exponential_backoff=True
        ),
        tags=["fraud-detection", "comparison", "risk-analysis"]
    )

    return workflow
```

**Workflow Design Patterns:**

1. **Parallel Execution**: File reads run simultaneously
2. **Sequential Processing**: Analysis stages run in order
3. **Node Dependencies**: `depends_on` ensures correct execution order
4. **Input Mapping**: Links outputs to inputs between nodes
5. **Type Safety**: Output types match input expectations
6. **Configuration**: Variables allow runtime customization
7. **Error Handling**: Retry policies for transient failures

#### 4. Workflow Engine Setup

Initialize and configure the workflow engine:

```python
from agent_workflow_framework import (
    WorkflowEngine,
    WorkflowStateManager,
    FilePersistenceBackend,
    WorkflowMonitor,
    console_alert_handler,
    ToolRegistry
)
from agent_workflow_framework.tools.builtin import register_builtin_tools

async def setup_fraud_detection_system():
    """Setup the fraud detection workflow system."""

    # 1. Setup Tool Registry
    tool_registry = ToolRegistry("fraud_tools")

    # Register built-in tools (file operations, text analysis, etc.)
    categories = register_builtin_tools(tool_registry)

    # Register custom CSV comparison tool
    csv_tool = CSVComparisonTool()
    tool_registry.register_tool(csv_tool)

    # 2. Setup Persistence Backend
    # Saves workflow state to disk for recovery and auditing
    persistence_backend = FilePersistenceBackend("./workflow_data")
    state_manager = WorkflowStateManager(
        backend=persistence_backend,
        auto_save=True,  # Automatically save state changes
        checkpoint_interval=30  # Save every 30 seconds
    )

    # 3. Setup Monitoring
    # Tracks workflow execution and generates alerts
    monitor = WorkflowMonitor(retention_hours=24)
    monitor.add_alert_handler(console_alert_handler)  # Print alerts to console

    # 4. Setup Workflow Engine
    engine = WorkflowEngine(config={
        "max_concurrent_executions": 3,
        "default_timeout": 300,  # 5 minutes
        "enable_persistence": True
    })

    # Register all agents with the engine
    engine.register_agent(fraud_analyst)
    engine.register_agent(pattern_detector)
    engine.register_agent(risk_scorer)
    engine.register_agent(fraud_classifier)
    engine.register_agent(report_generator)
    # ... register other agents

    # Register tool registry
    engine.set_tool_registry(tool_registry)

    # Connect monitoring events
    engine.on_event("workflow_started", monitor.on_workflow_started)
    engine.on_event("workflow_completed", monitor.on_workflow_completed)
    engine.on_event("workflow_failed", monitor.on_workflow_failed)
    engine.on_event("node_started", monitor.on_node_started)
    engine.on_event("node_completed", monitor.on_node_completed)
    engine.on_event("node_failed", monitor.on_node_failed)

    return {
        "engine": engine,
        "state_manager": state_manager,
        "monitor": monitor,
        "tool_registry": tool_registry
    }
```

**System Setup Components:**

1. **Tool Registry**: Manages all available tools
2. **Persistence Backend**: Saves workflow state for recovery
3. **State Manager**: Handles checkpointing and state transitions
4. **Monitoring**: Tracks execution and generates alerts
5. **Event Handlers**: Connect monitoring to workflow events

#### 5. Executing the Workflow

Run the fraud detection analysis:

```python
async def run_fraud_detection_workflow():
    """Run the fraud detection workflow."""

    # 1. Setup system
    system = await setup_fraud_detection_system()
    engine = system["engine"]
    state_manager = system["state_manager"]
    monitor = system["monitor"]

    # 2. Create workflow
    workflow = await create_fraud_detection_workflow(
        reported_file="./test_data/reported_transactions.csv",
        actual_file="./test_data/actual_transactions.csv"
    )

    # 3. Validate workflow structure
    validation_errors = workflow.validate_dependencies()
    if validation_errors:
        print(f"Validation failed: {validation_errors}")
        return

    # 4. Save workflow definition
    await state_manager.save_workflow(workflow)

    # 5. Execute workflow
    execution = await engine.execute_workflow(
        workflow=workflow,
        trigger_data={
            "triggered_by": "fraud_detection_system",
            "analysis_id": f"FRAUD-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "timestamp": datetime.now().isoformat()
        }
    )

    # 6. Check results
    print(f"Execution Status: {execution.status.value}")
    print(f"Duration: {(execution.completed_at - execution.started_at).total_seconds():.2f}s")

    # 7. Access node results
    for node_id, node_exec in execution.node_executions.items():
        print(f"{node_id}: {node_exec.status.value}")
        if node_exec.output:
            print(f"  Output: {str(node_exec.output)[:100]}...")

    # 8. Save execution results
    await state_manager.save_execution(execution, immediate=True)

    # 9. Get monitoring statistics
    stats = monitor.get_workflow_stats()
    print(f"Success Rate: {stats['global_stats']['success_rate']:.1%}")

    return execution
```

**Execution Flow:**

1. **System initialization**: Setup engine, monitoring, persistence
2. **Workflow creation**: Define structure and dependencies
3. **Validation**: Check for circular dependencies and errors
4. **State persistence**: Save workflow definition
5. **Execution**: Run the workflow with trigger data
6. **Result retrieval**: Access outputs from each node
7. **State saving**: Persist execution results
8. **Monitoring**: Review execution statistics

---

### Running the Example

#### Option 1: Command Line

```bash
# 1. Navigate to example directory
cd examples/fraud_detection_workflow

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
python setup_env.py

# 4. Run the workflow
python fraud_detection_workflow.py
```

#### Option 2: Streamlit Web UI

```bash
# 1. Run the Streamlit launcher
python run_streamlit.py

# 2. Open browser to http://localhost:8501

# 3. Upload CSV files and configure settings

# 4. Click "Run Analysis" to execute workflow
```

---

### Expected Output

```
🔍 FRAUD DETECTION WORKFLOW - CSV COMPARISON & ANALYSIS
========================================================================

✅ Using Azure AI configuration

📁 Creating test CSV files with fraud indicators...
   - Reported transactions: ./test_data/reported_transactions.csv
   - Actual transactions: ./test_data/actual_transactions.csv

1. Setting up tool registry...
✅ Registered 15 tools including custom CSV comparator

2. Creating fraud detection agents...
✅ Created 7 specialized fraud detection agents

3. Setting up persistence and monitoring...
✅ Persistence and monitoring configured

4. Setting up workflow engine...
✅ Workflow engine configured

📋 Creating fraud detection workflow...
✅ Fraud detection workflow created: Fraud Detection Analysis
   - Total nodes: 10
   - Execution strategy: SEQUENTIAL
   - Analysis stages: Initial Analysis → Pattern Detection → Risk Scoring → Classification → Reporting

🚀 Executing fraud detection analysis...
   Analyzing transaction discrepancies...
   Detecting fraud patterns...
   Calculating risk scores...

✅ Fraud detection completed!
   - Execution ID: wf_20250103_143022
   - Status: completed
   - Duration: 47.32s

📊 Analysis Stage Results:
   ✅ Compare Files: completed (2.15s)
   ✅ Initial Analysis: completed (12.43s)
   ✅ Pattern Detection: completed (10.28s)
   ✅ Risk Scoring: completed (8.67s)
   ✅ Classify Fraud: completed (7.91s)
   ✅ Generate Report: completed (5.88s)

📈 Fraud Detection Statistics:
   - System Health: healthy
   - Total Executions: 1
   - Success Rate: 100.0%
   - Average Duration: 47.32s

========================================================================
✅ FRAUD DETECTION WORKFLOW COMPLETED SUCCESSFULLY!
========================================================================

💡 Check the generated fraud report for detailed findings and recommendations
```

---

### Key Features Demonstrated

#### 1. Custom Tool Integration

The CSV comparison tool shows how to:

- Extend `BaseTool` for custom functionality
- Define tool schemas for LLM interaction
- Handle errors gracefully with `ToolResult`
- Return structured data for agent processing

#### 2. Multi-Agent Orchestration

The workflow demonstrates:

- **Specialist agents** with domain expertise
- **Sequential processing** through analysis stages
- **Data flow** between agents via node outputs
- **Parallel execution** for independent operations

#### 3. State Management

The persistence layer provides:

- **Workflow definitions** saved to disk
- **Execution state** tracked in real-time
- **Checkpointing** for recovery from failures
- **Audit trails** for compliance and debugging

#### 4. Error Handling

Robust error management includes:

- **Retry policies** for transient failures
- **Timeout handling** for long-running operations
- **Graceful degradation** when services fail
- **Detailed error messages** for troubleshooting

#### 5. Monitoring and Alerts

The monitoring system tracks:

- **Execution metrics** (duration, success rate)
- **Node-level performance** for bottleneck identification
- **Alert generation** for high-risk detections
- **Health status** of the overall system

---

### Configuration Options

#### Workflow Variables

Customize the workflow by modifying variables:

```python
variables = [
    WorkflowVariable(
        name="fraud_threshold",
        type="number",
        default_value=80,  # Increase for stricter detection
        description="Fraud risk score threshold (0-100)"
    ),
    WorkflowVariable(
        name="key_column",
        type="string",
        default_value="transaction_id",  # Change for different key
        description="Primary key column for matching records"
    )
]
```

#### Agent Customization

Modify agent system prompts for different domains:

```python
fraud_analyst = create_specialist(
    name="fraud_analyst",
    domain="healthcare_fraud",  # Domain-specific
    system_prompt="""You are a healthcare fraud analyst specializing in medical billing fraud.

Your expertise includes:
- Upcoding and unbundling detection
- Duplicate billing identification
- Medical necessity validation
- Provider fraud patterns

Always cite specific billing codes and regulations."""
)
```

#### Execution Strategy

Adjust workflow execution behavior:

```python
workflow = WorkflowDefinition(
    # ...
    execution_strategy=ExecutionStrategy.PARALLEL,  # or SEQUENTIAL
    max_parallel_nodes=5,  # Increase for more concurrency
    global_retry_policy=RetryPolicy(
        max_attempts=3,  # More retries for unreliable services
        delay_seconds=10.0,  # Longer delays
        exponential_backoff=True
    )
)
```

---

### Code Organization Best Practices

#### 1. Separation of Concerns

```
fraud_detection/
├── tools/
│   └── csv_tools.py          # Custom tools
├── agents/
│   ├── fraud_analyst.py      # Agent definitions
│   ├── pattern_detector.py
│   └── risk_scorer.py
├── workflows/
│   └── fraud_workflow.py     # Workflow definitions
├── config/
│   └── settings.py           # Configuration
└── main.py                   # Entry point
```

#### 2. Configuration Management

Use environment-based configuration:

```python
# config/settings.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Azure AI
    azure_endpoint: str
    azure_model: str

    # Workflow
    fraud_threshold: int = 70
    max_concurrent_executions: int = 3
    default_timeout: int = 300

    # Persistence
    workflow_data_dir: str = "./workflow_data"
    auto_save: bool = True
    checkpoint_interval: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Usage
settings = Settings()
```

#### 3. Error Handling Patterns

Centralized error handling:

```python
# utils/error_handlers.py
from functools import wraps
import logging

logger = logging.getLogger(__name__)

def handle_workflow_errors(func):
    """Decorator for workflow error handling."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except WorkflowValidationError as e:
            logger.error(f"Workflow validation failed: {e}")
            raise
        except WorkflowExecutionError as e:
            logger.error(f"Workflow execution failed: {e}")
            # Attempt recovery
            await save_error_state(e)
            raise
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            raise
    return wrapper

@handle_workflow_errors
async def run_workflow():
    # ...
```

#### 4. Testing Strategy

Comprehensive testing approach:

```python
# tests/test_fraud_workflow.py
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_csv_comparison_tool():
    """Test CSV comparison with sample data."""
    tool = CSVComparisonTool()
    result = await tool.execute(
        file1_content=SAMPLE_CSV_1,
        file2_content=SAMPLE_CSV_2,
        key_column="transaction_id"
    )

    assert result.success is True
    assert "value_mismatches" in result.result
    assert len(result.result["value_mismatches"]) > 0

@pytest.mark.asyncio
async def test_fraud_analyst_agent():
    """Test fraud analyst with mock comparison data."""
    with patch.object(fraud_analyst, '_agent') as mock_agent:
        mock_agent.run = AsyncMock(return_value=Mock(text="Fraud detected"))

        result = await fraud_analyst.run(MOCK_COMPARISON_DATA)

        assert "fraud" in result.lower()
        mock_agent.run.assert_called_once()

@pytest.mark.asyncio
async def test_workflow_execution():
    """Test complete workflow with test data."""
    workflow = await create_fraud_detection_workflow(
        reported_file="./test_data/test_reported.csv",
        actual_file="./test_data/test_actual.csv"
    )

    # Validate workflow structure
    errors = workflow.validate_dependencies()
    assert len(errors) == 0

    # Execute workflow
    engine = await setup_test_engine()
    execution = await engine.execute_workflow(workflow)

    assert execution.status == ExecutionStatus.COMPLETED
    assert "generate_report" in execution.node_executions
```

---

### Deployment Considerations

#### Production Checklist

- [ ] **Security**: Implement authentication and authorization
- [ ] **Logging**: Add comprehensive logging for audit trails
- [ ] **Monitoring**: Set up alerts for failures and performance issues
- [ ] **Persistence**: Use database backend instead of file storage
- [ ] **Scalability**: Configure for horizontal scaling
- [ ] **Error Recovery**: Implement workflow resume capabilities
- [ ] **Testing**: Achieve >80% code coverage
- [ ] **Documentation**: Maintain up-to-date API docs

#### Azure Deployment

Deploy as Azure Function or Container App:

```python
# function_app.py (Azure Functions)
import azure.functions as func
from fraud_detection_workflow import run_fraud_detection_workflow

app = func.FunctionApp()

@app.route(route="detect-fraud", methods=["POST"])
async def detect_fraud(req: func.HttpRequest) -> func.HttpResponse:
    """HTTP trigger for fraud detection workflow."""

    # Parse request
    req_body = req.get_json()
    reported_file = req_body.get('reported_file')
    actual_file = req_body.get('actual_file')

    # Execute workflow
    result = await run_fraud_detection_workflow(reported_file, actual_file)

    # Return results
    return func.HttpResponse(
        json.dumps({
            "status": result.status.value,
            "execution_id": result.id,
            "summary": result.outputs.get("fraud_report")
        }),
        mimetype="application/json"
    )
```

---

## Example 2: Simple Agent Workflow _(Coming Soon)_

A basic single-agent workflow demonstrating core concepts.

### What You'll Learn

- Creating a simple agent with one tool
- Defining a minimal workflow
- Running and testing workflows
- Accessing agent outputs

### Use Case

Build a weather information agent that:
1. Accepts a city name
2. Calls a weather API
3. Returns formatted weather information

_Full example coming soon. Check back for updates._

---

## Example 3: Multi-Agent Collaboration _(Coming Soon)_

Multiple agents working together on complex tasks.

### What You'll Learn

- Coordinating multiple specialized agents
- Passing data between agents
- Implementing agent handoffs
- Managing shared state

### Use Case

Research and content creation pipeline:
1. **Research Agent**: Gathers information from web
2. **Analysis Agent**: Extracts key insights
3. **Writer Agent**: Creates blog post
4. **Editor Agent**: Reviews and refines content

_Full example coming soon. Check back for updates._

---

## Example 4: Custom Tool Integration _(Coming Soon)_

Building and integrating custom tools.

### What You'll Learn

- Creating custom tools with `BaseTool`
- Defining tool schemas
- Error handling in tools
- Testing custom tools

### Use Case

Database query tool that:
1. Accepts SQL queries
2. Executes against database
3. Returns formatted results
4. Handles connection errors

_Full example coming soon. Check back for updates._

---

## Additional Resources

### Documentation

- [Installation Guide](installation.md) - Setup instructions
- [Quick Start](quickstart.md) - 5-minute introduction
- [Core Concepts](guide/concepts.md) - Framework architecture
- [Creating Agents](guide/agents.md) - Agent development
- [Working with Tools](guide/tools.md) - Tool creation
- [Building Workflows](guide/workflows.md) - Workflow orchestration
- [API Reference](api.md) - Complete API documentation

### Example Code

All examples are available in the `examples/` directory:

```
examples/
├── fraud_detection_workflow/    # Complete fraud detection system
├── simple_agent/               # (Coming soon)
├── multi_agent/                # (Coming soon)
└── custom_tools/               # (Coming soon)
```

### Community Examples

Looking for more examples? Check the community contributions:

- [GitHub Discussions](https://github.com/microsoft/agent-framework/discussions)
- [Example Gallery](https://example-gallery.agent-framework.com) _(coming soon)_

---

## Contributing Examples

Have a great example to share? We welcome contributions!

### Contribution Guidelines

1. **Clear use case**: Explain the real-world problem solved
2. **Complete code**: Provide working, tested implementation
3. **Documentation**: Include setup instructions and explanations
4. **Best practices**: Follow framework patterns and conventions
5. **Testing**: Include test cases demonstrating functionality

### How to Submit

1. Fork the repository
2. Create example in `examples/` directory
3. Add documentation to this file
4. Submit pull request with description

---

## Getting Help

Need assistance with these examples?

- **Documentation**: Review the [guide section](guide/concepts.md)
- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/microsoft/agent-framework/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/microsoft/agent-framework/discussions)
- **Support**: Contact the maintainers for enterprise support

---

*Last Updated: January 2025*
