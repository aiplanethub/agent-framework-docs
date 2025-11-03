# Examples

Complete, production-ready example demonstrating the Agent Workflow Framework capabilities.

## Overview

This guide walks through a comprehensive fraud detection system that showcases:

- Multi-agent orchestration with specialized roles
- Custom tool creation and integration
- Complex workflow state management
- Real-world data processing patterns
- Streamlit UI for interactive usage

## Prerequisites

Before running this example, ensure you have:

- Framework installed - See [Installation Guide](installation.md)
- Azure OpenAI or OpenAI API access configured
- Python 3.10 or higher
- Basic understanding of async Python

### Environment Setup

Create a `.env` file in the example directory:

```bash
# Azure AI Configuration (recommended)
AZURE_AI_PROJECT_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_AI_MODEL_DEPLOYMENT_NAME=gpt-4o-mini

# OR OpenAI Configuration
# OPENAI_API_KEY=your-api-key-here
# LLM_MODEL=gpt-4o-mini
```

---

## Fraud Detection Workflow

A production-grade multi-agent system for detecting fraudulent transactions through collaborative analysis.

### Use Case

Financial institutions need to analyze transactions for fraud indicators. This system:

1. Loads transaction data from CSV files
2. Performs initial discrepancy analysis
3. Detects suspicious patterns across transactions
4. Calculates risk scores
5. Makes final fraud determinations
6. Generates detailed audit reports

### Real-World Applications

- Banking transaction monitoring
- Insurance claim verification
- E-commerce fraud prevention
- Credit card fraud detection
- Payment processor security

### Architecture Overview

The system uses **five specialized agents** working in a coordinated workflow:

```
Data Loading
     ↓
Fraud Analyst (Initial Analysis)
     ↓
Pattern Detection Agent (Cross-transaction patterns)
     ↓
Risk Scoring Agent (Quantitative assessment)
     ↓
Fraud Classifier (Final determination)
     ↓
Report Generator (Audit documentation)
```

Each agent focuses on a specific aspect of fraud detection, creating a robust multi-layered analysis system.

### Project Structure

```
fraud_detection_workflow/
├── fraud_detection_workflow.py    # Main workflow implementation
├── streamlit_fraud_detection_app.py  # Interactive UI
├── run_streamlit.py               # UI launcher script
├── requirements.txt               # Dependencies
├── .env                          # Configuration
├── sample_source.csv             # Example source data
├── sample_target.csv             # Example target data
└── workflow_data/                # Output directory
    └── reports/                  # Generated reports
```

### Key Components

#### 1. Custom Tools

The workflow includes specialized tools for data operations.

#### 2. Specialized Agents

**Fraud Analyst Agent**
- Role: Initial discrepancy identification
- Focus: Comparing source vs target data
- Output: List of suspicious transactions

**Pattern Detection Agent**
- Role: Cross-transaction pattern analysis
- Focus: Identifying systematic fraud patterns
- Output: Pattern descriptions and affected transactions

**Risk Scoring Agent**
- Role: Quantitative risk assessment
- Focus: Calculating fraud probability scores (0-100)
- Output: Numeric risk scores with justification

**Fraud Classifier Agent**
- Role: Final fraud determination
- Focus: Binary classification (fraud/legitimate)
- Output: Classification with confidence level

**Report Generator Agent**
- Role: Audit documentation
- Focus: Comprehensive analysis summary
- Output: Formatted markdown report

### Running the Example

#### Option 1: Command Line

```bash
# 1. Navigate to example directory
cd fraud_detection_workflow

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your Azure/OpenAI credentials

# 4. Run the workflow
python fraud_detection_workflow.py
```

**Output:**
```
Starting Fraud Detection Workflow...
Agent: fraud_analyst - Analyzing discrepancies...
Agent: pattern_detector - Detecting patterns...
Agent: risk_scorer - Calculating risk scores...
Agent: classifier - Making final determination...
Agent: report_generator - Generating report...

Report saved to: workflow_data/reports/fraud_report_20251103_1230.md
```

#### Option 2: Streamlit Web UI

For an interactive, visual experience:

```bash
# Run the Streamlit app
python run_streamlit.py

# Or directly:
streamlit run streamlit_fraud_detection_app.py
```

**Streamlit UI Features:**
- Upload custom CSV files
- Configure analysis parameters
- Real-time workflow execution monitoring
- Interactive report viewing
- Download generated reports
- Visualize transaction analysis

**UI Components:**
1. **File Upload Section**: Upload source and target CSVs
2. **Configuration Panel**: Adjust detection thresholds
3. **Execution Controls**: Start/stop workflow
4. **Progress Tracking**: Real-time agent status updates
5. **Results Display**: Interactive report viewing
6. **Export Options**: Download reports in multiple formats

### Expected Output

The workflow generates a comprehensive fraud analysis report:

```markdown
# Fraud Detection Report
Generated: 2025-11-03 12:30:45

## Executive Summary
Analyzed 150 transactions
Found 12 suspicious patterns
Flagged 8 transactions as fraudulent

## Detailed Findings

### High-Risk Transactions
- Transaction T001: Risk Score 95/100
  - Multiple account indicators
  - Unusual amount pattern
  - Geographic anomaly

### Detected Patterns
- Pattern 1: Sequential transactions from same IP
- Pattern 2: Amount structuring to avoid limits

## Recommendations
[Action items and next steps]
```

### Configuration Options

**Environment Variables:**

```bash
# Model configuration
LLM_MODEL=gpt-4o-mini              # Model to use
TEMPERATURE=0.7                     # Response creativity (0.0-1.0)
MAX_TOKENS=2000                     # Max response length

# Detection thresholds
RISK_THRESHOLD=70                   # Score for flagging (0-100)
PATTERN_CONFIDENCE=0.8              # Pattern detection confidence

# Output configuration
REPORT_FORMAT=markdown              # Output format (markdown/json)
SAVE_INTERMEDIATE=true              # Save agent intermediate outputs
```

**Customization:**

Modify `fraud_detection_workflow.py` to adjust:
- Agent instructions and behavior
- Workflow execution order
- State management logic
- Tool functionality
- Report formatting

### Key Framework Patterns Demonstrated

**1. Multi-Agent Orchestration**
- Sequential agent execution
- State passing between agents
- Specialized agent roles

**2. Tool Integration**
- Custom tool creation with `@ai_function`
- Tool parameter validation
- Error handling in tools

**3. State Management**
- Workflow state initialization
- State updates by agents
- Final state aggregation

**4. Error Handling**
- Graceful failure handling
- Retry logic for API calls
- Validation of agent outputs

**5. Production Readiness**
- Logging and monitoring
- Configuration management
- Output persistence
- Audit trail generation

### Adapting This Example

Use this fraud detection workflow as a template for:

**Similar Use Cases:**
- Content moderation workflows
- Data validation pipelines
- Multi-stage approval processes
- Quality assurance systems
- Compliance checking

**Key Patterns to Reuse:**
- Multi-agent sequential workflow structure
- Specialized agent roles with focused instructions
- Custom tool creation for domain-specific operations
- State accumulation across workflow stages
- Report generation and audit trails

**Customization Steps:**
1. Replace tools with your domain-specific operations
2. Adjust agent instructions for your use case
3. Modify state structure for your data model
4. Update workflow sequence for your process
5. Customize report format for your requirements

---

## Additional Resources

### Framework Documentation
- [Core Concepts](guide/concepts.md) - Framework architecture
- [Creating Agents](guide/agents.md) - Agent configuration
- [Building Workflows](guide/workflows.md) - Workflow patterns
- [Working with Tools](guide/tools.md) - Tool development

### Example Code
- Full source code: `fraud_detection_workflow/`
- Sample data: Included CSV files
- Streamlit UI: `streamlit_fraud_detection_app.py`

### Community Examples
Looking for more examples? Contributions welcome!

Share your workflow implementations:
1. Fork the repository
2. Add your example with documentation
3. Submit a pull request

## Getting Help

Questions about this example?

- Open an issue on GitHub
- Check the [API Reference](api.md)
- Review [Quick Start Guide](quickstart.md) for basics

---

*Ready to build your own workflow? Start with [Core Concepts](guide/concepts.md) to understand the framework architecture.*
