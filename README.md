# Multi-Agent AutoML Team

An autonomous AI Data Science Team consisting of three specialized LLM agents that collaborate to process datasets, engineer features, and train machine learning models.

## Overview

Unlike traditional AutoML systems that run predefined grid searches, this system features **three distinct LLM agents** that reason about data, make semantic decisions, and write their own code.

### The Three Agents

1. **Agent 1: The Data Cleaner** ("The Auditor")
   - Audits data quality
   - Handles missing values and outliers
   - Makes intelligent decisions about data types and column removal

2. **Agent 2: The Feature Engineer** ("The Architect")
   - Creates new interaction features
   - Encodes categorical variables
   - Performs feature selection to maximize information density

3. **Agent 3: The Model Trainer** ("The Coder")
   - Generates Python code to train XGBoost models
   - Executes code and evaluates performance
   - **Implements a feedback loop**: iteratively improves the model until satisfied

## Architecture

The system uses a **sequential handoff workflow**:

```
Raw Data → Agent 1 → Clean Data → Agent 2 → Engineered Data → Agent 3 → Trained Model
           (Audit)    +Report     (Create)    +Report          (Train)    +Metrics
```

Each agent:
- Receives data and a report from the previous agent
- Uses tools to analyze and transform the data
- **Asks an LLM to decide** what actions to take (NOT hardcoded!)
- Executes those decisions
- Passes results to the next agent


## Key Features

### 1. LLM-Driven Decision Making
- **No hardcoded rules**: The LLM decides what to do based on the data
- **Adaptive behavior**: Different datasets get different treatments
- **Explainable reasoning**: Every decision is logged with explanations

### 2. The Feedback Loop (Agent 3)
The most critical feature: Agent 3 doesn't just train once—it iterates!

### 3. Comprehensive Logging
Every action is logged with:
- Timestamp
- Agent name
- Action taken
- LLM reasoning
- Results

### 4. Structured Reports
Each agent generates:
- JSON report with structured data
- Human-readable summary
- List of actions taken
- Reasoning for decisions

## Usage Examples

### Example 1: Sample Data
```bash
python test_full_pipeline.py
```

### Example 2: Custom Data
```bash
python main.py your_data.csv target_column_name
```

### Example 3: Individual Testing
```bash
python test_agent1.py  # test cleaning
python test_agent2.py  # test feature engineering
python test_agent3.py  # test model training with feedback loop
```

### Deliverables 

#### 1. System Code 
All files created and working:
- `config.py` - OpenAI configuration
- `logger.py` - logging system
- `base_agent.py` - Base agent class and tool registry
- `handoff.py` - Data transfer mechanism
- `pipeline.py` - Pipeline orchestrator
- `agent_data_cleaner.py` - Agent 1 implementation
- `agent_feature_engineer.py` - Agent 2 implementation
- `agent_model_trainer.py` - Agent 3 implementation
- `main.py` - Entry point
- Test scripts for each agent + full pipeline

#### 2. Execution Logs 
Generated automatically:
- `outputs/agent_execution.log` - Complete agent conversation
- Shows reasoning: "I see 90% missing values in 'x_feature', I will drop it"
- Timestamped actions for all decisions
- LLM reasoning for each action

#### 3. Final Report 
- `outputs/final_report.md` - Markdown summary
- Shows transformation: Raw Data → Clean → Engineered → Final Metrics
- Individual agent summaries
- Complete action log