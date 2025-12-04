# Titanic ML Agent - Claude Agent SDK

An autonomous ML agent using the Claude Agent SDK to handle the Titanic survival prediction pipeline.

## Overview

This project demonstrates an autonomous ML agent that:
- Loads and explores the Titanic dataset
- Preprocesses data and engineers features
- Trains classification models
- Generates predictions

The agent autonomously decides which tools to use and in what order.

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the agent:
```bash
python titanic_agent.py
```

This will load data, explore, preprocess, train a Random Forest model, and generate predictions.csv.

## Available Tools

- `load_data`: Load training and test CSV files
- `explore_data`: Analyze dataset statistics and distributions
- `preprocess_data`: Clean data and engineer features (family size, titles, encoding)
- `train_model`: Train Random Forest or Logistic Regression
- `predict`: Generate predictions on test data

## How It Works

The agent uses an agentic loop where Claude autonomously analyzes state, selects tools, executes them, and interprets results until the task is complete.

Example execution:
```
1. load_data → 891 training samples loaded
2. explore_data → Statistics and correlations analyzed
3. preprocess_data → Features engineered
4. train_model → Random Forest trained, CV accuracy: 0.82
5. predict → 418 predictions saved
```

## Output

- `predictions.csv`: Survival predictions for test set
- `execution_trace.json`: Debug trace of all tool calls

## Learn More

- [Claude Agent SDK Documentation](https://platform.claude.com/docs/en/agent-sdk/python)
- [Titanic Competition (Kaggle)](https://www.kaggle.com/c/titanic)

