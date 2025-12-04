"""
Titanic ML Agent using Claude Agent SDK
Minimal implementation with tool-based approach
"""

from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeAgentOptions, ClaudeSDKClient
import os
from ml_tools import TitanicMLTools


# Initialize ML tools globally
ml = TitanicMLTools()


# Define tools
@tool("load_data", "Load train and test CSV files", {"train_path": str, "test_path": str})
async def load_data(args):
    result = ml.load_data(args['train_path'], args['test_path'])
    return {"content": [{"type": "text", "text": str(result)}]}


@tool("explore", "Explore dataset statistics", {"dataset": str})
async def explore(args):
    result = ml.explore_data(args['dataset'])
    return {"content": [{"type": "text", "text": str(result)}]}


@tool("preprocess", "Preprocess and engineer features", {"operations": list})
async def preprocess(args):
    result = ml.preprocess_data(args['operations'])
    return {"content": [{"type": "text", "text": str(result)}]}


@tool("train", "Train ML model", {"model_type": str, "n_estimators": int, "max_depth": int})
async def train(args):
    params = {k: v for k, v in args.items() if k in ['n_estimators', 'max_depth'] and v}
    result = ml.train_model(args['model_type'], params or None)
    return {"content": [{"type": "text", "text": str(result)}]}


@tool("predict", "Generate predictions", {"output_path": str})
async def predict(args):
    result = ml.predict(args['output_path'])
    return {"content": [{"type": "text", "text": str(result)}]}


# Create MCP server
server = create_sdk_mcp_server(
    name="titanic",
    version="1.0.0",
    tools=[load_data, explore, preprocess, train, predict]
)


async def run_agent(train_path: str, test_path: str, output_path: str):
    """Run the agent"""
    options = ClaudeAgentOptions(
        mcp_servers={"titanic": server},
        allowed_tools=["mcp__titanic__load_data", "mcp__titanic__explore",
                      "mcp__titanic__preprocess", "mcp__titanic__train", "mcp__titanic__predict"],
        system_prompt=f"""Build Titanic ML pipeline:
1. Load data from {train_path} and {test_path}
2. Explore train data (check missing values)
3. Preprocess with: fill_age, fill_embarked, fill_fare, family_size, is_alone, encode_sex, encode_embarked, extract_title
4. Train 3 models and compare: random_forest (n_estimators=100, max_depth=5), logistic_regression, gradient_boosting (n_estimators=100, max_depth=3)
5. Use best model to predict to {output_path}"""
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("Execute the ML pipeline.")
        async for msg in client.receive_response():
            if hasattr(msg, 'content'):
                for block in msg.content:
                    if hasattr(block, 'text'):
                        print(block.text)


async def main():
    await run_agent(
        train_path="g:/My Drive/AZ/llm_evo/titanic/data/train.csv",
        test_path="g:/My Drive/AZ/llm_evo/titanic/data/test.csv",
        output_path="g:/My Drive/AZ/llm_evo/titanic/predictions.csv"
    )


if __name__ == "__main__":
    import anyio
    anyio.run(main)
