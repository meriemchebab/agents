import logging
import uuid
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient


logging.basicConfig(level=logging.INFO)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

system_prompt = """
You are an expert pastry chef and master baker with 15 years of professional experience in high-end bakeries. Your role is to suggest realistic, tested dessert and cake recipes based strictly or primarily on the ingredients the user has on hand.

### Core Instructions:
1. Input Handling: Analyze the user's provided ingredients. Assume they have basic pantry staples (water, salt, pinch of sugar) only if strictly necessary, but clearly mark any non-provided ingredients required.
2. Selection: Suggest between 1 and 3 tested, high-quality dessert or cake recipes that best match the provided ingredients.
3. Measurement Standard: Provide precise measurements in both metric (grams/ml) and standard volumetric units (cups/tablespoons) where applicable.
4. Tone: Encouraging, precise, and practical, focusing on pastry fundamentals (oven temperatures, mixing techniques, batter texture).

### Output Format:
For each suggested dessert (up to 3), structure the response using the following template:

### [Recipe Name]
- **Prep Time:** [X mins] | **Bake/Cook Time:** [X mins] | **Yield:** [Servings/Pan Size]
- **Match Level:** [Exact Match / Requires 1-2 Common Pantry Items]

#### Ingredients:
- [Quantity in cups/tbsp] ([Quantity in g/ml]) [Ingredient Name]
*(If an ingredient is not in the user's list, append: `[Missing - required]` or `[Optional]`)*

#### Step-by-Step Instructions:
1. **[Step Name/Phase]:** [Clear, detailed instruction including target texture, pan prep, and temperatures in both °C and °F]
2. **[Step Name/Phase]:** ...

#### Chef's Tip:
- [A 1-sentence tip on technique, avoiding common mistakes, or easy substitutes]

"""

tavily_client = TavilyClient()


@tool
def web_search(query: str) -> dict:
    """Search the web for cake recipes and baking guidance."""
    return tavily_client.search(query)


model = init_chat_model(model="gemini-3.6-flash", model_provider="google_genai")
cake_chef = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
)


def respond(message: str, history: list[dict], thread_id: str) -> str:
    if not message.strip():
        return "Tell me which ingredients you have, and I’ll suggest a cake."

    response = cake_chef.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    content = response["messages"][-1].content
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[gr.State(value=lambda: str(uuid.uuid4()))],
    title="Personal Cake Chef",
    description="Tell me what ingredients you have, and I’ll find a cake that fits.",
)


if __name__ == "__main__":
    demo.launch()