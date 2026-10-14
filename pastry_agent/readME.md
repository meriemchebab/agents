# Personal Cake Chef

A Gradio chat app where you give it the ingredients you already have and it
returns one to three detailed dessert recipes — each with exact metric and
volumetric measures, step-by-step instructions, and a chef's tip.

Built with LangChain agents, the Gemini model, and Tavily web search.
Thirduni Module project.


### The agent

Built with LangChain's `create_agent`, which handles the model/tool loop:

```python
cake_chef = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
)
```

The agent decides on its own when a recipe needs a fact it does not have and
calls `web_search`. There is no routing logic in Python — the loop is the
agent's job.

### The system prompt

This is the core of the app. `system_prompt` constrains the model to:

- **Use what the user has.** Grounded in their listed ingredients rather than
  suggesting a shopping list first.
- **Stay at three basic pantry staples** (water, salt, sugar) unless the user
  has them — anything beyond that is called out as `[Missing - required]` or
  `[Optional]`.
- **Always give both units** — grams/ml plus cups/tablespoons — since recipes
  get read in both regions and nobody wants to convert mid-bake.
- **Fix a strict output template**, so every suggestion has the same headings:
  name, prep/cook time, yield, ingredients, numbered steps, and a tip. Step
  text explicitly requires oven temperatures in both °C and °F and doneness
  cues described by texture rather than clock time alone.

The prompt also carries fifteen years of invented seniority and a
"bake like a professional" instruction. Whether that measurably improves the
output is worth an A/B test, but it does keep the tone consistent.

### Conversation memory

`InMemorySaver` plus a `thread_id` means the agent recalls earlier turns — ask
for a chocolate cake, then "make it dairy-free" and it adjusts the recipe it
just gave you rather than starting over.

The `thread_id` is a UUID held in `gr.State`, generated on page load. A refresh
starts a new thread, so history does not carry across reloads.

### Response extraction

`respond()` reads the last message and handles both shapes the model can
return — a plain string, or a list of content blocks — so a mixed-type
response never renders as a Python list inside a chat bubble.

---

## Requirements

- Python 3.13 (this repo's venv)
- `gradio`, `langchain`, `langchain-google-genai`, `tavily-python` — all
  already listed in the repo-root `requirements.txt`

## Setup

Add these to the repo-root `.env`:

```dotenv
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Both are required. A missing `GOOGLE_API_KEY` fails at startup; a missing
`TAVILY_API_KEY` fails on the first search.

Get a Gemini key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey),
and a Tavily key from [tavily.com](https://tavily.com).

## Run

```bash
python pastry_agent/personal_chef_cake.py
```

Gradio serves on `http://127.0.0.1:7860` and opens a browser tab.

`demo.launch()` takes no arguments here, so it uses Gradio's defaults. To pin a
port or suppress the auto-open, pass them directly:

```python
demo.launch(server_port=8000, inbrowser=False)
```

## Try it with

- `chocolate, eggs, butter, flour`
- `I have bananas and coconut milk, nothing else`
- `rice, mango, condensed milk` (a three-ingredient dessert, all-pantry)
- `eggs, flour, butter, and my oven only goes to 200C`

## Notes and limitations

- **The model is Gemini** (`gemini-3.6-flash`), not OpenAI, despite
  `OPENAI_API_KEY` appearing in `.env`. That key is for other modules.
- **Conversations are in-process.** `InMemorySaver` holds state in memory, so
  history is lost when the app stops and is not shared between users.
- **No error handling.** A Tavily or Gemini failure surfaces as a raw Gradio
  traceback rather than a friendly message.
- **Search results are not filtered for quality**, so a weak blog post can
  influence the output. The prompt asks the model to be accurate, but it has
  no way to verify a source it just read.
- **Gradio's share link is off** (`launch()` defaults). Do not enable it for
  anything account-bound.