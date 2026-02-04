# AI Finance Agent Team (Groq)

A multi-agent finance assistant that pairs web search with market data tools, orchestrated by a team agent and served through a Streamlit UI (with an optional Playground runner).

## What this does
- Answers finance questions by combining live web context with Yahoo Finance data
- Splits work across two specialists: a Web Agent and a Finance Agent
- Uses Groq models for fast inference

## Architecture
- Web Agent: searches the web using DuckDuckGo
- Finance Agent: pulls price, company info, recommendations, and news via `yfinance`
- Team Agent: routes tasks to the best specialist and composes a single response
- Storage: local SQLite (`agents.db`) for agent memory

## Quick start (Streamlit UI)
1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set your Groq API key:
```bash
export GROQ_API_KEY="your_groq_api_key"
```
Or create a `.env` file (recommended for local dev):
```bash
cp .env.example .env
```
Then edit `.env` and set `GROQ_API_KEY`.
4. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```
5. Open the URL Streamlit prints in the terminal.

## Optional: Playground UI
If you prefer the `phi` Playground:
```bash
python3 finance_agent_team.py
```
Then open the Playground URL printed in the terminal.

## Environment variables
- `GROQ_API_KEY`: required for Groq model access
- `GROQ_MODEL`: optional model override (default: `llama-3.3-70b-versatile`)

Example:
```bash
export GROQ_MODEL="llama-3.3-70b-versatile"
```

## Example prompts
Try these in the UI:
- "Compare Apple and Microsoft on valuation and recent news."
- "Summarize Tesla's latest earnings and analyst sentiment."
- "Which cloud stocks have the best recent momentum?"

## Project structure
- `streamlit_app.py`: Streamlit UI entrypoint
- `finance_agent_team.py`: Playground entrypoint and agent definitions
- `requirements.txt`: Python dependencies
- `LICENSE`: Apache 2.0 license

## Customization
Common tweaks you might want:
- Change the model: set `GROQ_MODEL` or edit `MODEL_ID` in `finance_agent_team.py`.
- Add tools: extend the `tools` list on either agent.
- Add a new agent: create another `Agent` and include it in the `team` list.
- Reset memory: delete `agents.db` to clear saved history.

## Troubleshooting
- Missing API key: ensure `GROQ_API_KEY` is set in your shell.
- Dependency issues: re-run `pip install -r requirements.txt` inside your active venv.
- Network errors: confirm you can reach Groq, DuckDuckGo, and Yahoo Finance from your network.
- Streamlit tool failures: the Streamlit UI executes tools directly and should not fail due to model tool-calls.

## Security notes
- Do not commit API keys or secrets to git.
- If you share this repo, provide instructions to set `GROQ_API_KEY` via environment variables.

## Attribution
This project is adapted from the `ai_finance_agent_team` example in Shubham Saboo's awesome-llm-apps repository and is distributed under the Apache 2.0 License.
