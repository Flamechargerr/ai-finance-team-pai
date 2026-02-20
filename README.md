# AI Finance Agent Team (Groq)

A multi-agent finance assistant that pairs web search with market data tools, orchestrated by a team agent and served through a Streamlit UI (with an optional Playground runner).

## What this does
- Answers finance questions by combining live web/news context with Yahoo Finance data
- Uses Groq to reason and synthesize results into decision-ready summaries
- Streamlit mode executes tools directly for reliability (no model tool-calls)
- Includes an Investment Compare mode (single-agent) for two-ticker analysis
- Designed with a production-ready posture: deterministic tool execution, clear outputs, and graceful failures

## Architecture (AI Agent System)
- **Collect**: DuckDuckGo + Yahoo Finance gather live web/news and market data
- **Structure**: Ticker-aware parsing normalizes data into a clean context block
- **Reason**: Groq summarizes and compares results in a reliable, production-minded flow

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

## Investment Compare (single-agent mode)
1. Open the **Investment Compare** tab in Streamlit.
2. Enter two tickers (e.g., `AAPL` and `MSFT`).
3. Optional: add a focus area (valuation, growth, risk, catalysts).
4. Run the comparison to get a structured, Groq-summarized report.

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
Investment Compare tab:
- "AAPL vs MSFT"
Tip: use $TICKER (e.g., $AAPL, $MSFT) for best finance accuracy.

## Streamlit UX features
- Quick prompt buttons for common queries
- Investment Compare tab for two-ticker analysis
- Detected tickers badge in the sidebar
- Toggles for web/news/finance sources
- Adjustable result counts and max tickers
- Optional raw data viewer
- Quick prompts and KPI metrics for fast iteration

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
This project is adapted from the `ai_finance_agent_team` and `ai_investment_agent` examples in Shubham Saboo's awesome-llm-apps repository and is distributed under the Apache 2.0 License.

## One-command start

From the project directory:

```bash
./run_all.sh
```

This command will:
- create `.venv` if needed
- install dependencies from `requirements.txt`
- load `.env`
- start Streamlit on `http://localhost:8501`
 Minor update 1
 Minor update 2
 Minor update 3
 Minor update 4
 Minor update 5
 Minor update 6
 Minor update 7
 Minor update 8
 Minor update 9
 Minor update 10
 Minor update 11
 Minor update 12
 Minor update 13
 Minor update 14
 Minor update 15
 Minor update 16
 Minor update 17
 Minor update 18
 Minor update 19
 Minor update 20
 Minor update 21
 Minor update 22
 Minor update 23
 Minor update 24
 Minor update 25
 Minor update 26
 Minor update 27
 Minor update 28
 Minor update 29
 Minor update 30
 Minor update 31
 Minor update 32
 Minor update 33
 Minor update 34
 Minor update 35
 Update 1
 Update 2
 Update 3
 Update 4
 Update 5
 Update 6
 Update 7
 Update 8
 Update 9
 Update 10
 Update 11
 Update 12
 Update 13
 Update 14
 Update 15
 Update 16
 Update 17
 Update 18
 Update 19
 Update 20
 Update 21
 Update 22
 Update 23
 Update 24
 Update 25
 Update 26
 Update 27
 Update 28
