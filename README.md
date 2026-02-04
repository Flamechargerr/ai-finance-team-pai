# AI Finance Agent Team (Groq)

AI finance agent team with web access and financial data tools, built with the `phi` agent framework and Groq models.

## Features
- Web agent that searches the internet
- Finance agent that fetches market data and company info via Yahoo Finance
- Team agent that orchestrates both agents
- Playground UI for interactive use

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your Groq API key:
   ```bash
   export GROQ_API_KEY="your_groq_api_key"
   ```
3. (Optional) Override the default model:
   ```bash
   export GROQ_MODEL="llama-3.3-70b-versatile"
   ```

## Run
```bash
python3 finance_agent_team.py
```

Then open the Playground URL shown in the terminal.

## Attribution
This project is adapted from the `ai_finance_agent_team` example in Shubham Saboo's awesome-llm-apps repository and is distributed under the Apache 2.0 License.
