import os

from phi.agent import Agent
from phi.model.groq import Groq
from phi.playground import Playground, serve_playground_app
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

MODEL_ID = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Groq(id=MODEL_ID),
    tools=[DuckDuckGo()],
    storage=SqlAgentStorage(table_name="web_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Groq(id=MODEL_ID),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True,
        )
    ],
    instructions=["Always use tables to display data"],
    storage=SqlAgentStorage(table_name="finance_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    name="Agent Team (Web+Finance)",
    model=Groq(id=MODEL_ID),
    show_tool_calls=True,
    markdown=True,
)

app = Playground(agents=[agent_team]).get_app()

@app.get("/")
def root():
    return {"status": "ok"}

if __name__ == "__main__":
    serve_playground_app("finance_agent_team:app", reload=True)
