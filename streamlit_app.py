import os

import streamlit as st

from phi.agent import Agent
from phi.model.groq import Groq
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools


DEFAULT_MODEL_ID = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


@st.cache_resource(show_spinner=False)
def get_agent_team(model_id: str) -> Agent:
    web_agent = Agent(
        name="Web Agent",
        role="Search the web for information",
        model=Groq(id=model_id),
        tools=[DuckDuckGo()],
        storage=SqlAgentStorage(table_name="web_agent", db_file="agents.db"),
        add_history_to_messages=True,
        markdown=True,
    )

    finance_agent = Agent(
        name="Finance Agent",
        role="Get financial data",
        model=Groq(id=model_id),
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
        model=Groq(id=model_id),
        show_tool_calls=True,
        markdown=True,
    )

    return agent_team


def main() -> None:
    st.set_page_config(page_title="AI Finance Agent Team", layout="wide")
    st.title("AI Finance Agent Team (Groq)")
    st.caption("Web + Finance multi-agent team with Groq models.")

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.subheader("Settings")
        model_id = st.text_input("Model", value=DEFAULT_MODEL_ID)
        st.caption("Override with the GROQ_MODEL environment variable.")
        if not os.getenv("GROQ_API_KEY"):
            st.warning("GROQ_API_KEY is not set. The app will fail without it.")
        if st.button("Clear chat"):
            st.session_state.history = []

    agent_team = get_agent_team(model_id)

    for item in st.session_state.history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant" and item.get("tools"):
                with st.expander("Tool calls"):
                    st.json(item["tools"])

    prompt = st.chat_input("Ask a finance question...")
    if prompt:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent_team.run(prompt, stream=False)
                content = response.get_content_as_string()
                st.markdown(content)
                if response.tools:
                    with st.expander("Tool calls"):
                        st.json(response.tools)

        st.session_state.history.append(
            {
                "role": "assistant",
                "content": content,
                "tools": response.tools,
            }
        )


if __name__ == "__main__":
    main()
