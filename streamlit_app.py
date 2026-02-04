import json
import os
import re
from typing import Dict, List, Set

import streamlit as st
from dotenv import load_dotenv

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools


load_dotenv()

DEFAULT_MODEL_ID = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

COMMON_TICKER_STOPWORDS: Set[str] = {
    "A",
    "AN",
    "AND",
    "ARE",
    "AS",
    "AT",
    "BE",
    "BY",
    "CEO",
    "CFO",
    "COO",
    "EPS",
    "ETF",
    "ETFS",
    "FOR",
    "FROM",
    "GDP",
    "IN",
    "INC",
    "IS",
    "IT",
    "LA",
    "LLC",
    "LTD",
    "OF",
    "ON",
    "OR",
    "Q",
    "THE",
    "TO",
    "US",
    "USA",
    "USD",
    "WEEK",
    "YEAR",
}


def _extract_tickers_from_prompt(prompt: str) -> List[str]:
    tickers: Set[str] = set()

    # $TICKER format
    for match in re.findall(r"\\$([A-Za-z]{1,5})", prompt):
        tickers.add(match.upper())

    # Uppercase tokens 1-5 chars (basic heuristic)
    for token in re.findall(r"\\b[A-Z]{1,5}\\b", prompt):
        if token not in COMMON_TICKER_STOPWORDS:
            tickers.add(token)

    return sorted(tickers)


def _try_resolve_tickers_from_names(prompt: str) -> List[str]:
    try:
        from yfinance.search import Search

        search = Search(prompt, max_results=5)
        quotes = getattr(search, "quotes", []) or []
        tickers = [q.get("symbol") for q in quotes if q.get("symbol")]
        return sorted(set(tickers))
    except Exception:
        return []


def _get_manual_tool_outputs(prompt: str, tickers: List[str]) -> Dict[str, object]:
    data: Dict[str, object] = {"tickers": tickers, "web": {}, "finance": {}}

    ddg = DuckDuckGo()
    try:
        data["web"]["news"] = json.loads(ddg.duckduckgo_news(query=prompt, max_results=5))
    except Exception as exc:
        data["web"]["news_error"] = f"{exc}"
    try:
        data["web"]["search"] = json.loads(ddg.duckduckgo_search(query=prompt, max_results=5))
    except Exception as exc:
        data["web"]["search_error"] = f"{exc}"

    yf_tools = YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        company_info=True,
        company_news=True,
    )
    for ticker in tickers:
        ticker_data: Dict[str, object] = {}
        try:
            ticker_data["price"] = yf_tools.get_current_stock_price(ticker)
        except Exception as exc:
            ticker_data["price_error"] = f"{exc}"
        try:
            info = yf_tools.get_company_info(ticker)
            ticker_data["company_info"] = json.loads(info) if info and info.startswith("{") else info
        except Exception as exc:
            ticker_data["company_info_error"] = f"{exc}"
        try:
            ticker_data["analyst_recommendations"] = yf_tools.get_analyst_recommendations(ticker)
        except Exception as exc:
            ticker_data["analyst_recommendations_error"] = f"{exc}"
        try:
            ticker_data["company_news"] = yf_tools.get_company_news(ticker)
        except Exception as exc:
            ticker_data["company_news_error"] = f"{exc}"
        data["finance"][ticker] = ticker_data

    return data


def _build_summary_prompt(prompt: str, tickers: List[str], tool_data: Dict[str, object]) -> str:
    tickers_line = ", ".join(tickers) if tickers else "None detected"
    return (
        "You are a finance analyst. Summarize the following data and answer the user prompt. "
        "Use tables for comparisons. If news items include links, cite them. "
        "If tickers are missing, explain that you used general web/news context only.\\n\\n"
        f"User prompt:\\n{prompt}\\n\\n"
        f"Detected tickers: {tickers_line}\\n\\n"
        f"Tool data (JSON):\\n{json.dumps(tool_data, indent=2)}"
    )


@st.cache_resource(show_spinner=False)
def get_summarizer(model_id: str) -> Agent:
    return Agent(
        name="Summarizer",
        role="Summarize tool outputs and answer user questions",
        model=Groq(id=model_id),
        markdown=True,
    )


def main() -> None:
    st.set_page_config(page_title="AI Finance Agent Team", layout="wide")
    st.title("AI Finance Agent Team (Groq)")
    st.caption("Web + Finance assistant with manual tools and Groq summarization.")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_tools" not in st.session_state:
        st.session_state.last_tools = None
    if "last_tickers" not in st.session_state:
        st.session_state.last_tickers = []

    with st.sidebar:
        st.subheader("Settings")
        model_id = st.text_input("Model", value=DEFAULT_MODEL_ID)
        st.caption("Override with the GROQ_MODEL environment variable.")
        if not os.getenv("GROQ_API_KEY"):
            st.warning("GROQ_API_KEY is not set. The app will fail without it.")
        if st.session_state.last_tickers:
            st.markdown(f"**Detected tickers:** {', '.join(st.session_state.last_tickers)}")
        else:
            st.markdown("**Detected tickers:** None")
        if st.button("Clear chat"):
            st.session_state.history = []
            st.session_state.last_tools = None
            st.session_state.last_tickers = []

    summarizer = get_summarizer(model_id)

    for item in st.session_state.history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant" and item.get("tools"):
                with st.expander("Raw data"):
                    st.json(item["tools"])

    prompt = st.chat_input("Ask a finance question...")
    if prompt:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                tickers = _extract_tickers_from_prompt(prompt)
                if not tickers:
                    tickers = _try_resolve_tickers_from_names(prompt)

                tool_data = _get_manual_tool_outputs(prompt, tickers)
                summary_prompt = _build_summary_prompt(prompt, tickers, tool_data)

                try:
                    response = summarizer.run(summary_prompt, stream=False)
                    content = response.get_content_as_string()
                except Exception as exc:
                    content = (
                        "I couldn't generate a summary, but I did gather the raw data below. "
                        f"Error: {exc}"
                    )
                st.markdown(content)
                with st.expander("Raw data"):
                    st.json(tool_data)

                st.session_state.last_tools = tool_data
                st.session_state.last_tickers = tickers

        st.session_state.history.append(
            {
                "role": "assistant",
                "content": content,
                "tools": tool_data,
            }
        )


if __name__ == "__main__":
    main()
