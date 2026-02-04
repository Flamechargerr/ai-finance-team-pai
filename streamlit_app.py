import json
import os
import re
from datetime import datetime
from typing import Dict, List, Set

import streamlit as st
from dotenv import load_dotenv

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools


load_dotenv()

DEFAULT_MODEL_ID = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_NEWS_RESULTS = 5
DEFAULT_SEARCH_RESULTS = 5
DEFAULT_COMPANY_NEWS_STORIES = 3
DEFAULT_MAX_TICKERS = 6

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

KNOWN_COMPANY_TICKERS: Dict[str, str] = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta": "META",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "netflix": "NFLX",
    "amd": "AMD",
    "intel": "INTC",
    "salesforce": "CRM",
    "oracle": "ORCL",
    "ibm": "IBM",
}


def _inject_css() -> None:
    st.markdown(
        """
<style>
:root { --accent: #0ea5e9; --accent-2: #16a34a; --accent-3: #f97316; }
.hero { background: linear-gradient(135deg, #f6f9ff 0%, #eefcf8 100%); border: 1px solid #e2e8f0; padding: 22px; border-radius: 16px; }
.hero h1 { margin: 0; font-size: 28px; }
.badge { display: inline-block; padding: 6px 10px; border-radius: 999px; color: white; font-size: 12px; margin-right: 6px; }
.badge.primary { background: var(--accent); }
.badge.success { background: var(--accent-2); }
.badge.warn { background: var(--accent-3); }
.card { background: white; border: 1px solid #e5e7eb; border-radius: 14px; padding: 16px; min-height: 120px; }
.muted { color: #6b7280; }
.section-title { font-weight: 700; font-size: 18px; margin-bottom: 6px; }
.kv { font-size: 13px; color: #6b7280; }
.divider { height: 1px; background: #e5e7eb; margin: 16px 0; }
</style>
""",
        unsafe_allow_html=True,
    )


def _normalize_spaced_text(text: str) -> str:
    pattern = r"(?:\b[0-9A-Za-z]\s){3,}[0-9A-Za-z]\b"
    return re.sub(pattern, lambda m: m.group(0).replace(" ", ""), text)


def _maybe_parse_json(value: object) -> object:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] in "[{":
        try:
            return json.loads(stripped)
        except Exception:
            return value
    return value


def _extract_tickers_from_prompt(prompt: str) -> List[str]:
    tickers: Set[str] = set()

    for match in re.findall(r"\$([A-Za-z]{1,5})", prompt):
        tickers.add(match.upper())

    for token in re.findall(r"\b[A-Z]{1,5}\b", prompt):
        if token not in COMMON_TICKER_STOPWORDS:
            tickers.add(token)

    prompt_lower = prompt.lower()
    for name, ticker in KNOWN_COMPANY_TICKERS.items():
        if name in prompt_lower:
            tickers.add(ticker)

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


def _get_manual_tool_outputs(
    prompt: str,
    tickers: List[str],
    include_web_news: bool,
    include_web_search: bool,
    include_finance: bool,
    news_max_results: int,
    search_max_results: int,
    company_news_count: int,
) -> Dict[str, object]:
    data: Dict[str, object] = {"tickers": tickers, "web": {}, "finance": {}}

    ddg = DuckDuckGo()
    if include_web_news:
        try:
            data["web"]["news"] = json.loads(ddg.duckduckgo_news(query=prompt, max_results=news_max_results))
        except Exception as exc:
            data["web"]["news_error"] = f"{exc}"
    if include_web_search:
        try:
            data["web"]["search"] = json.loads(ddg.duckduckgo_search(query=prompt, max_results=search_max_results))
        except Exception as exc:
            data["web"]["search_error"] = f"{exc}"

    if include_finance:
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
                ticker_data["company_info"] = _maybe_parse_json(info)
            except Exception as exc:
                ticker_data["company_info_error"] = f"{exc}"
            try:
                recs = yf_tools.get_analyst_recommendations(ticker)
                ticker_data["analyst_recommendations"] = _maybe_parse_json(recs)
            except Exception as exc:
                ticker_data["analyst_recommendations_error"] = f"{exc}"
            try:
                news = yf_tools.get_company_news(ticker, num_stories=company_news_count)
                ticker_data["company_news"] = _maybe_parse_json(news)
            except Exception as exc:
                ticker_data["company_news_error"] = f"{exc}"
            data["finance"][ticker] = ticker_data

    return data


def _build_summary_prompt(prompt: str, tickers: List[str], tool_data: Dict[str, object]) -> str:
    tickers_line = ", ".join(tickers) if tickers else "None detected"
    return (
        "You are a production-grade finance analyst agent. Summarize the following data and answer the user prompt. "
        "Use tables for comparisons. If news items include links, cite them. "
        "Do not insert extra spaces between letters. "
        "If tickers are missing, explain that you used general web/news context only.\n\n"
        f"User prompt:\n{prompt}\n\n"
        f"Detected tickers: {tickers_line}\n\n"
        f"Tool data (JSON):\n{json.dumps(tool_data, indent=2)}"
    )


@st.cache_resource(show_spinner=False)
def get_summarizer(model_id: str) -> Agent:
    return Agent(
        name="Summarizer",
        role="Summarize tool outputs and answer user questions",
        model=Groq(id=model_id),
        markdown=True,
    )


def _count_items(value: object) -> int:
    if isinstance(value, list):
        return len(value)
    return 0


def _render_hero() -> None:
    st.markdown(
        """
<div class="hero">
  <div>
    <span class="badge primary">AI Agent System</span>
    <span class="badge success">Manual Tools (No-Fail)</span>
    <span class="badge warn">Groq Summarizer</span>
  </div>
  <h1>AI Finance Agent Team</h1>
  <p class="muted">Production-minded agentic workflow that combines live web/news + market data with Groq reasoning.</p>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_how_it_works() -> None:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("**How this AI agent works**")
    cols = st.columns(3)
    cols[0].markdown(
        "<div class='card'><div class='section-title'>1) Collect</div><div class='kv'>DuckDuckGo + Yahoo Finance</div><div class='muted'>We fetch news, market data, and company info directly in code for reliability.</div></div>",
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        "<div class='card'><div class='section-title'>2) Structure</div><div class='kv'>Ticker-aware parsing</div><div class='muted'>We extract or resolve tickers, normalize data, and build a clean context block.</div></div>",
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        "<div class='card'><div class='section-title'>3) Reason</div><div class='kv'>Groq LLM summary</div><div class='muted'>The model focuses on analysis and formatting, not tool calls.</div></div>",
        unsafe_allow_html=True,
    )


def _render_production_note() -> None:
    st.markdown(
        "<div class='card'><div class='section-title'>Production-ready mindset</div>"
        "<div class='muted'>Deterministic tool execution, graceful failures, and clear outputs. "
        "Use $TICKER (e.g., $AAPL) for maximum accuracy.</div></div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="AI Finance Agent Team", layout="wide")
    _inject_css()

    _render_hero()
    _render_how_it_works()
    _render_production_note()

    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_tools" not in st.session_state:
        st.session_state.last_tools = None
    if "last_tickers" not in st.session_state:
        st.session_state.last_tickers = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "last_run_at" not in st.session_state:
        st.session_state.last_run_at = None

    with st.sidebar:
        st.subheader("Settings")
        model_id = st.text_input("Model", value=DEFAULT_MODEL_ID)
        st.caption("Override with the GROQ_MODEL environment variable.")
        if not os.getenv("GROQ_API_KEY"):
            st.warning("GROQ_API_KEY is not set. The app will fail without it.")

        include_finance = st.checkbox("Include finance data", value=True)
        include_web_news = st.checkbox("Include web news", value=True)
        include_web_search = st.checkbox("Include web search", value=True)

        news_max_results = st.slider("News results", 1, 10, DEFAULT_NEWS_RESULTS)
        search_max_results = st.slider("Search results", 1, 10, DEFAULT_SEARCH_RESULTS)
        company_news_count = st.slider("Company news per ticker", 1, 10, DEFAULT_COMPANY_NEWS_STORIES)
        max_tickers = st.slider("Max tickers", 1, 12, DEFAULT_MAX_TICKERS)
        show_raw_data = st.toggle("Show raw data", value=False)

        if st.session_state.last_tickers:
            st.markdown(f"**Detected tickers:** {', '.join(st.session_state.last_tickers)}")
        else:
            st.markdown("**Detected tickers:** None")

        if st.session_state.last_run_at:
            st.caption(f"Last run: {st.session_state.last_run_at}")

        if st.button("Clear chat"):
            st.session_state.history = []
            st.session_state.last_tools = None
            st.session_state.last_tickers = []

    summarizer = get_summarizer(model_id)

    st.markdown("Tip: Use $TICKER (e.g., $AAPL, $MSFT) for best finance accuracy.")

    quick_cols = st.columns(3)
    if quick_cols[0].button("Compare Apple vs Microsoft"):
        st.session_state.pending_prompt = "Compare Apple and Microsoft on valuation and recent news."
    if quick_cols[1].button("Summarize Tesla earnings"):
        st.session_state.pending_prompt = "Summarize Tesla's latest earnings and analyst sentiment."
    if quick_cols[2].button("Top AI stocks this week"):
        st.session_state.pending_prompt = "What's happening in AI stocks this week?"

    last_tools = st.session_state.last_tools or {}
    news_count = _count_items(last_tools.get("web", {}).get("news"))
    search_count = _count_items(last_tools.get("web", {}).get("search"))
    tickers_count = len(st.session_state.last_tickers)

    metrics = st.columns(3)
    metrics[0].metric("Tickers", tickers_count)
    metrics[1].metric("News items", news_count)
    metrics[2].metric("Search results", search_count)

    for item in st.session_state.history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if show_raw_data and item["role"] == "assistant" and item.get("tools"):
                with st.expander("Raw data"):
                    st.json(item["tools"])

    prompt = st.chat_input("Ask a finance question...")
    if not prompt and st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if prompt:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                tickers = _extract_tickers_from_prompt(prompt)
                if not tickers:
                    tickers = _try_resolve_tickers_from_names(prompt)
                tickers = tickers[:max_tickers]

                tool_data = _get_manual_tool_outputs(
                    prompt=prompt,
                    tickers=tickers,
                    include_web_news=include_web_news,
                    include_web_search=include_web_search,
                    include_finance=include_finance,
                    news_max_results=news_max_results,
                    search_max_results=search_max_results,
                    company_news_count=company_news_count,
                )
                summary_prompt = _build_summary_prompt(prompt, tickers, tool_data)

                try:
                    response = summarizer.run(summary_prompt, stream=False)
                    content = response.get_content_as_string()
                    content = _normalize_spaced_text(content)
                except Exception as exc:
                    content = (
                        "I couldn't generate a summary, but I did gather the raw data below. "
                        f"Error: {exc}"
                    )
                st.markdown(content)
                if show_raw_data:
                    with st.expander("Raw data"):
                        st.json(tool_data)

                st.session_state.last_tools = tool_data
                st.session_state.last_tickers = tickers
                st.session_state.last_run_at = datetime.now().strftime("%b %d, %Y %I:%M %p")

        st.session_state.history.append(
            {
                "role": "assistant",
                "content": content,
                "tools": tool_data,
            }
        )


if __name__ == "__main__":
    main()
