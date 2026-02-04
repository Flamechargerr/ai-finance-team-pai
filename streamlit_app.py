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

KNOWN_COMPANY_MULTI_TICKERS: Dict[str, List[str]] = {
    "tata": ["TCS.NS", "TATAMOTORS.NS", "TATASTEEL.NS"],
}

QUERY_EXPANSIONS: Dict[str, str] = {
    "tata": "Tata Group",
}


def _inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');

:root {
  --panel: #ffffff;
  --border: #e5e7eb;
  --ink: #0f172a;
  --muted: #64748b;
  --accent: #2563eb;
  --accent-2: #16a34a;
  --accent-3: #f59e0b;
}

.hero {
  font-family: "Space Grotesk", sans-serif;
  background: linear-gradient(135deg, rgba(238,242,255,0.96) 0%, rgba(236,254,255,0.96) 100%);
  border: 1px solid var(--border);
  padding: 24px;
  border-radius: 18px;
}

.hero h1 { margin: 8px 0 6px; font-size: 28px; color: var(--ink); }
.hero p { margin: 0; color: var(--muted); }

.chip { display: inline-block; padding: 6px 10px; border-radius: 999px; font-size: 12px; margin-right: 6px; color: white; }
.chip.primary { background: var(--accent); }
.chip.success { background: var(--accent-2); }
.chip.warn { background: var(--accent-3); }

.card { font-family: "Space Grotesk", sans-serif; background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 16px; }
.card-title { font-weight: 700; font-size: 16px; margin-bottom: 6px; color: var(--ink); }
.card-muted { color: var(--muted); font-size: 13px; }

.section-title { font-family: "Space Grotesk", sans-serif; font-weight: 700; font-size: 18px; margin-bottom: 8px; color: var(--ink); }

.divider { height: 1px; background: var(--border); margin: 16px 0; }

.note { font-family: "Space Grotesk", sans-serif; background: #f8fafc; border: 1px dashed var(--border); padding: 12px; border-radius: 12px; color: var(--muted); }

.stat { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 12px; }
.stat-label { font-size: 12px; color: var(--muted); }
.stat-value { font-size: 20px; font-weight: 700; color: var(--ink); }

.pipeline { display: flex; gap: 16px; }
.pipeline-step { flex: 1; display: flex; gap: 10px; align-items: flex-start; }
.step { font-weight: 700; font-size: 12px; color: white; background: var(--accent); border-radius: 999px; padding: 4px 8px; }
.clean-list { margin: 8px 0 0 18px; color: var(--muted); font-size: 13px; }
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

    for name, tickers_list in KNOWN_COMPANY_MULTI_TICKERS.items():
        if name in prompt_lower:
            tickers.update(tickers_list)

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


def _build_queries(prompt: str, tickers: List[str]) -> Dict[str, str]:
    prompt_clean = prompt.strip()
    prompt_lower = prompt_clean.lower()
    query_base = prompt_clean

    for key, value in QUERY_EXPANSIONS.items():
        if key in prompt_lower:
            query_base = value
            break

    if not tickers:
        if "company" not in query_base.lower() and len(query_base.split()) <= 2:
            query_base = f"{query_base} company"

    news_query = query_base if "news" in query_base.lower() else f"{query_base} news"

    return {"search": query_base, "news": news_query}


def _item_has_keywords(item: object, keywords: Set[str]) -> bool:
    if not keywords:
        return True
    if isinstance(item, dict):
        text = " ".join(str(item.get(k, "")) for k in ["title", "body", "snippet", "description"]).lower()
        return any(keyword in text for keyword in keywords)
    if isinstance(item, str):
        return any(keyword in item.lower() for keyword in keywords)
    return False


def _filter_items(items: object, keywords: Set[str]) -> List[object]:
    if not isinstance(items, list):
        return []
    return [item for item in items if _item_has_keywords(item, keywords)]


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
    queries = _build_queries(prompt, tickers)
    keyword_set = {token.lower() for token in re.findall(r"[A-Za-z]{3,}", prompt)}

    ddg = DuckDuckGo()
    if include_web_news:
        try:
            news_items = json.loads(ddg.duckduckgo_news(query=queries["news"], max_results=news_max_results))
            data["web"]["news"] = news_items
            data["web"]["news_filtered"] = _filter_items(news_items, keyword_set)
        except Exception as exc:
            data["web"]["news_error"] = f"{exc}"
    if include_web_search:
        try:
            search_items = json.loads(ddg.duckduckgo_search(query=queries["search"], max_results=search_max_results))
            data["web"]["search"] = search_items
            data["web"]["search_filtered"] = _filter_items(search_items, keyword_set)
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

    data["queries"] = queries
    return data


def _build_summary_prompt(prompt: str, tickers: List[str], tool_data: Dict[str, object]) -> str:
    tickers_line = ", ".join(tickers) if tickers else "None detected"
    return (
        "You are a production-grade finance analyst agent. Summarize the following data and answer the user prompt. "
        "Use tables for comparisons. If news items include links, cite them. "
        "If web/news data looks unrelated, rely on general knowledge and say it's a best-effort answer. "
        "If tickers are missing, explain you used general web/news context only. "
        "If the prompt refers to a group (e.g., Tata), mention the major listed tickers used. "
        "Do not insert extra spaces between letters.\n\n"
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


def _render_header(api_ok: bool, model_id: str, tickers_count: int, news_count: int, search_count: int) -> None:
    col1, col2 = st.columns([3, 1], gap="large")
    with col1:
        status_badge = "Ready" if api_ok else "API key missing"
        st.markdown(
            f"""
<div class="hero">
  <div>
    <span class="chip primary">AI Agent System</span>
    <span class="chip success">Manual Tools (No-Fail)</span>
    <span class="chip warn">Groq Summarizer</span>
    <span class="chip primary">{status_badge}</span>
  </div>
  <h1>AI Finance Agent Team</h1>
  <p>Production-minded agentic workflow that combines live web/news + market data with Groq reasoning.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
<div class="card">
  <div class="card-title">System Snapshot</div>
  <div class="card-muted">Model</div>
  <div style="font-weight:700; margin-bottom:8px;">{model_id}</div>
  <div class="divider"></div>
  <div class="stat">
    <div class="stat-label">Tickers</div>
    <div class="stat-value">{tickers_count}</div>
  </div>
  <div style="height:8px"></div>
  <div class="stat">
    <div class="stat-label">News Items</div>
    <div class="stat-value">{news_count}</div>
  </div>
  <div style="height:8px"></div>
  <div class="stat">
    <div class="stat-label">Search Results</div>
    <div class="stat-value">{search_count}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )


def _render_how_it_works() -> None:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Agent pipeline</div>", unsafe_allow_html=True)
    st.markdown(
        """
<div class='card'>
  <div class='pipeline'>
    <div class='pipeline-step'>
      <div class='step'>01</div>
      <div>
        <div class='card-title'>Collect</div>
        <div class='card-muted'>Web/news + market data</div>
      </div>
    </div>
    <div class='pipeline-step'>
      <div class='step'>02</div>
      <div>
        <div class='card-title'>Structure</div>
        <div class='card-muted'>Ticker parsing + normalization</div>
      </div>
    </div>
    <div class='pipeline-step'>
      <div class='step'>03</div>
      <div>
        <div class='card-title'>Reason</div>
        <div class='card-muted'>Groq summary + tables</div>
      </div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_production_note() -> None:
    st.markdown(
        """
<div class='card'>
  <div class='card-title'>Reliability defaults</div>
  <ul class='clean-list'>
    <li>Deterministic tool execution (no LLM tool calls)</li>
    <li>Graceful fallbacks with transparent raw data</li>
    <li>Best accuracy with explicit tickers (e.g., AAPL, MSFT)</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="AI Finance Agent Team", layout="wide")
    _inject_css()

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

    last_tools = st.session_state.last_tools or {}
    news_count = _count_items(last_tools.get("web", {}).get("news_filtered") or last_tools.get("web", {}).get("news"))
    search_count = _count_items(last_tools.get("web", {}).get("search_filtered") or last_tools.get("web", {}).get("search"))
    tickers_count = len(st.session_state.last_tickers)

    api_ok = bool(os.getenv("GROQ_API_KEY"))

    with st.sidebar:
        st.subheader("Control Panel")
        model_id = st.text_input("Model", value=DEFAULT_MODEL_ID)
        st.caption("Override with the GROQ_MODEL environment variable.")
        if not api_ok:
            st.warning("GROQ_API_KEY is not set. The app will fail without it.")

        st.markdown("**Data sources**")
        include_finance = st.checkbox("Include finance data", value=True)
        include_web_news = st.checkbox("Include web news", value=True)
        include_web_search = st.checkbox("Include web search", value=True)

        st.markdown("**Result limits**")
        news_max_results = st.slider("News results", 1, 10, DEFAULT_NEWS_RESULTS)
        search_max_results = st.slider("Search results", 1, 10, DEFAULT_SEARCH_RESULTS)
        company_news_count = st.slider("Company news per ticker", 1, 10, DEFAULT_COMPANY_NEWS_STORIES)
        max_tickers = st.slider("Max tickers", 1, 12, DEFAULT_MAX_TICKERS)

        st.markdown("**Display**")
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

    _render_header(api_ok, model_id, tickers_count, news_count, search_count)
    _render_how_it_works()
    _render_production_note()

    quick_cols = st.columns(3)
    if quick_cols[0].button("Compare Apple vs Microsoft"):
        st.session_state.pending_prompt = "Compare Apple and Microsoft on valuation and recent news."
    if quick_cols[1].button("Summarize Tesla earnings"):
        st.session_state.pending_prompt = "Summarize Tesla's latest earnings and analyst sentiment."
    if quick_cols[2].button("Top AI stocks this week"):
        st.session_state.pending_prompt = "What's happening in AI stocks this week?"

    metrics = st.columns(3)
    metrics[0].metric("Tickers", tickers_count)
    metrics[1].metric("News items", news_count)
    metrics[2].metric("Search results", search_count)

    tabs = st.tabs(["Chat", "Insights", "About"])

    summarizer = get_summarizer(model_id)

    with tabs[0]:
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

    with tabs[1]:
        st.subheader("Latest run overview")
        if not st.session_state.last_tools:
            st.info("Run a query to see insights and raw data.")
        else:
            st.markdown("**Queries used**")
            st.code(json.dumps(st.session_state.last_tools.get("queries", {}), indent=2))
            st.markdown("**Detected tickers**")
            st.write(", ".join(st.session_state.last_tickers) or "None")
            if show_raw_data:
                st.markdown("**Raw tool data**")
                st.json(st.session_state.last_tools)

    with tabs[2]:
        st.subheader("Why this is an AI agent project")
        st.write(
            "This system is a production-minded AI agent pipeline. It collects live data, structures it, "
            "and uses a Groq LLM to synthesize decisions and comparisons. The model never calls tools directly, "
            "which eliminates tool-call failures and keeps responses consistent."
        )
        st.markdown(
            """
- Deterministic data collection
- Transparent sources and raw data visibility
- Reliable summarization with Groq
- Safe fallbacks on failures
"""
        )


if __name__ == "__main__":
    main()
