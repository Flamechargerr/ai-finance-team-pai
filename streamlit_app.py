import concurrent.futures
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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Global settings to override dark mode defaults with our custom vibe */
:root {
  --panel-bg: rgba(17, 25, 40, 0.75);
  --panel-border: rgba(255, 255, 255, 0.125);
  --ink: #f8fafc;
  --muted: #94a3b8;
  --accent: #3b82f6;
  --accent-hover: #2563eb;
  --accent-2: #10b981;
  --accent-3: #f59e0b;
  --glow-primary: rgba(59, 130, 246, 0.5);
}

/* Force dark mode appearance for Streamlit app background and generic elements */
.stApp {
    background-color: #0b0f19;
    color: var(--ink);
    font-family: 'Outfit', sans-serif !important;
}

/* Hero Section */
.hero {
  font-family: "Outfit", sans-serif;
  background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--panel-border);
  padding: 32px;
  border-radius: 20px;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  position: relative;
  overflow: hidden;
}

.hero::before {
  content: "";
  position: absolute;
  top: -50%; left: -50%; width: 200%; height: 200%;
  background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, rgba(0,0,0,0) 70%);
  z-index: 0;
  pointer-events: none;
}

.hero > div, .hero h1, .hero p {
  position: relative;
  z-index: 1;
}

.hero:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
}

.hero h1 { 
  margin: 12px 0 8px; 
  font-size: 36px; 
  font-weight: 700;
  background: linear-gradient(to right, #60a5fa, #a78bfa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hero p { margin: 0; color: var(--muted); font-size: 16px; font-weight: 300; }

/* Chips */
.chip { display: inline-block; padding: 6px 14px; border-radius: 999px; font-size: 12px; font-weight: 600; margin-right: 8px; margin-bottom: 8px; color: #fff; letter-spacing: 0.5px; text-transform: uppercase;}
.chip.primary { background: linear-gradient(135deg, var(--accent), #1d4ed8); box-shadow: 0 4px 10px var(--glow-primary); }
.chip.success { background: linear-gradient(135deg, var(--accent-2), #059669); }
.chip.warn { background: linear-gradient(135deg, var(--accent-3), #d97706); }

/* Cards */
.card { 
  font-family: "Outfit", sans-serif; 
  background: var(--panel-bg); 
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--panel-border); 
  border-radius: 16px; 
  padding: 20px; 
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
  transition: all 0.3s ease;
}
.card:hover { border-color: rgba(255,255,255,0.25); box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4); transform: translateY(-2px); }
.card-title { font-weight: 600; font-size: 18px; margin-bottom: 8px; color: var(--ink); letter-spacing: 0.5px;}
.card-muted { color: var(--muted); font-size: 14px; font-weight: 300;}

/* Layout Utilities */
.section-title { font-family: "Outfit", sans-serif; font-weight: 600; font-size: 22px; margin-bottom: 12px; color: var(--ink); }
.divider { height: 1px; background: linear-gradient(90deg, transparent, var(--panel-border), transparent); margin: 24px 0; }
.note { font-family: "Outfit", sans-serif; background: rgba(59, 130, 246, 0.1); border: 1px dashed rgba(59, 130, 246, 0.4); padding: 16px; border-radius: 12px; color: #bfdbfe; font-size: 14px;}

/* Stats */
.stat { 
  background: rgba(15, 23, 42, 0.5); 
  border: 1px solid var(--panel-border); 
  border-radius: 12px; 
  padding: 16px; 
  transition: all 0.2s ease;
}
.stat:hover { background: rgba(30, 41, 59, 0.8); border-color: rgba(96, 165, 250, 0.5); transform: scale(1.02); }
.stat-label { font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 500;}
.stat-value { font-size: 28px; font-weight: 700; color: #fff; text-shadow: 0 0 10px rgba(255,255,255,0.2); margin-top: 4px;}

/* Pipeline Steps */
.pipeline { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }
.pipeline-step { 
  background: rgba(15, 23, 42, 0.6); 
  border: 1px solid var(--panel-border); 
  border-radius: 16px; 
  padding: 16px; 
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}
.pipeline-step:hover {
  background: rgba(30, 41, 59, 0.9);
  border-color: rgba(96, 165, 250, 0.6);
  transform: translateY(-4px);
  box-shadow: 0 10px 25px rgba(0,0,0,0.5);
}
.pipeline-step::after {
  content: ""; position: absolute; top: 0; right: 0; width: 50px; height: 50px;
  background: radial-gradient(circle top right, rgba(59,130,246,0.2) 0%, transparent 70%);
}
.step { font-weight: 700; font-size: 12px; color: #0f172a; background: #60a5fa; border-radius: 999px; padding: 6px 12px; display: inline-block; box-shadow: 0 0 10px rgba(96, 165, 250, 0.5);}
.step-title { font-weight: 600; color: #fff; margin-top: 12px; font-size: 16px;}
.step-desc { color: var(--muted); font-size: 13px; margin-top: 4px; font-weight: 300;}

/* Reliabilities */
.reliability { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
.pill { background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); color: #93c5fd; font-size: 12px; font-weight: 500; padding: 6px 14px; border-radius: 999px; letter-spacing: 0.5px;}

/* Custom Streamlit Element Overrides */
/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-family: 'Outfit', sans-serif !important;
    transition: all 0.3s ease !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.4) !important;
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    color: white !important;
}
/* Inputs */
.stTextInput input, .stChatInputContainer {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid var(--panel-border) !important;
    color: white !important;
    border-radius: 8px !important;
}
.stTextInput input:focus, .stChatInput textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
}
/* Metrics */
[data-testid="stMetricValue"] {
    font-family: 'Outfit', sans-serif;
    color: white;
}
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


def _sanitize_ticker(value: str) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9.-]", "", value.strip())
    return cleaned.upper()


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

    def fetch_web_news():
        try:
            news_items = json.loads(ddg.duckduckgo_news(query=queries["news"], max_results=news_max_results))
            data["web"]["news"] = news_items
            data["web"]["news_filtered"] = _filter_items(news_items, keyword_set)
        except Exception as exc:
            data["web"]["news_error"] = f"{exc}"

    def fetch_web_search():
        try:
            search_items = json.loads(ddg.duckduckgo_search(query=queries["search"], max_results=search_max_results))
            data["web"]["search"] = search_items
            data["web"]["search_filtered"] = _filter_items(search_items, keyword_set)
        except Exception as exc:
            data["web"]["search_error"] = f"{exc}"

    def fetch_finance_for_ticker(ticker: str):
        yf_tools = YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True,
        )
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
        return ticker, ticker_data

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        if include_web_news:
            futures.append(executor.submit(fetch_web_news))
        if include_web_search:
            futures.append(executor.submit(fetch_web_search))
        
        finance_futures = []
        if include_finance:
            for ticker in tickers:
                finance_futures.append(executor.submit(fetch_finance_for_ticker, ticker))
        
        concurrent.futures.wait(futures)
        
        for future in concurrent.futures.as_completed(finance_futures):
            ticker, ticker_data = future.result()
            data["finance"][ticker] = ticker_data

    data["queries"] = queries
    return data


def _get_investment_tool_outputs(
    ticker_a: str,
    ticker_b: str,
    focus: str,
    include_web_news: bool,
    include_web_search: bool,
    include_finance: bool,
    news_max_results: int,
    search_max_results: int,
    company_news_count: int,
) -> Dict[str, object]:
    base_prompt = (
        f"Compare {ticker_a} and {ticker_b} on valuation, financial strength, "
        "growth, analyst sentiment, and recent news."
    )
    if focus:
        base_prompt = f"{base_prompt} Focus: {focus}."
    tool_data = _get_manual_tool_outputs(
        prompt=base_prompt,
        tickers=[ticker_a, ticker_b],
        include_web_news=include_web_news,
        include_web_search=include_web_search,
        include_finance=include_finance,
        news_max_results=news_max_results,
        search_max_results=search_max_results,
        company_news_count=company_news_count,
    )
    tool_data["comparison_prompt"] = base_prompt
    if focus:
        tool_data["focus"] = focus
    return tool_data


def _build_summary_prompt(prompt: str, tickers: List[str], tool_data: Dict[str, object]) -> str:
    tickers_line = ", ".join(tickers) if tickers else "None detected"
    return (
        "You are a top-tier, production-grade financial analyst AI. Your task is to provide a highly structured, "
        "insightful, and data-driven response based on the provided live market data and news.\n\n"
        "Guidelines:\n"
        "1. Start with a brief <thought> block where you review the raw data and plan your response.\n"
        "2. Provide a clear, executive-style summary answering the user's prompt directly.\n"
        "3. Use professional markdown formatting including tables, bullet points, and bold text for key metrics.\n"
        "4. Always cite news items with their exact full URLs if available.\n"
        "5. Do not hallucinate financial numbers. If data is missing or unrelated, explicitly state it is best-effort.\n"
        "6. Do not insert extra spaces between letters.\n\n"
        f"User prompt:\n{prompt}\n\n"
        f"Detected tickers: {tickers_line}\n\n"
        f"Tool data (JSON):\n{json.dumps(tool_data, indent=2)}"
    )


def _build_investment_summary_prompt(
    ticker_a: str, ticker_b: str, focus: str, tool_data: Dict[str, object]
) -> str:
    focus_line = f"Focus area: {focus}" if focus else "Focus area: valuation, growth, risk, catalysts."
    return (
        "You are an elite investment analysis AI. Compare the two requested tickers using the provided live data.\n\n"
        "Guidelines:\n"
        "1. Start with a brief <thought> block to analyze the comparative data.\n"
        "2. Provide a side-by-side comparison table for key metrics (price, market cap, P/E, EPS, analyst consensus) if available.\n"
        "3. Summarize recent news with links and list key catalysts and risks.\n"
        "4. Do not provide personalized financial advice.\n"
        "5. Do not hallucinate numbers. If data is missing, explicitly say so.\n"
        "6. Do not insert extra spaces between letters.\n\n"
        f"Tickers: {ticker_a}, {ticker_b}\n"
        f"{focus_line}\n\n"
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
      <div class='step-title'>Collect</div>
      <div class='step-desc'>DuckDuckGo + Yahoo Finance</div>
    </div>
    <div class='pipeline-step'>
      <div class='step'>02</div>
      <div class='step-title'>Structure</div>
      <div class='step-desc'>Ticker parsing and data normalization</div>
    </div>
    <div class='pipeline-step'>
      <div class='step'>03</div>
      <div class='step-title'>Reason</div>
      <div class='step-desc'>Groq summary with comparisons</div>
    </div>
  </div>
  <div class='divider'></div>
  <div class='card-muted'>Reliability defaults</div>
  <div class='reliability'>
    <span class='pill'>Deterministic tools</span>
    <span class='pill'>Graceful fallbacks</span>
    <span class='pill'>Best with explicit tickers</span>
  </div>
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
    if "investment_last" not in st.session_state:
        st.session_state.investment_last = None

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

    tabs = st.tabs(["Chat", "Investment Compare", "Insights", "About"])

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
        st.subheader("Investment comparison")
        st.write(
            "Compare two tickers using live market data plus recent news. "
            "This mirrors the single-agent investment flow while keeping tool execution deterministic."
        )

        with st.form("investment_compare"):
            col_a, col_b = st.columns(2)
            ticker_a = col_a.text_input("Ticker A", value="AAPL", key="inv_ticker_a")
            ticker_b = col_b.text_input("Ticker B", value="MSFT", key="inv_ticker_b")
            focus = st.text_input(
                "Focus (optional)",
                value="",
                key="inv_focus",
                placeholder="e.g., valuation, growth, risk, catalysts",
            )
            submitted = st.form_submit_button("Run investment comparison")

        if submitted:
            clean_a = _sanitize_ticker(ticker_a)
            clean_b = _sanitize_ticker(ticker_b)
            if not clean_a or not clean_b:
                st.warning("Add two tickers (e.g., AAPL, MSFT) to run a comparison.")
            else:
                with st.spinner("Gathering data..."):
                    tool_data = _get_investment_tool_outputs(
                        ticker_a=clean_a,
                        ticker_b=clean_b,
                        focus=focus,
                        include_web_news=include_web_news,
                        include_web_search=include_web_search,
                        include_finance=include_finance,
                        news_max_results=news_max_results,
                        search_max_results=search_max_results,
                        company_news_count=company_news_count,
                    )
                    summary_prompt = _build_investment_summary_prompt(clean_a, clean_b, focus, tool_data)

                    try:
                        response = summarizer.run(summary_prompt, stream=False)
                        summary = _normalize_spaced_text(response.get_content_as_string())
                    except Exception as exc:
                        summary = (
                            "I couldn't generate an investment summary, but the raw data is available below. "
                            f"Error: {exc}"
                        )

                    st.session_state.investment_last = {
                        "tickers": [clean_a, clean_b],
                        "focus": focus,
                        "summary": summary,
                        "tools": tool_data,
                        "run_at": datetime.now().strftime("%b %d, %Y %I:%M %p"),
                    }
                    st.session_state.last_tools = tool_data
                    st.session_state.last_tickers = [clean_a, clean_b]
                    st.session_state.last_run_at = st.session_state.investment_last["run_at"]

        if st.session_state.get("investment_last"):
            last = st.session_state.investment_last
            st.markdown("### Latest comparison")
            st.caption(f"Last run: {last['run_at']}")
            st.markdown(last["summary"])
            if show_raw_data:
                with st.expander("Raw data"):
                    st.json(last["tools"])

    with tabs[2]:
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

    with tabs[3]:
        st.subheader("Why this is an AI agent project")
        st.write(
            "This system is an agentic pipeline: it collects live data, structures it into a clean context, "
            "and uses Groq to synthesize insights. The model never calls tools directly, which makes outputs "
            "deterministic and production-friendly."
        )
        st.markdown("**Production-ready posture**")
        st.markdown(
            """
- Deterministic tool execution (manual data collection)
- Transparent sources with optional raw data visibility
- Reliable summarization with Groq and safe fallbacks
- Separate chat and investment-compare modes for focused workflows
"""
        )


if __name__ == "__main__":
    main()
