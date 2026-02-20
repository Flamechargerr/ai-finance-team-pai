from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

OUTPUT = Path("output/pdf/Finance_Agent_Synopsis_Anamay_Manas.pdf")


def build_pdf() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.65 * inch,
        title="Project Synopsis",
        author="Anamay Tripathy; Manas Gupta",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=10,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14.5,
        textColor=colors.HexColor("#111827"),
        spaceAfter=8,
    )
    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#111827"),
        spaceBefore=10,
        spaceAfter=8,
    )
    sub_style = ParagraphStyle(
        "Sub",
        parent=body_style,
        fontName="Helvetica-Bold",
        spaceAfter=4,
    )

    content = []
    content.append(Paragraph("Project Synopsis", title_style))
    content.append(
        Paragraph(
            "<b>Project Title:</b> AI Finance Agent Team: Retrieval-Augmented Multi-Agent Decision Support for Market Analysis "
            "<b>Domain:</b> Computational Finance / Artificial Intelligence (Natural Language Processing + Tool-Augmented Reasoning) "
            "<b>Type:</b> Applied Research and System Development",
            body_style,
        )
    )
    content.append(
        Paragraph(
            "<b>Prepared by:</b> Anamay Tripathy (230968270), Manas Gupta (230968318)",
            body_style,
        )
    )

    content.append(Paragraph("1. Problem Statement", heading_style))
    content.append(
        Paragraph(
            "In practical portfolio management, users can access abundant information but lack a reliable mechanism "
            "to convert unstructured narrative events into defensible investment decisions. Conventional quantitative "
            "models require structured numerical features and therefore underperform for open-ended text queries, "
            "whereas generic conversational systems often produce responses without verifiable market grounding. "
            "This creates a persistent gap between <b>natural-language user intent</b> and <b>evidence-backed financial reasoning</b>.",
            body_style,
        )
    )
    content.append(
        Paragraph(
            "The proposed system addresses this gap by implementing a retrieval-first, multi-agent workflow that "
            "(a) interprets user prompts, (b) acquires live macro and market signals, "
            "(c) consolidates evidence into structured context, and (d) generates concise analytical outputs "
            "within interactive latency constraints.",
            body_style,
        )
    )

    content.append(Paragraph("2. Study of Dataset", heading_style))
    content.append(
        Paragraph(
            "The platform combines live market telemetry and web intelligence instead of relying on a static local corpus:",
            body_style,
        )
    )
    ds_items = [
        "<b>Yahoo Finance (yfinance):</b> Live and historical price series, analyst signals, company fundamentals, and recent company-level news.",
        "<b>Web Search Layer (DuckDuckGo):</b> Broader geopolitical and macroeconomic context relevant to asset movement interpretation.",
        "<b>Prompt-Derived Signals:</b> Extracted symbols, entities, and intent tokens from user queries for targeted retrieval and comparison.",
    ]
    content.append(
        ListFlowable(
            [ListItem(Paragraph(item, body_style), leftIndent=12) for item in ds_items],
            bulletType="bullet",
            start="circle",
            leftIndent=14,
            bulletFontName="Helvetica",
            bulletFontSize=8,
        )
    )

    content.append(Paragraph("3. Exploratory Analysis and Visualization", heading_style))
    content.append(
        Paragraph(
            "Exploratory analysis is operationalized through an interactive Streamlit dashboard that exposes retrieval controls, "
            "ticker detection, and structured response blocks for rapid scenario evaluation.",
            body_style,
        )
    )
    viz_items = [
        "<b>Investment Compare Mode:</b> Controlled side-by-side evaluation of two tickers across valuation, momentum, risk, and catalyst dimensions.",
        "<b>Source Selection Controls:</b> Explicit toggles for web search, web news, and market tools to support sensitivity analysis.",
        "<b>Raw Evidence View:</b> Expandable retrieval payloads for transparent manual verification.",
        "<b>Structured Analytical Output:</b> Response format prioritizes precision, comparability, and decision readability.",
    ]
    content.append(
        ListFlowable(
            [ListItem(Paragraph(item, body_style), leftIndent=12) for item in viz_items],
            bulletType="bullet",
            start="circle",
            leftIndent=14,
            bulletFontName="Helvetica",
            bulletFontSize=8,
        )
    )

    content.append(Paragraph("4. PEAS Framework", heading_style))
    content.append(
        Paragraph("A structured PEAS model is applied to define the intelligent behavior of the agent team:", body_style)
    )
    content.append(Paragraph("Performance (P)", sub_style))
    content.append(
        Paragraph(
            "System performance is evaluated using factual consistency, source coverage, coherence of synthesis, and end-to-end latency. "
            "The operational objective is dependable analytical output under interactive response-time constraints.",
            body_style,
        )
    )
    content.append(Paragraph("Environment (E)", sub_style))
    content.append(
        Paragraph(
            "The operating environment is dynamic, stochastic, and partially observable. Market states and news narratives evolve continuously, "
            "and prompt scope may shift abruptly between company-specific and macro-level perspectives.",
            body_style,
        )
    )
    content.append(Paragraph("Actuators (A)", sub_style))
    act_items = [
        "Multi-agent orchestration combining web retrieval, finance tools, and LLM synthesis",
        "Interactive dashboard modules for query execution, comparison, and diagnostics",
        "Structured answer generation for portfolio-oriented decision support",
    ]
    content.append(
        ListFlowable(
            [ListItem(Paragraph(item, body_style), leftIndent=12) for item in act_items],
            bulletType="bullet",
            start="circle",
            leftIndent=14,
            bulletFontName="Helvetica",
            bulletFontSize=8,
        )
    )
    content.append(Paragraph("Sensors (S)", sub_style))
    sens_items = [
        "Natural-language query stream from end users",
        "Ticker/entity extraction and intent parsing",
        "Live market and web/news evidence retrieval channels",
    ]
    content.append(
        ListFlowable(
            [ListItem(Paragraph(item, body_style), leftIndent=12) for item in sens_items],
            bulletType="bullet",
            start="circle",
            leftIndent=14,
            bulletFontName="Helvetica",
            bulletFontSize=8,
        )
    )

    content.append(Paragraph("5. Preprocessing", heading_style))
    prep_items = [
        "<b>Query normalization:</b> canonicalization of user prompts to stabilize downstream retrieval behavior.",
        "<b>Ticker and entity extraction:</b> symbol detection with alias mapping for robust instrument coverage.",
        "<b>Query expansion:</b> context-aware reformulation to improve recall in search and news retrieval.",
        "<b>Evidence filtering:</b> relevance-based filtering of retrieved snippets before synthesis.",
        "<b>Context packaging:</b> deterministic assembly of compact evidence blocks for Groq reasoning.",
    ]
    content.append(
        Paragraph(
            "Preprocessing is intentionally retrieval-first and deterministic in order to reduce hallucination risk and improve reproducibility:",
            body_style,
        )
    )
    content.append(
        ListFlowable(
            [ListItem(Paragraph(item, body_style), leftIndent=12) for item in prep_items],
            bulletType="bullet",
            start="circle",
            leftIndent=14,
            bulletFontName="Helvetica",
            bulletFontSize=8,
        )
    )

    content.append(Paragraph("6. Project Objectives", heading_style))
    content.append(
        Paragraph(
            "<b>Primary Objective:</b> Deliver a robust finance intelligence assistant that transforms natural-language prompts into "
            "verifiable, market-grounded analysis via multi-agent orchestration.",
            body_style,
        )
    )
    obj_items = [
        "<b>Reliability:</b> preserve source-grounded reasoning with deterministic retrieval and transparent evidence flow.",
        "<b>Usability:</b> provide a concise dashboard workflow for exploratory and comparative financial analysis.",
        "<b>Extensibility:</b> maintain modular architecture to integrate additional tools, providers, and specialist agents.",
        "<b>Performance:</b> sustain low-latency execution suitable for iterative, analyst-in-the-loop decision making.",
    ]
    content.append(
        ListFlowable(
            [ListItem(Paragraph(item, body_style), leftIndent=12) for item in obj_items],
            bulletType="bullet",
            start="circle",
            leftIndent=14,
            bulletFontName="Helvetica",
            bulletFontSize=8,
        )
    )

    content.append(Spacer(1, 8))

    doc.build(content)
    return OUTPUT


if __name__ == "__main__":
    out = build_pdf()
    print(out)
