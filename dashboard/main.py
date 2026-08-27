import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.ingestion.ticker_search import CompanyDirectorySearch
from app.ingestion.sec_fetcher import SECFilingFetcher
from app.rag.qa_engine import FinancialQAService
from app.analytics.technical import TechnicalAnalysisEngine
from app.core.database import SyncSessionLocal
from app.models.domain import Company, Document, FinancialMetric
from sqlalchemy import select, and_

# Configure Page Layout & Title
st.set_page_config(
    page_title="Financial RAG & Equity Intelligence Platform",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional Dark Theme CSS
st.markdown("""
<style>
    html {
        scroll-behavior: smooth;
    }
    .metric-box {
        background: linear-gradient(135deg, #1e2235 0%, #151824 100%);
        border: 1px solid #2e344e;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        color: #8b95a5;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 24px;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-subtitle {
        color: #00e676;
        font-size: 12px;
        font-weight: 500;
        margin-top: 2px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        border-bottom: 1px solid #2e344e;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        font-size: 15px;
        font-weight: 600;
    }
</style>
<div id="top-anchor"></div>
""", unsafe_allow_html=True)

def run_async_safe(coro):
    """Executes asynchronous coroutines cleanly in an isolated event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

def format_currency_smart(val_in_millions):
    """Formats millions into readable $B or $M without truncation."""
    if val_in_millions is None or pd.isna(val_in_millions):
        return "N/A"
    val = float(val_in_millions)
    if abs(val) >= 1000:
        return f"${val / 1000:,.2f}B"
    else:
        return f"${val:,.0f}M"

def get_all_companies_annual_data():
    """Fetches ONLY annual 10-K filings across companies (strictly excluding 10-Qs)."""
    with SyncSessionLocal() as session:
        stmt = select(Company, Document, FinancialMetric).\
            join(Document, Company.id == Document.company_id).\
            join(FinancialMetric, Document.id == FinancialMetric.document_id).\
            where(Document.form_type == "10-K").\
            order_by(Company.ticker, Document.fiscal_year.desc())
        res = session.execute(stmt).all()
        return res

def get_company_annual_history(ticker):
    """Fetches clean chronological annual 10-K records and company name together."""
    with SyncSessionLocal() as session:
        stmt = select(Company, Document, FinancialMetric).\
            join(Document, Company.id == Document.company_id).\
            join(FinancialMetric, Document.id == FinancialMetric.document_id).\
            where(and_(Document.ticker == ticker, Document.form_type == "10-K")).\
            order_by(Document.fiscal_year.asc())
        res = session.execute(stmt).all()
        
        year_dict = {}
        company_name = f"{ticker} Inc."
        for comp, doc, met in res:
            company_name = comp.company_name
            year_dict[doc.fiscal_year] = (doc, met)
            
        sorted_years = sorted(year_dict.keys())
        filings_list = [year_dict[y] for y in sorted_years]
        return company_name, filings_list

all_filings_data = get_all_companies_annual_data()
ticker_set = sorted(list(set([comp.ticker for comp, doc, met in all_filings_data]))) if all_filings_data else []

# ----------------- SIDEBAR: NAVIGATION & INGESTION -----------------
with st.sidebar:
    st.title("🏦 Financial RAG")
    st.caption("AI-Powered SEC 10-K Equity Research Platform")
    st.markdown("---")

    view_mode = st.radio("Navigation View", ["📊 Portfolio Benchmark (Landing Page)", "🔍 Company Deep-Dive (Intelligence Studio)"])

    st.markdown("---")
    st.subheader("⚡ Ingest SEC 10-K Company")
    st.caption("Search across 10,000+ US public equities")
    
    search_query = st.text_input("🔍 Company Name or Ticker", value="", placeholder="Type e.g. Apple, Microsoft, Tesla...").strip()
    
    target_ticker = None
    if search_query:
        search_results = CompanyDirectorySearch.search(search_query, limit=12)
        if search_results:
            options_labels = [f"{item['ticker']} — {item['name']}" for item in search_results]
            selected_option = st.selectbox("Matching SEC Filers:", options=options_labels, index=0)
            target_ticker = selected_option.split(" — ")[0].strip()
        else:
            st.info(f"Direct ticker ingestion: '{search_query.upper()}'")
            target_ticker = search_query.upper()

    if target_ticker:
        sec_fetcher = SECFilingFetcher()
        available_years = sec_fetcher.get_available_years(target_ticker)
        min_yr = min(available_years)
        max_yr = max(available_years)
        
        st.markdown(f"**📅 Available 10-K Filings ({min_yr} – {max_yr}):**")
        
        selected_years = st.multiselect(
            "Select Fiscal Years to Ingest:",
            options=available_years,
            default=available_years[:5],
            help="Select one, several, or all 10 years to ingest."
        )

        if st.button(f"🚀 Ingest Selected 10-Ks ({len(selected_years)} Years)", use_container_width=True):
            if selected_years:
                with st.spinner(f"Pulling {len(selected_years)} annual 10-K filings for {target_ticker}..."):
                    if hasattr(sec_fetcher, "fetch_and_ingest_years_sync"):
                        results = sec_fetcher.fetch_and_ingest_years_sync(target_ticker, selected_years)
                    else:
                        results = run_async_safe(sec_fetcher.fetch_and_ingest_years(target_ticker, selected_years))
                    st.success(f"✅ Ingested {len(results)} annual filing(s) for {target_ticker} ({min(selected_years)}–{max(selected_years)})!")
                    st.rerun()
            else:
                st.warning("Please select at least one fiscal year.")

    st.markdown("---")
    st.caption("Active Database: Neon Serverless PostgreSQL (pgvector)")
    st.caption("LLM Engine: Google Gemini 3.6 Flash")


# =========================================================================
# VIEW 1: PORTFOLIO BENCHMARK & COMPARISON (LANDING PAGE)
# =========================================================================
if view_mode == "📊 Portfolio Benchmark (Landing Page)":
    st.title("🏛️ Multi-Company Equity Benchmark Matrix")
    st.caption("Side-by-side annual 10-K performance comparison across ingested public companies.")

    if not all_filings_data:
        st.info("No companies ingested yet. Use the search box in the sidebar to ingest your first company.")
    else:
        companies_name_map = {}
        for comp, doc, met in all_filings_data:
            companies_name_map[comp.ticker] = comp.company_name

        all_unique_tickers = sorted(list(companies_name_map.keys()))
        selected_companies = st.multiselect(
            "🏢 Select Companies to Benchmark:",
            options=all_unique_tickers,
            default=all_unique_tickers,
            format_func=lambda t: f"{t} ({companies_name_map.get(t, t)})",
            help="Select companies to compare. All annual 10-Ks for chosen companies are loaded."
        )

        all_ingested_years = sorted(
            list(set([doc.fiscal_year for comp, doc, met in all_filings_data if comp.ticker in selected_companies])),
            reverse=True
        )

        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.6, 1.2, 1.2])
        with ctrl_col1:
            time_preset = st.selectbox(
                "⏱️ Time Horizon Preset:",
                [
                    "All Ingested Years (Complete History)",
                    "Last 5 Years (2021–2025)",
                    "Last 3 Years (2023–2025)",
                    "Latest Year Only (FY2025)",
                    "Custom Year Checkbox Selection"
                ],
                index=0
            )

        if time_preset == "Latest Year Only (FY2025)":
            active_years = all_ingested_years[:1]
        elif time_preset == "Last 3 Years (2023–2025)":
            active_years = [y for y in all_ingested_years if y >= 2023] or all_ingested_years[:3]
        elif time_preset == "Last 5 Years (2021–2025)":
            active_years = [y for y in all_ingested_years if y >= 2021] or all_ingested_years[:5]
        elif time_preset == "Custom Year Checkbox Selection":
            active_years = st.multiselect(
                "📅 Check/Uncheck Specific Fiscal Years:",
                options=all_ingested_years,
                default=all_ingested_years
            )
        else:
            active_years = all_ingested_years

        with ctrl_col2:
            top_n_filter = st.selectbox("🔝 Ranking Filter", ["All Matching", "Top 5 by Revenue", "Top 10 by Revenue", "Top 10 by Op. Margin"], index=0)
        with ctrl_col3:
            chart_view_type = st.selectbox("📊 Chart Style", ["Horizontal Bars (Scales to 50+)", "2D Scatter Matrix (Bubble Chart)"])

        records = []
        for comp, doc, met in all_filings_data:
            if comp.ticker in selected_companies and doc.fiscal_year in active_years:
                fy_str = f"FY{doc.fiscal_year}"
                label_with_year = f"{comp.ticker} ({fy_str})"
                
                rev_m = met.revenue or 0
                gp_m = met.gross_profit or 0
                op_m = met.operating_income or 0
                net_m = met.net_income or 0
                fcf_m = met.free_cash_flow or 0
                debt_m = met.total_debt or 0
                cash_m = met.total_cash_and_equivalents or 0

                records.append({
                    "Ticker": comp.ticker,
                    "Company": comp.company_name,
                    "Label": label_with_year,
                    "Fiscal Year": fy_str,
                    "Year": doc.fiscal_year,
                    "Revenue": format_currency_smart(rev_m),
                    "Revenue ($M)": rev_m,
                    "Gross Margin (%)": f"{met.gross_margin:.1f}%" if met.gross_margin else "N/A",
                    "Gross Margin Raw": met.gross_margin or 0.0,
                    "Op. Margin (%)": f"{met.operating_margin:.1f}%" if met.operating_margin else "N/A",
                    "Op. Margin Raw": met.operating_margin or 0.0,
                    "Net Income": format_currency_smart(net_m),
                    "Net Income ($M)": net_m,
                    "Net Margin (%)": f"{met.net_profit_margin:.1f}%" if met.net_profit_margin else "N/A",
                    "Net Margin Raw": met.net_profit_margin or 0.0,
                    "Diluted EPS": f"${met.diluted_eps:.2f}" if met.diluted_eps else "N/A",
                    "Free Cash Flow": format_currency_smart(fcf_m),
                    "Free Cash Flow ($M)": fcf_m,
                    "Debt/Equity": f"{met.debt_to_equity:.2f}x" if met.debt_to_equity else "N/A",
                    "Debt-to-Equity (x)": met.debt_to_equity or 0.0,
                    "Total Debt": format_currency_smart(debt_m),
                    "Total Debt ($M)": debt_m,
                    "Cash & Equivalents": format_currency_smart(cash_m)
                })

        df_filtered = pd.DataFrame(records)

        if df_filtered.empty:
            st.warning("No records match the current company and year selection.")
        else:
            if top_n_filter == "Top 5 by Revenue":
                df_filtered = df_filtered.sort_values(by="Revenue ($M)", ascending=False).head(5)
            elif top_n_filter == "Top 10 by Revenue":
                df_filtered = df_filtered.sort_values(by="Revenue ($M)", ascending=False).head(10)
            elif top_n_filter == "Top 10 by Op. Margin":
                df_filtered = df_filtered.sort_values(by="Op. Margin Raw", ascending=False).head(10)

            st.subheader(f"📌 Peer Benchmark Matrix ({len(selected_companies)} Companies, {len(active_years)} Fiscal Years)")
            st.dataframe(
                df_filtered[[
                    "Label", "Company", "Fiscal Year", "Revenue", "Gross Margin (%)",
                    "Op. Margin (%)", "Net Income", "Net Margin (%)",
                    "Diluted EPS", "Free Cash Flow", "Debt/Equity", "Total Debt", "Cash & Equivalents"
                ]],
                use_container_width=True
            )

            st.markdown("---")
            st.subheader("📊 Comparative Financial Widgets")

            if chart_view_type == "2D Scatter Matrix (Bubble Chart)":
                st.caption("💡 X-Axis: Revenue Scale ($M) | Y-Axis: Operating Margin (%) | Bubble Size: Free Cash Flow ($M) | Color: Net Profit Margin (%)")
                fig_scatter = px.scatter(
                    df_filtered,
                    x="Revenue ($M)",
                    y="Op. Margin Raw",
                    size=df_filtered["Free Cash Flow ($M)"].apply(lambda v: max(float(v), 50.0)),
                    color="Net Margin Raw",
                    hover_name="Label",
                    text="Label",
                    color_continuous_scale="Viridis",
                    template="plotly_dark",
                    height=550
                )
                fig_scatter.update_traces(textposition="top center")
                st.plotly_chart(fig_scatter, use_container_width=True)

            else:
                dynamic_height = max(380, len(df_filtered) * 42)

                col_w1, col_w2 = st.columns(2)
                with col_w1:
                    fig_h_rev = px.bar(
                        df_filtered.sort_values(by="Revenue ($M)", ascending=True),
                        y="Label",
                        x=["Revenue ($M)", "Net Income ($M)"],
                        barmode="group",
                        orientation="h",
                        title="Annual Revenue vs. Net Income ($ Millions)",
                        color_discrete_sequence=["#4a86e8", "#00c853"],
                        template="plotly_dark",
                        height=dynamic_height
                    )
                    st.plotly_chart(fig_h_rev, use_container_width=True)

                with col_w2:
                    fig_h_mar = px.bar(
                        df_filtered.sort_values(by="Op. Margin Raw", ascending=True),
                        y="Label",
                        x=["Gross Margin Raw", "Op. Margin Raw", "Net Margin Raw"],
                        barmode="group",
                        orientation="h",
                        title="Margin Durability (% Breakdown)",
                        color_discrete_sequence=["#ff9800", "#9c27b0", "#00e676"],
                        template="plotly_dark",
                        height=dynamic_height
                    )
                    st.plotly_chart(fig_h_mar, use_container_width=True)

                col_w3, col_w4 = st.columns(2)
                with col_w3:
                    fig_h_fcf = px.bar(
                        df_filtered.sort_values(by="Free Cash Flow ($M)", ascending=True),
                        y="Label",
                        x=["Free Cash Flow ($M)", "Total Debt ($M)"],
                        barmode="group",
                        orientation="h",
                        title="Free Cash Flow vs. Total Debt ($ Millions)",
                        color_discrete_sequence=["#00b0ff", "#ff5252"],
                        template="plotly_dark",
                        height=dynamic_height
                    )
                    st.plotly_chart(fig_h_fcf, use_container_width=True)

                with col_w4:
                    fig_h_de = px.bar(
                        df_filtered.sort_values(by="Debt-to-Equity (x)", ascending=True),
                        y="Label",
                        x="Debt-to-Equity (x)",
                        orientation="h",
                        title="Debt-to-Equity Leverage (x)",
                        color="Debt-to-Equity (x)",
                        color_continuous_scale="Viridis",
                        template="plotly_dark",
                        height=dynamic_height
                    )
                    st.plotly_chart(fig_h_de, use_container_width=True)

        st.markdown("---")

        # Global Portfolio Copilot
        st.subheader("💬 Portfolio Financial Copilot (Cross-Company Q&A)")
        st.caption("Ask comparative questions across all ingested annual 10-K reports.")

        if "portfolio_messages" not in st.session_state:
            st.session_state.portfolio_messages = []

        port_chat_container = st.container(height=360)
        with port_chat_container:
            for p_msg in st.session_state.portfolio_messages:
                with st.chat_message(p_msg["role"]):
                    st.markdown(p_msg["content"])
                    if p_msg.get("citations"):
                        with st.expander("📑 Source SEC Citations"):
                            for cit in p_msg["citations"]:
                                st.markdown(f"**{cit['ticker']} — Page {cit['page_number']} [{cit['section']}]** *(Score: {cit['score']})*")
                                st.caption(cit["content_snippet"])

        # Dedicated Form Input to avoid auto-scrolling
        with st.form(key="port_chat_form", clear_on_submit=True):
            p_cols = st.columns([5, 1])
            with p_cols[0]:
                portfolio_prompt = st.text_input("Ask a cross-company question...", label_visibility="collapsed", placeholder="Type a comparative question across all companies...")
            with p_cols[1]:
                submit_port = st.form_submit_button("Send 💬", use_container_width=True)

            if submit_port and portfolio_prompt:
                st.session_state.portfolio_messages.append({"role": "user", "content": portfolio_prompt})
                with port_chat_container:
                    with st.chat_message("user"):
                        st.markdown(portfolio_prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing annual 10-K SEC filings..."):
                            qa = FinancialQAService()
                            res = run_async_safe(qa.answer_question(question=portfolio_prompt, ticker="ALL"))
                            st.markdown(res["answer"])
                            if res.get("citations"):
                                with st.expander("📑 Source SEC Citations"):
                                    for cit in res["citations"]:
                                        st.markdown(f"**{cit['ticker']} — Page {cit['page_number']} [{cit['section']}]** *(Score: {cit['score']})*")
                                        st.caption(cit["content_snippet"])

                            st.session_state.portfolio_messages.append({
                                "role": "assistant",
                                "content": res["answer"],
                                "citations": res.get("citations", [])
                            })
                st.rerun()

# =========================================================================
# VIEW 2: COMPANY DEEP-DIVE (PREMIUM INTELLIGENCE STUDIO)
# =========================================================================
else:
    selected_ticker = st.selectbox("📌 Select Target Company", options=ticker_set, index=0)
    company_name_resolved, company_filings = get_company_annual_history(selected_ticker)
    
    timeline_records = []
    latest_doc = None
    latest_metric = None

    for doc_item, met_item in company_filings:
        fy_str = f"FY{doc_item.fiscal_year}"
        
        rev_m = met_item.revenue or 0
        gp_m = met_item.gross_profit or 0
        op_m = met_item.operating_income or 0
        net_m = met_item.net_income or 0
        fcf_m = met_item.free_cash_flow or 0
        debt_m = met_item.total_debt or 0
        cash_m = met_item.total_cash_and_equivalents or 0

        timeline_records.append({
            "Fiscal Year": fy_str,
            "Year": doc_item.fiscal_year,
            "Revenue": format_currency_smart(rev_m),
            "Revenue ($M)": rev_m,
            "Gross Margin (%)": f"{met_item.gross_margin:.1f}%" if met_item.gross_margin else "N/A",
            "Gross Margin Raw": met_item.gross_margin or 0.0,
            "Operating Margin (%)": f"{met_item.operating_margin:.1f}%" if met_item.operating_margin else "N/A",
            "Op. Margin Raw": met_item.operating_margin or 0.0,
            "Net Income": format_currency_smart(net_m),
            "Net Income ($M)": net_m,
            "Net Margin (%)": f"{met_item.net_profit_margin:.1f}%" if met_item.net_profit_margin else "N/A",
            "Diluted EPS": f"${met_item.diluted_eps:.2f}" if met_item.diluted_eps else "N/A",
            "Free Cash Flow": format_currency_smart(fcf_m),
            "Free Cash Flow ($M)": fcf_m,
            "Total Debt": format_currency_smart(debt_m),
            "Total Debt ($M)": debt_m,
            "Cash & Equivalents": format_currency_smart(cash_m),
            "Debt/Equity": f"{met_item.debt_to_equity:.2f}x" if met_item.debt_to_equity else "N/A"
        })
        latest_doc = doc_item
        latest_metric = met_item

    df_multiyear = pd.DataFrame(timeline_records)

    # 1. Sleek Header Banner
    year_range_str = f"{df_multiyear['Fiscal Year'].iloc[0]} – {df_multiyear['Fiscal Year'].iloc[-1]}" if not df_multiyear.empty else "FY2025"

    st.markdown(f"## 💎 {selected_ticker} — {company_name_resolved}")
    st.caption(f"Comprehensive SEC Form 10-K Equity Research & Analytics ({year_range_str})")

    # 2. Executive Fundamental Scorecard (6 Cards with Clear Formatted Values)
    if latest_metric:
        st.markdown(f"#### 📊 Latest Fiscal Position (FY{latest_doc.fiscal_year})")
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
        with kpi1:
            st.markdown(f"""<div class="metric-box"><div class="metric-title">Annual Revenue</div><div class="metric-value">{format_currency_smart(latest_metric.revenue)}</div><div class="metric-subtitle">Top-Line Scale</div></div>""", unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""<div class="metric-box"><div class="metric-title">Gross Margin</div><div class="metric-value">{latest_metric.gross_margin:.1f}%</div><div class="metric-subtitle">Pricing Power</div></div>""", unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""<div class="metric-box"><div class="metric-title">Operating Margin</div><div class="metric-value">{latest_metric.operating_margin:.1f}%</div><div class="metric-subtitle">EBIT Efficiency</div></div>""", unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""<div class="metric-box"><div class="metric-title">Net Income</div><div class="metric-value">{format_currency_smart(latest_metric.net_income)}</div><div class="metric-subtitle">Bottom-Line Profit</div></div>""", unsafe_allow_html=True)
        with kpi5:
            st.markdown(f"""<div class="metric-box"><div class="metric-title">Diluted EPS</div><div class="metric-value">${latest_metric.diluted_eps:.2f}</div><div class="metric-subtitle">Per Share Earning</div></div>""", unsafe_allow_html=True)
        with kpi6:
            st.markdown(f"""<div class="metric-box"><div class="metric-title">Free Cash Flow</div><div class="metric-value">{format_currency_smart(latest_metric.free_cash_flow)}</div><div class="metric-subtitle">Cash Conversion</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 3. Clean Multi-Year Annual Statement Table
    if not df_multiyear.empty:
        st.subheader("📅 Multi-Year Annual Financial Model Trajectory")
        st.dataframe(
            df_multiyear[[
                "Fiscal Year", "Revenue", "Gross Margin (%)", "Operating Margin (%)",
                "Net Income", "Net Margin (%)", "Diluted EPS", "Free Cash Flow",
                "Total Debt", "Cash & Equivalents", "Debt/Equity"
            ]],
            use_container_width=True
        )

    # 4. AI Strategic Overview Drawer
    if latest_doc and latest_doc.executive_summary:
        with st.expander(f"🤖 View AI Strategic Synthesis & Risk Factors (FY{latest_doc.fiscal_year})", expanded=False):
            st.write(latest_doc.executive_summary)
            col_r, col_c = st.columns(2)
            with col_r:
                st.markdown("##### ⚠️ Core Risk Factors (Item 1A)")
                try:
                    risks = json.loads(latest_doc.key_risks) if isinstance(latest_doc.key_risks, str) else latest_doc.key_risks
                    for r in risks:
                        st.markdown(f"* {r}")
                except Exception:
                    st.write(latest_doc.key_risks)
            with col_c:
                st.markdown("##### 🚀 Growth Catalysts & Strategic Drivers")
                try:
                    cats = json.loads(latest_doc.growth_catalysts) if isinstance(latest_doc.growth_catalysts, str) else doc.growth_catalysts
                    for c in cats:
                        st.markdown(f"* {c}")
                except Exception:
                    st.write(doc.growth_catalysts)

    st.markdown("---")

    # 5. SIDE-BY-SIDE SPLIT VIEW: Interactive Studio (Left) vs. Annual Copilot (Right)
    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.subheader("📈 Technical & Fundamental Studio")
        
        tab_tech, tab_growth, tab_bs = st.tabs(["🕯️ Candlesticks & Technicals", "📊 Growth Trajectory", "⚖️ Debt & Cash Balance"])

        with tab_tech:
            ta_engine = TechnicalAnalysisEngine()
            ta_data = ta_engine.fetch_and_analyze(selected_ticker)

            if "error" in ta_data:
                st.error(ta_data["error"])
            else:
                tc1, tc2, tc3 = st.columns(3)
                with tc1:
                    st.metric("Live Market Price", f"${ta_data['current_price']}", f"{ta_data['price_change_pct']}%")
                with tc2:
                    st.metric("RSI (14-Day)", f"{ta_data['rsi_14']}")
                with tc3:
                    st.metric("50 SMA / 200 SMA", f"${ta_data['sma_50']} / ${ta_data['sma_200']}")

                with st.expander("🎯 Active Technical Momentum Signals", expanded=True):
                    for sig in ta_data["technical_signals"]:
                        st.info(sig)

                history = ta_data["history"]
                if history:
                    df_chart = pd.DataFrame(history)
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.06,
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f"{selected_ticker} Candlesticks & Moving Averages", "RSI (14)")
                    )

                    fig.add_trace(
                        go.Candlestick(
                            x=df_chart["Date"],
                            open=df_chart["Open"],
                            high=df_chart["High"],
                            low=df_chart["Low"],
                            close=df_chart["Close"],
                            name="OHLC"
                        ),
                        row=1, col=1
                    )

                    fig.add_trace(
                        go.Scatter(x=df_chart["Date"], y=df_chart["SMA_50"], line=dict(color="orange", width=1.5), name="50 SMA"),
                        row=1, col=1
                    )

                    fig.add_trace(
                        go.Scatter(x=df_chart["Date"], y=df_chart["SMA_200"], line=dict(color="cyan", width=1.5), name="200 SMA"),
                        row=1, col=1
                    )

                    fig.add_trace(
                        go.Scatter(x=df_chart["Date"], y=df_chart["RSI_14"], line=dict(color="#b388ff", width=1.5), name="RSI"),
                        row=2, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                    fig.update_layout(
                        height=480,
                        xaxis_rangeslider_visible=False,
                        template="plotly_dark",
                        margin=dict(l=10, r=10, t=30, b=10)
                    )

                    st.plotly_chart(fig, use_container_width=True)

        with tab_growth:
            if not df_multiyear.empty:
                fig_t_rev = px.bar(
                    df_multiyear,
                    x="Fiscal Year",
                    y=["Revenue ($M)", "Net Income ($M)"],
                    barmode="group",
                    title=f"{selected_ticker} Multi-Year Revenue & Net Income ($M)",
                    color_discrete_sequence=["#4a86e8", "#00c853"],
                    template="plotly_dark",
                    height=460
                )
                st.plotly_chart(fig_t_rev, use_container_width=True)

                fig_t_mar = px.line(
                    df_multiyear,
                    x="Fiscal Year",
                    y=["Gross Margin Raw", "Op. Margin Raw"],
                    markers=True,
                    title=f"{selected_ticker} Margin Evolution Over Time (%)",
                    color_discrete_sequence=["#ff9800", "#00e676"],
                    template="plotly_dark",
                    height=380
                )
                st.plotly_chart(fig_t_mar, use_container_width=True)

        with tab_bs:
            if not df_multiyear.empty:
                fig_t_fcf = px.bar(
                    df_multiyear,
                    x="Fiscal Year",
                    y=["Free Cash Flow ($M)", "Total Debt ($M)"],
                    barmode="group",
                    title=f"{selected_ticker} Free Cash Flow vs. Total Debt ($M)",
                    color_discrete_sequence=["#00b0ff", "#ff5252"],
                    template="plotly_dark",
                    height=460
                )
                st.plotly_chart(fig_t_fcf, use_container_width=True)

    # RIGHT COLUMN: Dedicated Annual Chat Studio
    with right_col:
        col_c_hdr1, col_c_hdr2 = st.columns([3, 1])
        with col_c_hdr1:
            st.subheader(f"💬 {selected_ticker} Copilot")
        with col_c_hdr2:
            if st.button("🗑️ Reset", key=f"reset_chat_{selected_ticker}", help="Clear conversation history"):
                st.session_state[f"chat_history_{selected_ticker}"] = []
                st.rerun()

        st.caption(f"Grounded in {selected_ticker} Form 10-K Annual Disclosures.")

        chat_history_key = f"chat_history_{selected_ticker}"
        if chat_history_key not in st.session_state:
            st.session_state[chat_history_key] = [
                {
                    "role": "assistant",
                    "content": f"Hello! I am your Equity Research Copilot for **{selected_ticker}**. Ask me any question about multi-year revenue growth, gross/operating margins, balance sheet health, or Item 1A risk factors.",
                    "citations": []
                }
            ]

        deep_chat_container = st.container(height=430)
        with deep_chat_container:
            for msg in st.session_state[chat_history_key]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("citations"):
                        with st.expander("📑 Audit Citations"):
                            for cit in msg["citations"]:
                                st.markdown(f"**Page {cit['page_number']} [{cit['section']}]** *(Score: {cit['score']})*")
                                st.caption(cit["content_snippet"])

        with st.form(key=f"deep_chat_form_{selected_ticker}", clear_on_submit=True):
            d_cols = st.columns([5, 1])
            with d_cols[0]:
                deep_prompt = st.text_input("Ask about financials...", label_visibility="collapsed", placeholder=f"Ask about {selected_ticker} annual revenue, margins, debt, risks...")
            with d_cols[1]:
                submit_deep = st.form_submit_button("Send 💬", use_container_width=True)

            if submit_deep and deep_prompt:
                st.session_state[chat_history_key].append({"role": "user", "content": deep_prompt})
                with deep_chat_container:
                    with st.chat_message("user"):
                        st.markdown(deep_prompt)
                    with st.chat_message("assistant"):
                        with st.spinner(f"Retrieving {selected_ticker} 10-K disclosures & analyzing..."):
                            qa = FinancialQAService()
                            response = run_async_safe(qa.answer_question(question=deep_prompt, ticker=selected_ticker))
                            st.markdown(response["answer"])
                            if response.get("citations"):
                                with st.expander("📑 Audit Citations"):
                                    for cit in response["citations"]:
                                        st.markdown(f"**Page {cit['page_number']} [{cit['section']}]** *(Score: {cit['score']})*")
                                        st.caption(cit["content_snippet"])

                            st.session_state[chat_history_key].append({
                                "role": "assistant",
                                "content": response["answer"],
                                "citations": response.get("citations", [])
                            })
                st.rerun()
