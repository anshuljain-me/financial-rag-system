"""
Financial Benchmark Evaluation Dataset:
Curated test cases across single-filer and multi-company comparative dimensions.
"""

EVALUATION_DATASET = [
    {
        "id": "TC-01",
        "category": "REVENUE_SCALE",
        "ticker": "AAPL",
        "question": "What was Apple's total net sales / revenue for the latest reported fiscal year?",
        "ground_truth_keywords": ["revenue", "sales", "million", "billion"],
        "expected_section": "ITEM 8"
    },
    {
        "id": "TC-02",
        "category": "MARGIN_DURABILITY",
        "ticker": "AAPL",
        "question": "What was Apple's gross profit and operating income (EBIT)?",
        "ground_truth_keywords": ["gross profit", "operating income", "EBIT"],
        "expected_section": "ITEM 8"
    },
    {
        "id": "TC-03",
        "category": "CASH_FLOW_AND_LEVERAGE",
        "ticker": "AAPL",
        "question": "What was Apple's Free Cash Flow and how does it compare to its Total Debt?",
        "ground_truth_keywords": ["free cash flow", "operating cash flow", "capital expenditures", "total debt"],
        "expected_section": "ITEM 8"
    },
    {
        "id": "TC-04",
        "category": "RISK_FACTORS",
        "ticker": "AAPL",
        "question": "What are the core Item 1A operational, supply chain, and regulatory risk factors disclosed in the 10-K?",
        "ground_truth_keywords": ["risk", "macroeconomic", "supply chain", "regulatory", "antitrust"],
        "expected_section": "ITEM 1A"
    },
    {
        "id": "TC-05",
        "category": "BUSINESS_OVERVIEW",
        "ticker": "AAPL",
        "question": "What is the primary business model and operational scope described in Item 1?",
        "ground_truth_keywords": ["business", "hardware", "services", "global"],
        "expected_section": "ITEM 1"
    },
    {
        "id": "TC-06",
        "category": "PER_SHARE_METRICS",
        "ticker": "AAPL",
        "question": "What was Apple's Diluted Earnings Per Share (EPS)?",
        "ground_truth_keywords": ["diluted", "earnings per share", "EPS"],
        "expected_section": "ITEM 8"
    },
    {
        "id": "TC-07",
        "category": "CAPITAL_EXPENDITURES",
        "ticker": "AAPL",
        "question": "How much capital expenditure (CapEx) was allocated during the fiscal year?",
        "ground_truth_keywords": ["capital expenditure", "capex", "cash flow"],
        "expected_section": "ITEM 8"
    },
    {
        "id": "TC-08",
        "category": "CROSS_COMPANY_BENCHMARK",
        "ticker": "ALL",
        "question": "Compare the operating margin durability and capital structure across the ingested companies.",
        "ground_truth_keywords": ["margin", "revenue", "debt", "cash"],
        "expected_section": "ALL"
    }
]
