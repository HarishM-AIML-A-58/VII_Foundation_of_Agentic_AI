# Mini-Project: Evidence-First Multi-Agent AI for NSE Equity Research & Trading

**Course:** AD23731 - Foundations of Agentic AI (Unit II, III, IV, V & Mini-Project)  
**Student:** M Harish (231501058) | IV Year / VII Sem B.Tech AIML  
**Institution:** Rajalakshmi Engineering College (Autonomous), Chennai  

---

## 1. Executive Summary & Problem Definition

Standard LLM chatbots and prompt-only trading bots fail in financial markets due to:
1. **Hallucination & Recency Bias:** Generating convincing but inaccurate financial claims without dated factual grounding.
2. **The "LLM Risk Judge" Flaw:** Prompts acting as risk judges can be persuaded by bullish news sentiment or prompt injection, violating risk parameters.
3. **Friction Blindness:** Failing to model statutory exchange transaction costs (STT, GST, exchange charges, stamp duty = ~12 bps round-trip on NSE delivery), turning positive gross setups into losing trades.

This system presents an **evidence-first, multi-agent quantitative architecture** combining:
- **Agentic RAG:** An 8-tool financial research desk requiring dated evidence citation.
- **Adversarial Multi-Agent Debate:** Structured Bull vs. Bear graph to falsify hypotheses.
- **Deterministic Code Veto:** A pure Python arithmetic gate that calculates net reward-to-risk after round-trip costs, unconditionally blocking non-viable trades.
- **Jev System One Scanner:** A sub-second quantitative volatility-regime model evaluated via fractional Kelly sizing.
- **Walk-Forward Validation:** 20 rolling train/test windows across 10 years of NSE history achieving **0.95 Walk-Forward Efficiency (WFE)**.

---

## 2. System Architecture & Component Mapping

```
                                [ Daily Point-in-Time Data / Screener / Bars ]
                                                       │
                                                       ▼
                                          ┌─────────────────────────┐
                                          │  Evidence Research Desk │  (8 dated tools)
                                          └────────────┬────────────┘
                                                       │
                                                       ▼
                                          ┌─────────────────────────┐
                                          │ Multi-Agent Debate Graph│  (LangGraph stateful loop)
                                          │   Bull vs. Bear Debate  │
                                          └────────────┬────────────┘
                                                       │
                                                       ▼
                                          ┌─────────────────────────┐
                                          │   Jev Fast-Reflex Quant │  (Volatility regime)
                                          └────────────┬────────────┘
                                                       │
                     ┌─────────────────────────────────┴─────────────────────────────────┐
                     │                                                                   │
                     ▼                                                                   ▼
       [ Proposed Trade Signal ]                                           [ Deterministic Code Veto ]
    (Gross R:R, Entry, Stop, Target)                                        - Stop Geometry Check
                     │                                                      - Regulatory CNC Check
                     │                                                      - Net R:R >= 1.8 After Costs
                     │                                                      - Reward >= 3x Round-Trip
                     └─────────────────────────────────┬─────────────────────────────────┘
                                                       │
                                      ┌────────────────┴────────────────┐
                                      ▼                                 ▼
                                 [ APPROVED ]                       [ VETOED ]
                             Fractional Kelly Sizing            Logged to Ledger
                             Order Execution (SEBI API)         (Zero Risk Taken)
```

### Module Guide

| Directory | Core Files | Responsibility |
|---|---|---|
| **`agents/`** | `graph.py`, `prompts/`, `schemas.py`, `state.py` | Multi-agent definitions, Bull/Bear personas, prompt management, and Pydantic schemas. |
| **`orchestration/`** | `debate.py`, `pipelines/daily_research.py` | Stateful LangGraph workflow executing multi-turn debate turns and state merging. |
| **`risk/`** | `veto.py`, `limits.py`, `kill_switch.py`, `jev_kelly_sizer.py` | Deterministic Python risk gates, after-cost reward/risk verification, and Kelly sizing. |
| **`jev_quant/`** | `jev_questions.py`, `jev_runner.py`, `jev_client.py` | Sub-second System One fast-reflex probability scoring on market volatility regimes. |
| **`strategy/`** | `jev_calibrated_strategy.py`, `value_investing.py` | Quantitative strategy implementations conforming to the base Strategy protocol. |
| **`backtest/`** | `walk_forward.py`, `cpcv.py`, `statistics.py` | 20-window Walk-Forward Analysis (Pardo 2008), Deflated Sharpe Ratio, and CPCV. |
| **`costs/`** | `indian_statutory_sheet.py`, `calculator.py` | Exact Indian exchange fees (STT, GST, Stamp Duty, Turnover charges). |
| **`domain/`** | `signal.py`, `order.py`, `money.py`, `enums.py` | Immutable domain data structures with strict Decimal arithmetic. |

---

## 3. The Advise vs. Decide Boundary (The Code Veto)

The core principle distinguishing this system from toy chatbot frameworks is that **AI agents only advise; pure deterministic code decides**.

In [`risk/veto.py`](file:///Users/harishm/Desktop/VII_Foundation_of_Agentic_AI/mini_project_trading_agent/risk/veto.py), every trade is checked against six hard mathematical barriers:
1. **Actionability:** Validates that the signal is actionable (`BUY` or `SELL`), discarding unformed setups.
2. **Regulatory & Segment Soundness (`SHORT_IN_DELIVERY`):** Prohibits short selling under cash delivery (`CNC`) on NSE (which requires intraday `MIS` or futures).
3. **Protective Stop Geometry:** Enforces valid stop-loss placement ($> 0$ and on the protective side of entry).
4. **Net Reward-to-Risk Calculation:** Recomputes the reward-to-risk ratio *net of all round-trip statutory costs* (~12 basis points):
   $$\text{Net R:R} = \frac{(\text{Target} - \text{Entry}) - \text{Cost}}{(\text{Entry} - \text{Stop}) + \text{Cost}} \ge 1.8$$
5. **Cost Multiple Hurdle:** Reward must exceed round-trip friction by at least $3\times$ ($\ge 36\text{ bps}$), preventing micro-trades whose profits are consumed by taxes.

---

## 4. Empirical Validation & Outcomes

- **Untouched Out-of-Sample Holdout (2025-01 to 2026-09):** Candidate strategy delivered **+8.36%** after all taxes and statutory charges against **-1.39%** for the NIFTY 50 benchmark.
- **Walk-Forward Efficiency:** Best candidate achieved **0.95 WFE** across 20 walk-forward windows.
- **Jev Quant Calibrated Edge:** +0.10R edge across 1,300 unseen NSE samples.
- **Test Integrity:** 192 backend unit and integration tests passing in CI.
