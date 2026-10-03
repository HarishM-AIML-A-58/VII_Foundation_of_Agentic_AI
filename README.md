# Foundations of Agentic AI (AD23731)

Coursework repository, laboratory experiments, and capstone mini-project implementation for AD23731 Foundations of Agentic AI, Department of Artificial Intelligence and Machine Learning, Rajalakshmi Engineering College (Autonomous), Chennai.

---

## Student and Course Information

| Field | Detail |
| :--- | :--- |
| Student Name | M Harish |
| Register Number | 231501058 |
| Degree and Branch | B.Tech. Artificial Intelligence and Machine Learning |
| Year and Semester | IV Year / VII Semester |
| Course Code and Title | AD23731 - Foundations of Agentic AI |
| Academic Year | 2026 - 2027 |
| Institution | Rajalakshmi Engineering College (Autonomous), Thandalam, Chennai |

---

## Course Deliverables

The root directory contains all primary evaluated submissions, laboratory records, and capstone documentation:

1. `Agentic_AI_Lab_Record.pdf`: Official laboratory record documenting Experiment 1 and Experiment 2 with bonafide certificate, index, theory, implementation code, inputs, and verified outputs.
2. `AgenticAI Lab Record Front Pages.pdf`: Prescribed Anna University / REC institutional front sheets, bonafide declaration, and syllabus alignment index.
3. `Agentic_AI_Final_Project_Report.pdf`: Comprehensive 30-page engineering capstone project report with literature review, architectural flowcharts, methodology, results, and statutory references.
4. `LinearTrade Final Review.pdf`: Final project review presentation slide deck detailing the multi-agent trading system architecture, empirical backtest results, and risk gating.
5. `LinearTrade_Demo.mp4`: Complete system walkthrough video recording demonstrating multi-agent debate, Jev scanner execution, deterministic risk veto, and live portfolio dashboard.

---

## Repository Structure

```tree
.
├── Agentic AI Project/                 # Capstone: Multi-agent trading research system
│   ├── README.md                      # Mini-project specifications and architecture details
│   ├── agents/                        # LangGraph research graph, Bull/Bear analysts, prompts
│   ├── orchestration/                 # Stateful debate loops and daily research pipelines
│   ├── risk/                          # Deterministic code veto, hard limits, Kelly sizer
│   ├── jev_quant/                     # Sub-second fast-reflex quant regime scanner
│   ├── strategy/                      # Calibrated quantitative and value strategy definitions
│   ├── backtest/                      # 20-window Walk-Forward Analysis and Sharpe metrics
│   ├── costs/                         # NSE equity statutory fee model (STT, GST, Stamp duty)
│   └── domain/                        # Typed domain models for signals, trades, and orders
├── AgenticAI Lab Record Front Pages.pdf # Prescribed institutional front pages
├── Agentic_AI_Final_Project_Report.pdf # Final technical project report
├── Agentic_AI_Lab_Record.pdf          # Laboratory record with completed experiments
├── Lab Experiments/                   # Coursework laboratory experiments
│   ├── experiment_1_prompts.py        # Experiment 1: Prompt engineering and reasoning control
│   └── experiment_2_llm_apps.py       # Experiment 2: Core LLM applications (QA, Summary, Sentiment)
├── LinearTrade Final Review.pdf       # Capstone review presentation deck
├── LinearTrade_Demo.mp4               # Complete project demo walkthrough video
└── README.md                          # Repository documentation
```

---

## Laboratory Experiments

Located in [`Lab Experiments/`](file:///Users/harishm/Desktop/VII_Foundation_of_Agentic_AI/Lab%20Experiments/):

### Experiment 1: Prompt Engineering, Agent Autonomy, and Reasoning
* Source: `Lab Experiments/experiment_1_prompts.py`
* Focus: Systematic evaluation of prompting techniques on agent reasoning quality, latency, and determinism.
* Implementations:
  * Zero-shot baseline prompting for direct instruction adherence.
  * Few-shot in-context learning for structured schema conformity.
  * Chain-of-Thought (CoT) step-by-step logical decomposition.
  * ReAct (Reasoning + Acting) execution trace with tool invocations.
  * System persona enforcement and guardrail boundaries.

### Experiment 2: Core LLM Applications
* Source: `Lab Experiments/experiment_2_llm_apps.py`
* Focus: Production-oriented LLM application primitives evaluated against unstructured domain corpora.
* Implementations:
  * Extractive and synthesized Question Answering with confidence scoring and attribution.
  * Multi-perspective document summarization (Executive, Key Takeaways, Action Items).
  * Aspect-Based Sentiment Analysis (ABSA) with entity targeting and rationale output.

---

## Capstone Mini-Project Overview

### LinearTrade / Trading Agent: Evidence-First Multi-Agent Equity Research System
* Source directory: [`Agentic AI Project/`](file:///Users/harishm/Desktop/VII_Foundation_of_Agentic_AI/Agentic%20AI%20Project/)
* Scope: Architecture and core algorithms for automated equity analysis and risk-gated execution on the National Stock Exchange of India (NSE).
* Architecture:
  * Evidence-First RAG: Gathers real-time and historical financial data through 8 dated retrieval tools to prevent model hallucinations.
  * Adversarial Multi-Agent Debate: Bull and Bear analyst agents engage in structured counter-argument rounds to eliminate single-perspective bias.
  * Deterministic Code Veto: Hard safety layer implemented in pure Python. Recalculates expected reward-to-risk ratios net of round-trip Indian exchange statutory charges (~12 bps) and automatically rejects negative-expectancy trades.
  * Jev Fast-Reflex Quant Engine: Rule-based sub-second volatility regime detector operating in tandem with deliberative LLM agent reasoning.
  * Walk-Forward Validation: Validated across 20 rolling walk-forward windows over 10 years of NSE historical data, achieving 0.95 Walk-Forward Efficiency (WFE).

---

## Running the Code

### Requirements
* Python 3.10+
* Standard library (all core experiments run with zero external package dependencies)

### Execute Laboratory Experiments
```bash
python3 "Lab Experiments/experiment_1_prompts.py"
python3 "Lab Experiments/experiment_2_llm_apps.py"
```

### Inspect Capstone Code Modules
```bash
python3 -m py_compile "Agentic AI Project/risk/veto.py"
python3 -m py_compile "Agentic AI Project/jev_quant/scanner.py"
```

---

## Department and Institutional Affiliation

Department of Artificial Intelligence and Machine Learning  
Rajalakshmi Engineering College (Autonomous)  
Affiliated to Anna University, Chennai  
Rajalakshmi Nagar, Thandalam, Chennai - 602 105, Tamil Nadu, India
