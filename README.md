# Prompt Crash-Test Lab

**LLM Robustness Benchmark Framework**

CS 599: Contemporary Developments (Applications of Large Language Models)

---

## Overview

Prompt Crash-Test Lab is an automated framework that **systematically evaluates LLM robustness** by generating prompt variants, executing them across multiple models, and producing comprehensive stability metrics.

### Why This Exists

LLMs are sensitive to prompt phrasing — semantically equivalent prompts can yield drastically different outputs (20-40% accuracy fluctuation). This project provides:

- A **standardized benchmark dataset** (100 base prompts, 2000 variants)
- An **evaluation harness** testing 4 LLM providers
- **Quantitative robustness metrics** for evidence-based prompt engineering
- An **interactive dashboard** for exploring results

### Models Evaluated

| Model | Provider | Strengths |
|-------|----------|-----------|
| GPT-4-turbo | OpenAI | Instruction following, JSON mode |
| Claude 3.5 Sonnet | Anthropic | Reasoning, long context |
| Gemini 1.5 Pro | Google | Fast, cost-effective |
| Llama 3.1 70B | Together AI | Open source, cheapest |

---

## Project Structure

```
prompt-crashtest-lab/
├── config/
│   └── settings.py              # Central configuration
├── data/
│   ├── schemas/                  # 5 JSON schemas (ecommerce, medical, finance, restaurant, job)
│   ├── base_prompts/             # 100 base prompts (50 JSON + 50 Q&A)
│   └── variants/                 # Generated variants (2000 total)
├── src/
│   ├── variant_generator.py      # Generates 20 variants per prompt
│   ├── batch_runner.py           # Executes prompts across models with caching
│   ├── cache.py                  # SQLite response cache
│   ├── adversarial.py            # Adversarial mutation analysis
│   ├── parameter_sensitivity.py  # Temperature & system prompt studies
│   ├── analysis.py               # Statistical analysis & chart generation
│   ├── cli.py                    # Command-line interface
│   ├── model_clients/            # Unified API for 4 LLM providers
│   │   ├── base.py               # Abstract base class
│   │   ├── openai_client.py      # GPT-4 client
│   │   ├── anthropic_client.py   # Claude client
│   │   ├── gemini_client.py      # Gemini client
│   │   └── together_client.py    # Llama client (via Together AI)
│   └── scoring/                  # Evaluation metrics
│       ├── schema_validator.py   # JSON schema validation + field accuracy
│       ├── semantic_similarity.py # Embedding-based consistency
│       ├── answer_correctness.py # Q&A correctness + citation detection
│       └── robustness.py         # Core robustness score computation
├── dashboard/
│   └── app.py                    # Streamlit interactive dashboard
├── tests/                        # Unit tests (53 tests, all passing)
├── docs/                         # Project proposal & presentation
├── requirements.txt
└── pyproject.toml
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- API keys for at least one LLM provider

### 2. Installation

```bash
# Clone the repository
git clone <repo-url>
cd prompt-crashtest-lab

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy and edit environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 4. Run the Full Pipeline

```bash
# Step 1: Generate prompt variants (creates 2000 variants)
python -m src.cli generate

# Step 2: Run evaluation across models
# Full run (all models, all variants):
python -m src.cli run --task json_extraction
python -m src.cli run --task grounded_qa

# Budget-friendly run (subset):
python -m src.cli run --task json_extraction --models gpt-4-turbo claude-3.5-sonnet --max-variants 100

# Step 3: Generate adversarial variants
python -m src.cli adversarial

# Step 4: Run parameter sensitivity study
python -m src.cli sensitivity --task json_extraction --study temperature --model gpt-4-turbo
python -m src.cli sensitivity --task json_extraction --study system_prompt --model gpt-4-turbo

# Step 5: Score results and generate analysis
python -m src.cli score --task json_extraction
python -m src.cli score --task grounded_qa

# Step 6: Launch interactive dashboard
python -m src.cli dashboard

# Check status anytime
python -m src.cli status
```

---

## How to Test

### Run Unit Tests

```bash
# Run all 53 tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_scoring.py -v
python -m pytest tests/test_cache.py -v
python -m pytest tests/test_variant_generator.py -v
python -m pytest tests/test_adversarial.py -v
```

### Test Coverage Areas

| Module | Tests | What's Tested |
|--------|-------|---------------|
| `test_cache.py` | 6 | Cache put/get, miss, overwrite, key uniqueness, stats |
| `test_variant_generator.py` | 11 | Variant count, type distribution, content validation |
| `test_scoring.py` | 25 | JSON extraction, schema validation, field accuracy, keyword overlap, citation detection, robustness score |
| `test_adversarial.py` | 11 | Adversarial variant generation, negation, format swap, distractor injection, subtypes |

### Manual Integration Test

```bash
# Generate variants and verify
python -m src.cli generate
python -m src.cli status
# Expected: 1000 JSON extraction variants, 1000 Q&A variants

# Test with a small subset (no API keys needed for generation + scoring tests)
python -m pytest tests/ -v
```

---

## Evaluation Metrics

### Core Metrics

| Metric | Formula | Range | Description |
|--------|---------|-------|-------------|
| **Robustness Score** | `1 - (σ_variants / μ_accuracy)` | [0, 1] | How consistent a model is across prompt variants. Higher = more stable. |
| **Format Compliance** | `valid_outputs / total_outputs` | [0, 1] | Percentage of outputs that pass JSON schema validation. |
| **Field Accuracy** | `correct_fields / total_fields` | [0, 1] | Per-field extraction accuracy against ground truth. |
| **Semantic Consistency** | `mean(pairwise_cosine_sim)` | [0, 1] | Average embedding similarity between all output pairs. |
| **Answer Correctness** | `0.4×sem_sim + 0.3×kw_overlap + 0.3×citation` | [0, 1] | Weighted score for Q&A quality. |
| **Citation Rate** | `responses_with_quotes / total` | [0, 1] | How often the model includes supporting quotes. |
| **Cost Efficiency** | `tokens_per_correct_answer` | tokens | Tokens consumed per correct output. |

### Statistical Analysis

- **Mann-Whitney U test**: Non-parametric pairwise comparison between models (p < 0.05 significance)
- **Effect size**: Absolute mean difference between model pairs
- **Bootstrap confidence intervals**: 95% CI for robustness scores

### Variant Types Tested

| Type | Count/Prompt | Description |
|------|-------------|-------------|
| Paraphrase | 5 | Semantically equivalent rewording |
| Format | 4 | Markdown, plaintext, numbered, XML |
| Role | 3 | Expert, assistant, teacher personas |
| Constraint | 3 | Concise, detailed, simple language |
| Template | 5 | Zero-shot, few-shot, CoT, structured, step-by-step |

---

## Task Categories

### 1. JSON Extraction (50 base prompts)

Extract structured data from unstructured text across 5 domains:
- **E-commerce** (products): name, price, category, brand, stock, rating
- **Medical** (records): age, gender, diagnosis, treatment, medications, severity
- **Finance** (transactions): type, amount, currency, sender, recipient, date, status
- **Restaurant** (reviews): name, cuisine, rating, price range, dishes, atmosphere
- **Job Postings**: title, company, location, salary, type, skills, experience

### 2. Grounded Q&A (50 base prompts)

Answer questions based solely on provided context paragraphs:
- Topics: science, history, geography, technology, medicine, arts
- Requires: factual accuracy, context grounding, citation inclusion
- Prevents: hallucination, adding information beyond context

---

## Dashboard Features

The Streamlit dashboard provides:

1. **Model Leaderboard** — Ranked table of all models by robustness score
2. **Key Metrics Cards** — At-a-glance comparison of robustness, accuracy, compliance
3. **Model Comparison Charts** — Bar charts with error bars
4. **Variant Type Heatmap** — How each model performs on each variant type
5. **Distribution Analysis** — Box plots showing score spread per model
6. **Per-Prompt Robustness** — Scatter plot of robustness vs accuracy per prompt
7. **Cost Analysis** — Token usage vs accuracy trade-off visualization
8. **Statistical Significance** — Pairwise p-values between models
9. **Failure Analysis** — Drill into specific failures with response text
10. **Data Export** — Download CSV/JSON results

---

## How to Explain to Professor

### Elevator Pitch (30 seconds)
> "We built an automated framework that tests how robust LLMs are to prompt phrasing changes. We generated 2000 prompt variants across two tasks, evaluated 4 models, and found that [model X] is most robust while [model Y] struggles with [variant type Z]. Our dashboard lets you explore exactly where and why models fail."

### Key Talking Points

1. **Problem**: LLMs give different answers to the same question asked differently — this is dangerous for production systems
2. **Approach**: Systematically generate 20 variants per prompt (paraphrase, format, role, constraint, template) and test 4 models
3. **Innovation**: First benchmark focused specifically on prompt robustness (not just accuracy)
4. **Metrics**: Robustness Score quantifies consistency; combines with format compliance and semantic similarity
5. **Findings**: Share model-specific insights from your results
6. **Impact**: Helps practitioners choose models and write more robust prompts

### Impressive Aspects for Professor

- **Scale**: 100 base prompts × 20 variants × 4 models = 8,000 evaluations
- **Rigor**: Statistical significance testing (Mann-Whitney U), confidence intervals
- **Engineering**: SQLite caching, unified API abstraction, rate limiting, retry logic
- **Testing**: 53 unit tests covering all scoring, variant, and adversarial modules
- **Reproducibility**: Deterministic variant generation, cached results, open-source
- **Visualization**: Interactive Streamlit dashboard with 10+ chart types
- **Industry relevance**: Addresses a real production deployment challenge

---

## Budget & Cost Control

| Strategy | Savings |
|----------|---------|
| SQLite response caching | Eliminates duplicate API calls |
| `--max-variants` flag | Test subset before full run |
| Rate limiting + retry | Prevents wasted calls on failures |
| Cost estimation | Preview costs before execution |
| Multiple model tiers | Mix expensive/cheap models |

Estimated full run cost: ~$150-200 across all 4 models with 2000 variants.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11+ |
| LLM APIs | OpenAI, Anthropic, Google GenAI, Together AI |
| Data | Pandas, Pydantic, JSONL, SQLite |
| NLP | Sentence Transformers (all-MiniLM-L6-v2) |
| Validation | jsonschema |
| Dashboard | Streamlit, Plotly |
| Charts | Matplotlib, Seaborn |
| Testing | pytest (53 tests) |
| Code Quality | Ruff |

---

## License

MIT License — Open source for academic and research use.
