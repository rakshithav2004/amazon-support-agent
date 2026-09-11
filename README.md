# Amazon Support Agent

An evidence-grounded customer-support agent built from the Customer Support on Twitter dataset. The system classifies customer messages, retrieves similar historical AmazonHelp resolutions, drafts a grounded reply, and decides whether the case can be auto-handled or should be escalated.

## 1. Problem framing

Customer-support messages are often short, noisy, ambiguous, and highly repetitive. The goal of this project is to build a lightweight support agent that learns from historical AmazonHelp interactions rather than relying only on generic language-model knowledge.

For each incoming customer message, the system:

1. Classifies the message into one of 10 support intents.
2. Retrieves similar historical customer-to-AmazonHelp interactions.
3. Generates a customer-facing response grounded in the retrieved evidence.
4. Decides whether to `auto_handle` or `escalate`.
5. Returns the evidence IDs used to support the response.

The primary dataset is the Customer Support on Twitter dataset (`thoughtvector/customer-support-on-twitter`). The project focuses on the AmazonHelp brand.

## 2. System architecture

```text
Customer message
       |
       v
Gemini intent classifier
       |
       v
FAISS semantic retrieval
       |
       v
Historical AmazonHelp evidence
       |
       v
Gemini grounded reply generator
       |
       v
Explainable escalation rules
       |
       +--------------------+
       |                    |
       v                    v
 AUTO_HANDLE             ESCALATE
```

### Components

* `src/intents/` — intent taxonomy and Gemini classifier
* `src/retrieval/` — SentenceTransformer embeddings and FAISS retrieval
* `src/generation/` — evidence-grounded response generation
* `src/escalation/` — explainable escalation rules
* `src/evaluation/` — LLM-as-judge evaluation
* `src/data/` — reusable dataset loading
* `src/api/` — FastAPI interface
* `scripts/` — data preparation and experiments
* `tests/` — automated tests

## 3. Intent taxonomy

The final taxonomy contains 10 intents:

| Intent              | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `delivery_tracking` | Delivery status, tracking, delayed delivery, missing package |
| `order_issue`       | General order problems                                       |
| `product_issue`     | Damaged, defective, or wrong product                         |
| `return_refund`     | Returns and refund issues                                    |
| `payment_billing`   | Payments, billing, unexpected charges                        |
| `account_login`     | Account access and security                                  |
| `prime_membership`  | Prime subscription and membership                            |
| `digital_content`   | Kindle, Prime Video, and other digital content               |
| `cancellation`      | Order/service cancellation                                   |
| `other`             | Messages outside the defined categories                      |

## 4. Dataset preparation

The raw dataset is intentionally not committed to the repository because it is large.

The preparation pipeline:

```text
twcs.csv
   |
   +--> AmazonHelp extraction
   |
   +--> customer -> AmazonHelp response pairing
   |
   +--> retrieval corpus cleaning
   |
   +--> semantic embeddings
   |
   +--> FAISS index
```

The final retrieval corpus contains:

* 168,823 initial customer/reply pairs
* 160,626 cleaned historical examples after removing low-value/generic responses

### Reproduce the data pipeline

```powershell
python -m scripts.inspect_dataset
python -m scripts.extract_amazon
python -m scripts.reconstruct_threads
python -m scripts.build_retrieval_corpus
python -m scripts.clean_retrieval_corpus
```

The raw dataset should be placed at:

```text
data/raw/twcs.csv
```

## 5. Golden evaluation set

A manually reviewed golden set of **200 examples** was created across the 10 intents.

Distribution:

| Intent            | Examples |
| ----------------- | -------: |
| delivery_tracking |       52 |
| return_refund     |       31 |
| product_issue     |       22 |
| digital_content   |       22 |
| payment_billing   |       21 |
| prime_membership  |       18 |
| account_login     |       13 |
| cancellation      |       12 |
| order_issue       |        6 |
| other             |        3 |

The `other` class is small, so its class-level metrics are unstable.

## 6. Intent classification results

Two baselines were evaluated on the 200-example golden set using a stratified 80/20 split for the classical baseline.

| Method                       |  Accuracy | Macro-F1 |
| ---------------------------- | --------: | -------: |
| Majority class               |     26.0% |        — |
| TF-IDF + Logistic Regression |     35.0% |     0.19 |
| Gemini 3.5 Flash-Lite        | **78.5%** | **0.72** |

Gemini improved accuracy by **43.5 percentage points** over the TF-IDF + Logistic Regression baseline.

### Gemini class-level F1

| Intent            |   F1 |
| ----------------- | ---: |
| delivery_tracking | 0.86 |
| digital_content   | 0.85 |
| product_issue     | 0.85 |
| return_refund     | 0.81 |
| account_login     | 0.79 |
| payment_billing   | 0.76 |
| cancellation      | 0.70 |
| order_issue       | 0.60 |
| prime_membership  | 0.59 |
| other             | 0.44 |

## 7. Historical retrieval

The retriever uses:

* `all-MiniLM-L6-v2`
* normalized sentence embeddings
* FAISS inner-product similarity search
* top-5 historical customer/reply pairs

For a test message:

> My package says delivered but I never received it.

the retriever returned five highly relevant historical cases, with similarity scores approximately between `0.76` and `0.80`.

This retrieval layer is used as the evidence source for reply generation.

## 8. Grounded reply generation

The reply generator receives:

* customer message
* predicted intent
* top historical evidence

The prompt instructs the model not to invent policies, refunds, timelines, guarantees, or procedures that are unsupported by the retrieved evidence.

Historical evidence IDs are retained internally and returned with the final result.

## 9. Escalation policy

Escalation is intentionally rule-based and explainable.

Current rules include:

```text
Potential account compromise  -> ESCALATE
Fraud / unauthorized payment  -> ESCALATE
No supporting evidence        -> ESCALATE
Weak retrieval match          -> ESCALATE
Otherwise supported case      -> AUTO_HANDLE
```

Example:

```json
{
  "decision": "escalate",
  "decision_reason": "Potential account security compromise requires human review."
}
```

## 10. Reply-quality evaluation

A 60-example reply evaluation set was generated.

An LLM judge scored each response on:

* Grounding
* Correctness
* Helpfulness
* Unsupported claims

### LLM judge results

| Metric             |        Score |
| ------------------ | -----------: |
| Grounding          |     4.97 / 5 |
| Correctness        |     4.73 / 5 |
| Helpfulness        |     4.67 / 5 |
| Unsupported claims |     4.97 / 5 |
| Overall average    | **4.83 / 5** |

These scores indicate strong grounding and low rates of unsupported claims on the evaluated sample, but they should not be treated as definitive human-quality measurements.

## 11. Human-vs-LLM judge agreement

The same 60 examples were independently rated by a human using the same four dimensions.

### Agreement results

| Metric             | Exact agreement | Within ±1 |
| ------------------ | --------------: | --------: |
| Grounding          |           76.7% |     96.7% |
| Correctness        |           61.7% |     83.3% |
| Helpfulness        |           65.0% |     85.0% |
| Unsupported claims |           78.3% |     98.3% |
| **Overall**        |       **70.4%** | **90.8%** |

The average weighted Cohen's kappa was **-0.016**, and the overall Spearman correlation was **-0.050**.

### Interpretation

The judge frequently stayed within one point of the human rating, but the near-zero rank correlation shows that the LLM judge was not reliably ranking examples in the same way as the human evaluator.

Therefore, the LLM judge should be treated as a useful automated evaluation signal, **not as a fully validated substitute for human evaluation**.

## 12. Top 5 failure modes

### 1. Unsafe historical resolution copied forward

**Example:** Tweet `1445344`

The generated response reproduced a historical instruction asking the customer to submit details through a support link, even though another response in the same interaction warned about public-page privacy.

**Hypothesis:** Retrieval can surface historically used responses that were themselves inconsistent or inappropriate. Retrieval-grounding alone does not guarantee that the historical response is safe to reuse.

### 2. Empathy without actionable help

**Example:** Tweet `1239551`

The response was essentially an apology and acknowledgement without a useful next step.

**Hypothesis:** When the nearest historical response is only a short acknowledgement, the generator may preserve that lack of actionability.

### 3. Historical answer is grounded but inadequate

**Example:** Tweet `21936`

The system reproduced a historical response about forwarding feedback internally, but that did not resolve the customer's confusion about the returned order and refund.

**Hypothesis:** Exact historical similarity can favor an old response even when the historical response was not actually an effective resolution.

### 4. Contextually incorrect reuse of a past resolution

**Example:** Tweet `2454141`

The customer reported being charged for a cancelled item and wanted a refund, while the generated response focused on returning the item.

**Hypothesis:** Semantic similarity can retrieve a response with overlapping vocabulary but a different operational context.

### 5. Repeated support redirection

**Example:** Tweet `80621`

The customer explicitly said they had already followed previous support instructions, but the generated response redirected them to support again.

**Hypothesis:** The current retrieval setup does not model conversation history strongly enough to detect that a previous support path has already been attempted.

## 13. What is misleading about my headline number?

The headline **78.5% intent accuracy** is useful, but it is not a measure of end-to-end customer-support quality.

It is measured on a manually labelled 200-example golden set and evaluates only the intent classification component. It does not mean that 78.5% of customer conversations would receive a correct, useful, safe, or satisfactory final answer.

The reply evaluation was performed on a separate 60-example sample, and the human-vs-LLM judge analysis also showed that the automated judge has limitations.

Therefore, the headline number should be interpreted as **intent-classification accuracy under this evaluation setup**, not overall agent success.

## 14. What was not built

To keep the project focused, the following were intentionally not implemented:

* live Twitter/X API integration
* real Amazon account or order access
* payment processing
* production deployment
* autonomous refund/cancellation actions
* fine-tuning a language model
* a frontend application
* persistent customer profiles
* enterprise-scale monitoring

The system is a research/prototype support agent using historical public support interactions.

## 15. API

A FastAPI interface is provided.

### Health

```http
GET /health
```

### Support

```http
POST /support
```

Example request:

```json
{
  "message": "My package says delivered but I never received it."
}
```

Example response structure:

```json
{
  "customer_message": "...",
  "intent": "delivery_tracking",
  "intent_reason": "...",
  "reply": "...",
  "grounding_note": "...",
  "decision": "auto_handle",
  "decision_reason": "...",
  "evidence_ids": [
    "2198999_2198998",
    "1298456_1298454"
  ]
}
```

## 16. Installation

Use Python 3.11+.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

Never commit `.env`.

## 17. Run the project

### Test the intent classifier

```powershell
python -m src.intents.gemini_classifier
```

### Test retrieval

```powershell
python -m src.retrieval.retriever
```

### Test grounded reply generation

```powershell
python -m src.generation.reply_generator
```

### Test escalation

```powershell
python -m src.escalation.decision
```

### Test the complete agent

```powershell
python -m src.agent
```

### Run automated tests

```powershell
pytest -q
```

The current automated suite passes with:

```text
6 passed
```

## 18. One-week next steps

### Days 1–2

Improve conversation reconstruction and use multi-turn context during retrieval and generation.

### Days 3–4

Add retrieval quality checks, evidence filtering, and stronger safeguards against unsafe historical responses.

### Days 5–6

Improve escalation using calibrated retrieval confidence and conversation-state signals.

### Day 7

Expand the human evaluation set and redesign the judge rubric based on disagreements observed in this evaluation.

## 19. Repository structure

```text
amazon-support-agent/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── golden/
│
├── src/
│   ├── api/
│   ├── data/
│   ├── escalation/
│   ├── evaluation/
│   ├── generation/
│   ├── intents/
│   └── retrieval/
│
├── scripts/
├── tests/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 20. Final summary

This project demonstrates an evidence-grounded support-agent architecture combining:

* manually defined support intents
* classical ML baselines
* Gemini-based intent classification
* semantic retrieval over historical resolutions
* grounded response generation
* explainable escalation
* automated reply evaluation
* human-vs-LLM judge comparison
* automated tests

The strongest measured result is **78.5% intent accuracy with 0.72 macro-F1**, while the evaluation also documents important limitations and failure modes rather than treating the headline metric as end-to-end success.
