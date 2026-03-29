# 💼 Amazon Luggage Market Intelligence Dashboard

An AI-powered competitive intelligence dashboard for luggage brands on Amazon India. Built for the Moonshot AI Agent Internship Assignment.

---

## 🚀 Live Demo

👉 **[https://amazon-review-dashboard.streamlit.app/](https://amazon-review-dashboard.streamlit.app/)**

> To use the AI Assistant tab, bring your own free Gemini API key from [aistudio.google.com](https://aistudio.google.com/app/apikey)

---

## 📦 Project Structure

```
MOONSHOT ASSIGNMENT/
├── main.py                  # Streamlit dashboard
├── generate_reviews.py      # Step 1: Generate synthetic reviews via local LLM
├── sentiment.py             # Step 2: Sentiment analysis + brand aggregation + agent insights
├── brand_summary.csv        # Cleaned brand-level data
├── product_sentiment.csv    # Cleaned product-level sentiment data
├── agent_insights.csv       # 7 AI-generated competitive intelligence insights
├── requirements.txt         # Python dependencies
├── i3.png                   # Sidebar logo
├── data/
│   ├── raw/
│   │   ├── products.csv     # Raw product listings (53 products, 6 brands)
│   │   └── reviews_raw.csv  # Generated reviews (438 reviews)
│   └── cleaned/             # Same CSVs as root (pipeline output)
└── README.md
```

---

## 🧠 Architecture

```
products.csv
     │
     ▼
generate_reviews.py  ──►  reviews_raw.csv
(Local LLM via Ollama)
     │
     ▼
sentiment.py  ──►  product_sentiment.csv
(Per-product sentiment,        brand_summary.csv
 brand aggregation,            agent_insights.csv
 agent insights)
     │
     ▼
main.py  ──►  Streamlit Dashboard
```

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd "MOONSHOT ASSIGNMENT"
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard
```bash
streamlit run main.py
```

The cleaned CSVs are already included — you can run the dashboard immediately without re-running the pipeline.

> For the AI Assistant tab, enter your free Gemini API key in the sidebar. Get one at [aistudio.google.com](https://aistudio.google.com/app/apikey).

---

## 🔄 Re-running the Data Pipeline (Optional)

Only needed if you want to regenerate data from scratch.

### Prerequisites
- [Ollama](https://ollama.com) installed locally
- gemma3:12b model pulled

```bash
ollama pull gemma3:12b
```

### Run pipeline
```bash
# Step 1: Generate reviews (~20-30 mins)
python generate_reviews.py

# Step 2: Sentiment analysis + insights (~40-60 mins)
python sentiment.py
```

---

## 📊 Dashboard Views

| Tab | What it shows |
|-----|--------------|
| 📊 Executive Summary | KPI metrics, market overview, AI agent insights filtered by brand |
| 🥊 Comparison | Value vs sentiment scatter, discount analysis, pros/cons table |
| 🔍 Product Deep Dive | Per-product price, rating, sentiment, aspect scores, verdict |
| 🤖 AI Assistant | Chat interface powered by Gemini to query brand/product data |

---

## 📈 Data Scope

| Metric | Value |
|--------|-------|
| Brands | 6 (Safari, Skybags, American Tourister, VIP, Aristocrat, Nasher Miles) |
| Products | 53 |
| Reviews analyzed | 438 |
| Aspects scored | 8 (wheels, handle, zipper, material, weight, size, lock, durability) |

---

## 🏆 Key Findings

| Brand | Sentiment | Avg Price | Value Score |
|-------|-----------|-----------|-------------|
| American Tourister | 7.9/10 | ₹4,320 | 1.83 |
| Nasher Miles | 7.7/10 | ₹3,482 | 2.21 |
| Safari | 7.7/10 | ₹3,174 | 2.42 |
| Vip | 7.5/10 | ₹4,087 | 1.84 |
| Skybags | 7.4/10 | ₹2,633 | 2.81 |
| Aristocrat | 7.2/10 | ₹2,340 | 3.06 |

> **Safari is the sweet spot** — near-top sentiment at a mid-range price. American Tourister leads on sentiment but has the worst value score due to premium pricing. VIP charges near American Tourister prices but scores like Skybags — a clear positioning problem.

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| Data generation | Ollama + gemma3:12b (local LLM) |
| Sentiment analysis | Ollama + gemma3:12b (aspect-level) |
| Dashboard | Streamlit |
| Charts | Plotly Express |
| AI Assistant | Gemini 2.5 Flash via LangChain |
| Data processing | Pandas |

---

## ⚠️ Limitations

- Reviews are **AI-generated** (synthetic), not scraped from Amazon directly, due to Amazon's anti-scraping protections
- Sentiment scores are LLM-generated and may reflect prompt bias
- VIP brand had only 4 products due to catalog overlap with sub-brands
- AI Assistant is single-turn only (no follow-up memory)

---

## 🔮 Future Improvements

- Real scraping via Playwright with proxy rotation
- VADER / fine-tuned sentiment model as a cross-check
- Time-series tracking of price and sentiment changes
- Review trust signal detection (fake review patterns)
- Exportable PDF reports per brand
