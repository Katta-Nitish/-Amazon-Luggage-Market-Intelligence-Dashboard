import requests
import pandas as pd
import json
import time
import os
import re

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
OLLAMA_MODEL = "gemma3:12b"
OLLAMA_URL   = "http://localhost:11434/api/generate"

os.makedirs("data/cleaned", exist_ok=True)

# ─────────────────────────────────────────
# LOCAL LLM CALL
# ─────────────────────────────────────────
def call_llm(prompt, max_tokens=2000):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": max_tokens}
            },
            timeout=600
        )
        return response.json().get('response', '').strip()
    except Exception as e:
        print(f"  LLM error: {e}")
        return ""

def parse_json(raw, expected="{"):
    raw = raw.strip()
    # Strip <think>...</think> blocks
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip().lstrip("json").strip()
            if part.startswith(expected):
                raw = part
                break
    start = raw.find(expected)
    end   = raw.rfind("]" if expected == "[" else "}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(raw[start:end])
    except:
        return None

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
print("Loading data...")
products_df = pd.read_csv("data/raw/products.csv")
reviews_df  = pd.read_csv("data/raw/reviews_raw.csv")
print(f"  Products : {len(products_df)}")
print(f"  Reviews  : {len(reviews_df)}")
print(f"  Brands   : {reviews_df['brand'].nunique()}")

# ─────────────────────────────────────────
# STEP 1: PER-PRODUCT SENTIMENT
# ─────────────────────────────────────────
def analyze_product(asin, brand, title, price, rating, reviews):
    reviews_text = "\n".join([
        f"[{i+1}] {r['rating']}star | {r.get('title','')} - {r.get('body','')}"
        for i, r in enumerate(reviews[:15])
    ])

    prompt = f"""Analyze these Amazon India customer reviews for a luggage product.

Product: {title[:100]}
Brand: {brand} | Price: Rs.{price} | Listed rating: {rating}/5

Reviews:
{reviews_text}

Return ONLY a JSON object. No preamble, no explanation, no markdown. Start directly with {{ and end with }}.
{{
  "overall_sentiment_score": <float 0.0-10.0>,
  "sentiment_label": "<Excellent|Good|Average|Poor>",
  "total_positive": <int>,
  "total_negative": <int>,
  "total_neutral": <int>,
  "aspects": {{
    "wheels":     {{"score": <0-10>, "summary": "<one line>"}},
    "handle":     {{"score": <0-10>, "summary": "<one line>"}},
    "zipper":     {{"score": <0-10>, "summary": "<one line>"}},
    "material":   {{"score": <0-10>, "summary": "<one line>"}},
    "weight":     {{"score": <0-10>, "summary": "<one line>"}},
    "size":       {{"score": <0-10>, "summary": "<one line>"}},
    "lock":       {{"score": <0-10>, "summary": "<one line>"}},
    "durability": {{"score": <0-10>, "summary": "<one line>"}}
  }},
  "top_positives": ["<theme>", "<theme>", "<theme>"],
  "top_negatives": ["<theme>", "<theme>", "<theme>"],
  "one_line_verdict": "<one sentence customer perception summary>",
  "anomaly": "<surprising finding or null>"
}}"""

    raw = call_llm(prompt, max_tokens=1500)
    if not raw:
        return None
    return parse_json(raw, "{")


print(f"\n{'='*55}")
print("STEP 1: Per-product sentiment analysis...")
print(f"{'='*55}")

product_sentiments = []

for _, prod in products_df.iterrows():
    asin   = prod['asin']
    brand  = prod['brand']
    title  = prod['title']
    price  = prod['selling_price']
    rating = prod['rating']

    prod_reviews = reviews_df[reviews_df['asin'] == asin][
        ['rating', 'title', 'body']
    ].to_dict('records')

    if not prod_reviews:
        print(f"  ⚠️  No reviews for {asin}")
        continue

    print(f"  [{brand}] {str(title)[:50]}...", end=" ", flush=True)
    result = analyze_product(asin, brand, title, price, rating, prod_reviews)

    if result:
        asp = result.get('aspects', {})
        product_sentiments.append({
            'asin':             asin,
            'brand':            brand,
            'title':            title,
            'selling_price':    price,
            'list_price':       prod['list_price'],
            'discount_pct':     prod['discount_pct'],
            'actual_rating':    rating,
            'review_count':     prod['review_count'],
            'sentiment_score':  result.get('overall_sentiment_score'),
            'sentiment_label':  result.get('sentiment_label'),
            'total_positive':   result.get('total_positive'),
            'total_negative':   result.get('total_negative'),
            'total_neutral':    result.get('total_neutral'),
            'top_positives':    json.dumps(result.get('top_positives', [])),
            'top_negatives':    json.dumps(result.get('top_negatives', [])),
            'one_line_verdict': result.get('one_line_verdict'),
            'anomaly':          result.get('anomaly'),
            'asp_wheels':       asp.get('wheels',     {}).get('score'),
            'asp_handle':       asp.get('handle',     {}).get('score'),
            'asp_zipper':       asp.get('zipper',     {}).get('score'),
            'asp_material':     asp.get('material',   {}).get('score'),
            'asp_weight':       asp.get('weight',     {}).get('score'),
            'asp_size':         asp.get('size',       {}).get('score'),
            'asp_lock':         asp.get('lock',       {}).get('score'),
            'asp_durability':   asp.get('durability', {}).get('score'),
            'sum_wheels':       asp.get('wheels',     {}).get('summary'),
            'sum_handle':       asp.get('handle',     {}).get('summary'),
            'sum_zipper':       asp.get('zipper',     {}).get('summary'),
            'sum_material':     asp.get('material',   {}).get('summary'),
            'sum_weight':       asp.get('weight',     {}).get('summary'),
            'sum_size':         asp.get('size',       {}).get('summary'),
            'sum_lock':         asp.get('lock',       {}).get('summary'),
            'sum_durability':   asp.get('durability', {}).get('summary'),
        })
        print(f"✅ {result.get('overall_sentiment_score')}/10 ({result.get('sentiment_label')})")
    else:
        print("⚠️  failed")

    time.sleep(0.2)

sentiment_df = pd.DataFrame(product_sentiments)
sentiment_df.to_csv("data/cleaned/product_sentiment.csv", index=False)
print(f"\n✅ Saved {len(sentiment_df)} products → data/cleaned/product_sentiment.csv")


# ─────────────────────────────────────────
# STEP 2: BRAND AGGREGATION
# ─────────────────────────────────────────
print(f"\n{'='*55}")
print("STEP 2: Brand-level aggregation...")
print(f"{'='*55}")

brand_rows = []

for brand, grp in sentiment_df.groupby('brand'):
    all_pos, all_neg, all_anomalies = [], [], []
    for _, r in grp.iterrows():
        try: all_pos.extend(json.loads(r['top_positives']))
        except: pass
        try: all_neg.extend(json.loads(r['top_negatives']))
        except: pass
        if pd.notna(r.get('anomaly')) and str(r['anomaly']) not in ['null','None','']:
            all_anomalies.append(str(r['anomaly']))

    avg_price = grp['selling_price'].mean()
    if avg_price < 1500:   band = "Budget"
    elif avg_price < 3000: band = "Mid-Range"
    elif avg_price < 5000: band = "Upper Mid"
    else:                  band = "Premium"

    value_score = round(grp['sentiment_score'].mean() / (avg_price / 1000), 2) if avg_price > 0 else 0

    brand_rows.append({
        'brand':               brand,
        'price_band':          band,
        'total_products':      len(grp),
        'total_reviews':       int(grp['review_count'].sum()),
        'avg_selling_price':   round(avg_price, 0),
        'price_min':           grp['selling_price'].min(),
        'price_max':           grp['selling_price'].max(),
        'avg_discount_pct':    round(grp['discount_pct'].mean(), 1),
        'avg_actual_rating':   round(grp['actual_rating'].mean(), 2),
        'avg_sentiment_score': round(grp['sentiment_score'].mean(), 2),
        'value_score':         value_score,
        'avg_wheels':          round(grp['asp_wheels'].mean(), 1),
        'avg_handle':          round(grp['asp_handle'].mean(), 1),
        'avg_zipper':          round(grp['asp_zipper'].mean(), 1),
        'avg_material':        round(grp['asp_material'].mean(), 1),
        'avg_weight':          round(grp['asp_weight'].mean(), 1),
        'avg_size':            round(grp['asp_size'].mean(), 1),
        'avg_lock':            round(grp['asp_lock'].mean(), 1),
        'avg_durability':      round(grp['asp_durability'].mean(), 1),
        'top_positives':       json.dumps(list(dict.fromkeys(all_pos))[:5]),
        'top_negatives':       json.dumps(list(dict.fromkeys(all_neg))[:5]),
        'anomalies':           " | ".join(all_anomalies) if all_anomalies else "None detected",
    })
    print(f"  ✅ {brand:20s} | ₹{round(avg_price):,} | "
          f"sentiment:{round(grp['sentiment_score'].mean(),1)}/10 | "
          f"value:{value_score}")

brand_df = pd.DataFrame(brand_rows)
brand_df.to_csv("data/cleaned/brand_summary.csv", index=False)
print(f"\n✅ Saved → data/cleaned/brand_summary.csv")


# ─────────────────────────────────────────
# STEP 3: AGENT INSIGHTS
# ─────────────────────────────────────────
print(f"\n{'='*55}")
print("STEP 3: Agent Insights...")
print(f"{'='*55}")

brand_data = brand_df[[
    'brand','price_band','avg_selling_price','avg_discount_pct',
    'avg_actual_rating','avg_sentiment_score','value_score',
    'avg_durability','avg_wheels','avg_zipper',
    'top_positives','top_negatives','anomalies'
]].to_dict('records')

insight_prompt = f"""You are a senior market analyst for Amazon India luggage brands.

Data:
{json.dumps(brand_data, indent=2)}

Generate exactly 7 non-obvious competitive intelligence insights.
Go beyond surface stats. Find contradictions, gaps, opportunities.

Return ONLY a JSON array. No preamble, no explanation, no markdown. Start directly with [ and end with ].
[
  {{
    "insight_number": <1-7>,
    "category": "<Pricing|Sentiment|Quality|Positioning|Opportunity|Risk>",
    "title": "<punchy 5-8 word title>",
    "detail": "<2-3 sentences with specific numbers>",
    "implication": "<one sentence action for brand manager>"
  }}
]"""

raw = call_llm(insight_prompt, max_tokens=2500)
insights = parse_json(raw, "[")

if insights:
    insights_df = pd.DataFrame(insights)
    insights_df.to_csv("data/cleaned/agent_insights.csv", index=False)
    print(f"\n✅ {len(insights)} insights → data/cleaned/agent_insights.csv")
    for ins in insights:
        print(f"\n  [{ins.get('category')}] {ins.get('title')}")
        print(f"  → {ins.get('implication')}")
else:
    print("  Insights generation failed — try rerunning step 3 separately")
    pd.DataFrame().to_csv("data/cleaned/agent_insights.csv", index=False)


# ─────────────────────────────────────────
# FINAL
# ─────────────────────────────────────────
print(f"\n{'='*55}")
print("ALL DONE")
print(f"{'='*55}")
print("  data/cleaned/product_sentiment.csv")
print("  data/cleaned/brand_summary.csv")
print("  data/cleaned/agent_insights.csv")
print("\nBrand ranking by sentiment:")
print(brand_df[['brand','price_band','avg_selling_price',
                'avg_sentiment_score','value_score','avg_durability'
               ]].sort_values('avg_sentiment_score', ascending=False).to_string(index=False))
