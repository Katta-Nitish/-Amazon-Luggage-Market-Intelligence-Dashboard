import requests
import pandas as pd
import json
import time
import os
import re

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
OLLAMA_MODEL        = "gemma3:12b"
OLLAMA_URL          = "http://localhost:11434/api/generate"
BATCH_SIZE          = 3   # products per LLM call — keeps JSON small enough to not truncate
REVIEWS_PER_PRODUCT = 8   # reduced from 12; still gives 424+ total reviews

os.makedirs("data/raw", exist_ok=True)

# ─────────────────────────────────────────
# LOCAL LLM CALL
# ─────────────────────────────────────────
def call_llm(prompt, max_tokens=4000):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "num_predict": max_tokens
                }
            },
            timeout=600
        )
        return response.json().get('response', '').strip()
    except Exception as e:
        print(f"  LLM call failed: {e}")
        return ""

def parse_json_array(raw):
    """Safely parse a JSON array, handles markdown fences, <think> tags, truncation."""
    raw = raw.strip()
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip().lstrip("json").strip()
            if part.startswith("["):
                raw = part
                break
    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        # Salvage truncated array — drop last incomplete object
        truncated = raw[start:]
        last_complete = truncated.rfind("},")
        if last_complete > 0:
            try:
                return json.loads(truncated[:last_complete+1] + "]")
            except:
                pass
        return None

# ─────────────────────────────────────────
# LOAD & CLEAN PRODUCTS
# ─────────────────────────────────────────
products_df = pd.read_csv("data/raw/products.csv")

def fix_brand(row):
    title    = str(row['title']).lower()
    declared = str(row['brand']).strip()
    if declared == 'Vip':
        if 'nasher miles' in title: return 'Nasher Miles'
        if 'aristocrat'   in title: return 'Aristocrat'
        if 'safari'       in title: return 'Safari'
        if 'skybags'      in title: return 'Skybags'
    return declared

products_df['brand'] = products_df.apply(fix_brand, axis=1)

MARKUP = {
    'Safari': 1.30, 'Skybags': 1.25,
    'American Tourister': 1.35, 'Vip': 1.28,
    'Aristocrat': 1.22, 'Nasher Miles': 1.32
}
def estimate_list_price(row):
    if pd.notna(row['list_price']) and float(row['list_price']) > 0:
        return row['list_price']
    return round(row['selling_price'] * MARKUP.get(row['brand'], 1.30), 0)

products_df['list_price']    = products_df.apply(estimate_list_price, axis=1)
products_df['selling_price'] = pd.to_numeric(products_df['selling_price'], errors='coerce')
products_df['list_price']    = pd.to_numeric(products_df['list_price'],    errors='coerce')
products_df['discount_pct']  = (
    (products_df['list_price'] - products_df['selling_price'])
    / products_df['list_price'] * 100
).round(1).fillna(0)

products_df.to_csv("data/raw/products.csv", index=False)

print(f"Loaded {len(products_df)} products across {products_df['brand'].nunique()} brands")
print(products_df['brand'].value_counts().to_string())

# ─────────────────────────────────────────
# BRAND PERSONAS
# ─────────────────────────────────────────
BRAND_PERSONAS = {
    "Safari":             {"known_for": "sturdy build, 8-wheel system, value for money",
                           "complaints": "handle loosens over time, average zipper quality"},
    "Skybags":            {"known_for": "trendy designs, lightweight, youth-oriented",
                           "complaints": "wheels wobble after heavy use, weak zippers"},
    "American Tourister": {"known_for": "brand trust, smooth silent wheels, TSA lock",
                           "complaints": "heavier than competitors, price premium"},
    "Vip":                {"known_for": "classic reliable brand, durable frame",
                           "complaints": "outdated designs, heavier build"},
    "Aristocrat":         {"known_for": "very affordable, decent for occasional travel",
                           "complaints": "thin material, wheels can crack"},
    "Nasher Miles":       {"known_for": "premium polycarbonate finish, TSA lock, modern design",
                           "complaints": "pricier than Safari, fewer service centers"},
}

# ─────────────────────────────────────────
# GENERATE — small batches to avoid truncation
# ─────────────────────────────────────────
def generate_batch(brand, batch_df, n_reviews=REVIEWS_PER_PRODUCT):
    persona = BRAND_PERSONAS.get(brand, {})
    products_list = "\n".join([
        f"ASIN:{r['asin']} | {str(r['title'])[:70]} | Rs.{r['selling_price']} | Rating:{r['rating']}"
        for _, r in batch_df.iterrows()
    ])
    n_total = len(batch_df) * n_reviews

    prompt = f"""Generate exactly {n_total} Amazon India customer reviews for {brand} luggage.

Brand known for: {persona.get('known_for', '')}
Common complaints: {persona.get('complaints', '')}

Products — write exactly {n_reviews} reviews per ASIN:
{products_list}

Rules:
- Indian customers, mention cities (Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Pune)
- Cover aspects: wheels, handle, zipper, material, weight, lock, durability
- Average rating per product should be close to listed Rating
- Vary review lengths (some 1-2 sentences, some 4-5 sentences)

Respond with ONLY a JSON array. No intro text, no markdown, no backticks.
Start your entire response with [ and end with ].
Schema: {{"asin":"<ASIN>","brand":"{brand}","rating":<1-5>,"title":"<headline>","body":"<review>","date":"Reviewed in India on March 10, 2024","verified_purchase":true,"helpful_votes":<0-30>}}"""

    raw = call_llm(prompt, max_tokens=4000)
    if not raw:
        return []

    result = parse_json_array(raw)
    if not result:
        print(f"\n    ⚠️  parse failed. Preview: {raw[:120]}")
        return []

    valid_asins = set(batch_df['asin'].astype(str))
    clean = []
    for r in result:
        asin = str(r.get('asin', '')).strip()
        if not asin or not r.get('body') or asin not in valid_asins:
            continue
        clean.append({
            'asin':              asin,
            'brand':             brand,
            'product_title':     '',
            'rating':            min(5, max(1, int(r.get('rating', 3)))),
            'title':             str(r.get('title', '')),
            'body':              str(r.get('body', '')),
            'date':              str(r.get('date', '')),
            'verified_purchase': bool(r.get('verified_purchase', True)),
            'helpful_votes':     int(r.get('helpful_votes', 0)),
        })
    return clean

# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
all_reviews = []

print(f"\n{'='*55}")
print(f"Model: {OLLAMA_MODEL} | Batch: {BATCH_SIZE} products | {REVIEWS_PER_PRODUCT} reviews each")
print(f"{'='*55}")

for brand in products_df['brand'].unique():
    brand_prods = products_df[products_df['brand'] == brand].reset_index(drop=True)
    n_products  = len(brand_prods)
    n_batches   = (n_products + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n[{brand}] {n_products} products → {n_batches} batches")

    brand_reviews = []
    for i in range(n_batches):
        batch = brand_prods.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        print(f"  Batch {i+1}/{n_batches} ({len(batch)} products)...", end=" ", flush=True)
        got = generate_batch(brand, batch)
        brand_reviews.extend(got)
        print(f"got {len(got)}")
        time.sleep(0.5)

    # Retry any ASIN that got zero reviews
    covered = {r['asin'] for r in brand_reviews}
    missing_rows = [row for _, row in brand_prods.iterrows() if str(row['asin']) not in covered]
    if missing_rows:
        print(f"  ↩️  Retrying {len(missing_rows)} uncovered ASINs...")
        retry_df = pd.DataFrame(missing_rows)
        for i in range(0, len(retry_df), BATCH_SIZE):
            batch = retry_df.iloc[i:i+BATCH_SIZE]
            got = generate_batch(brand, batch)
            brand_reviews.extend(got)
            print(f"    retry: got {len(got)}")
            time.sleep(0.5)

    expected = n_products * REVIEWS_PER_PRODUCT
    print(f"  ✅ {len(brand_reviews)}/{expected} reviews for {brand}")
    all_reviews.extend(brand_reviews)

# ─────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────
reviews_df = pd.DataFrame(all_reviews)
reviews_df['rating']        = pd.to_numeric(reviews_df['rating'],        errors='coerce')
reviews_df['helpful_votes'] = pd.to_numeric(reviews_df['helpful_votes'], errors='coerce')
reviews_df.to_csv("data/raw/reviews_raw.csv", index=False)

print(f"\n{'='*55}")
print(f"Total reviews : {len(reviews_df)}")
print(reviews_df.groupby('brand')['rating'].count().to_string())
print(f"\n✅ Saved → data/raw/reviews_raw.csv")
print(f"\nReviews per ASIN (stats):")
print(reviews_df.groupby('asin').size().describe().to_string())