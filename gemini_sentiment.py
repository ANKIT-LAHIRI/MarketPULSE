# =====================================================================
#  Gemini financial-headline sentiment benchmark  (Ankit Lahiri)
#  Run in Google Colab: paste this whole file into ONE cell and run it.
#  You need a free Gemini API key: https://aistudio.google.com/apikey
#
#  What it does:
#   1. Loads human-labelled financial news headlines (Twitter Financial
#      News dataset: Bearish / Bullish / Neutral).
#   2. Takes a balanced random sample (70 per class = 210 headlines).
#   3. Sends them to Gemini in batches of 15 with a JSON-only prompt.
#   4. Compares Gemini's labels with the human labels and prints
#      accuracy, macro-F1, Cohen's kappa, a confusion matrix, latency
#      and token usage.
#   5. Saves every headline + both labels to gemini_sentiment_results.csv
#      (push this file to GitHub as evidence).
# =====================================================================

API_KEY = "PASTE_YOUR_KEY_HERE"   # <-- paste your Gemini key between the quotes (only place)

import subprocess
import sys
import getpass

import json
import os
import re
import time

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

# ----------------------------- settings ------------------------------
N_PER_CLASS = int(os.environ.get("N_PER_CLASS", 70))   # 70 x 3 classes = 210 headlines
BATCH_SIZE = 15                                        # headlines per Gemini request
PAUSE_SECONDS = 7                                      # stay under free-tier requests/minute
SEED = 42                                              # fixed seed -> same sample every run
MODELS_TO_TRY = ["gemini-flash-latest", "gemini-3.5-flash", "gemini-2.5-flash"]
CSV_PATH = os.environ.get("CSV_PATH")                  # optional: your own CSV with columns text,label
MOCK = os.environ.get("MOCK") == "1"                   # offline test mode (no API calls)

DATA_URL = ("https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment"
            "/resolve/main/sent_valid.csv")
DATASET_LABELS = {0: "negative", 1: "positive", 2: "neutral"}  # 0 Bearish, 1 Bullish, 2 Neutral
CLASSES = ["negative", "neutral", "positive"]

PROMPT = """You are a financial news analyst. For each headline below, judge the sentiment
for the stock or market it mentions. Use exactly one label per headline:
"positive" = good news for the price, "negative" = bad news, "neutral" = no clear price impact.
Return ONLY a JSON list with one object per headline, keeping the same ids:
[{"id": 0, "sentiment": "neutral"}, ...]

Headlines:
"""


# ----------------------------- 1. data -------------------------------
def pip_install(*packages):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *packages], check=True)


def read_public_dataset():
    try:
        return pd.read_csv(DATA_URL)
    except Exception as e:                          # fall back to the datasets library
        print(f"  direct download failed ({str(e)[:60]}); trying the datasets library ...")
        try:
            from datasets import load_dataset
        except ImportError:
            pip_install("datasets")
            from datasets import load_dataset
        return load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation").to_pandas()


def load_sample():
    if CSV_PATH:                                   # your own labelled headlines
        df = pd.read_csv(CSV_PATH)
        df["label"] = df["label"].astype(str).str.lower().str.strip()
        source = os.path.basename(CSV_PATH)
    else:                                          # public human-labelled dataset
        df = read_public_dataset()
        df["label"] = df["label"].map(DATASET_LABELS)
        source = "Twitter Financial News (zeroshot), validation split"
    df["text"] = (df["text"].astype(str)
                  .str.replace(r"https?://\S+", "", regex=True)   # drop links
                  .str.replace(r"\s+", " ", regex=True).str.strip())
    df = df[df["label"].isin(CLASSES) & (df["text"].str.len() > 15)].drop_duplicates("text")
    n = min(N_PER_CLASS, df["label"].value_counts().min())
    sample = (df.groupby("label").sample(n=n, random_state=SEED)
                .sample(frac=1, random_state=SEED).reset_index(drop=True))
    return sample, source


# ----------------------------- 2. model ------------------------------
class MockClient:
    """Offline stand-in for genai.Client, used only to test this script."""
    class _Models:
        def generate_content(self, model, contents, config=None):
            import random
            time.sleep(0.05)
            lines = [l for l in contents.splitlines() if re.match(r"^\d+: ", l)]
            out = [{"id": int(l.split(":")[0]), "sentiment": random.choice(CLASSES)} for l in lines]
            usage = type("U", (), {"total_token_count": 40 * len(lines)})()
            return type("R", (), {"text": json.dumps(out), "usage_metadata": usage, "model_version": "mock"})()

    def __init__(self):
        self.models = self._Models()


def make_client():
    if MOCK:
        return MockClient(), None
    try:
        from google import genai
    except ImportError:                             # install the Gemini SDK on first run
        pip_install("google-genai")
        from google import genai
    from google.genai import types
    key = API_KEY.strip()
    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0)
    return genai.Client(api_key=key), config


def parse_labels(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    data = json.loads(text)
    if isinstance(data, dict):                     # e.g. {"results": [...]}: use the first list inside
        data = next((v for v in data.values() if isinstance(v, list)), [])
    labels = {}
    for item in data:
        if not isinstance(item, dict) or "id" not in item:
            continue
        label = str(item.get("sentiment", "")).lower().strip()
        label = {"bullish": "positive", "bearish": "negative"}.get(label, label)
        if label in CLASSES:
            labels[int(item["id"])] = label
    return labels


def ask_gemini(client, config, model, batch):
    """One request for one batch; retries on rate limits (429) and bad JSON."""
    prompt = PROMPT + "\n".join(f"{i}: {t}" for i, t in batch)
    for attempt in range(5):
        try:
            start = time.perf_counter()
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            seconds = time.perf_counter() - start
            return parse_labels(resp.text), seconds, resp
        except (json.JSONDecodeError, ValueError, TypeError):
            time.sleep(2)                                        # malformed JSON: ask again
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:      # free-tier limit: wait, retry
                wait = 15 * (2 ** attempt)
                print(f"  rate limited, waiting {wait}s ...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gave up after 5 attempts on one batch")


def pick_model(client, config, first_batch):
    for model in MODELS_TO_TRY:
        try:
            return model, ask_gemini(client, config, model, first_batch)
        except Exception as e:
            print(f"  model {model} unavailable ({str(e)[:80]}), trying next ...")
    raise RuntimeError("No Gemini model worked. Check your API key at aistudio.google.com.")


# ----------------------------- 3. run --------------------------------
sample, source = load_sample()
print(f"Loaded {len(sample)} headlines ({sample['label'].value_counts().to_dict()}) from {source}")

client, config = make_client()
items = list(zip(sample.index, sample["text"]))
batches = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]

predictions, seconds_total, tokens_total, model_version = {}, 0.0, 0, None
model, first = pick_model(client, config, batches[0])
for b, batch in enumerate(batches):
    labels, seconds, resp = first if b == 0 else ask_gemini(client, config, model, batch)
    predictions.update(labels)
    seconds_total += seconds
    tokens_total += getattr(resp.usage_metadata, "total_token_count", 0) or 0
    model_version = getattr(resp, "model_version", None) or model
    print(f"  batch {b + 1}/{len(batches)}: {len(labels)}/{len(batch)} labelled in {seconds:.1f}s")
    if b < len(batches) - 1 and not MOCK:
        time.sleep(PAUSE_SECONDS)

sample["gemini"] = [predictions.get(i, "missing") for i in sample.index]
sample.to_csv("gemini_sentiment_results.csv", index=False)

# ----------------------------- 4. scores -----------------------------
y_true, y_pred = sample["label"], sample["gemini"]
n = len(sample)
acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
kappa = cohen_kappa_score(y_true, y_pred)
baseline = y_true.value_counts(normalize=True).max()
cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=CLASSES),
                  index=[f"true {c}" for c in CLASSES], columns=[f"pred {c}" for c in CLASSES])

print("\n========== RESULTS (copy everything below to Claude) ==========")
print(f"Model: {model_version}  (requested: {model})")
print(f"Data: {n} human-labelled headlines, {N_PER_CLASS} per class, seed {SEED} | {source}")
print(f"Accuracy: {acc:.1%} | Macro-F1: {macro_f1:.3f} | Cohen's kappa: {kappa:.3f} | "
      f"Majority baseline: {baseline:.1%}")
print(f"Requests: {len(batches)} batches of up to {BATCH_SIZE} | "
      f"Latency: {seconds_total / len(batches):.2f}s per request, {seconds_total / n:.2f}s per headline")
print(f"Tokens: {tokens_total:,} total ({tokens_total / n:.0f} per headline) | "
      f"Unlabelled: {(y_pred == 'missing').sum()}")
print("Confusion matrix:")
print(cm.to_string())
print("Saved: gemini_sentiment_results.csv")
