import pandas as pd
from sentence_transformers import SentenceTransformer
import os

# -----------------------
# Load & preprocess dataset once at startup
# -----------------------
def load_dataset(path):
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace("\n", "_")
                  .str.replace(" ", "_")
    )
    return df

def preprocess_dataset(df):
    df["text_features"] = df["difficulty"].astype(str) + " " + df["topics"].astype(str)
    return df

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(BASE_DIR, "LeetCode Questions.csv")
df = preprocess_dataset(load_dataset(path))

# Load model (not really needed for "diff", but kept for consistency)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------
# Different recommendation (just pick 5 random unseen problems)
# -----------------------
def recommend_diff(df, last_20_ids, top_k=5):
    unseen = df[~df["id"].isin(last_20_ids)]
    recs = unseen.sample(n=min(top_k, len(unseen)), random_state=42)
    recs["similarity"] = 0.0  # placeholder since no similarity calc
    return recs
