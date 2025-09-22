import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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

# -----------------------
# Smaller Sentence-BERT model (lighter for Render 512MB)
# -----------------------
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

# Encode using float16 to save memory
embeddings = model.encode(
    df["text_features"].tolist(),
    normalize_embeddings=True,
    convert_to_numpy=True,
    dtype=np.float16
)

embeddings = np.array(embeddings, dtype=np.float16)

# -----------------------
# Recommendation helpers
# -----------------------
def recommend(df, last_20_ids, top_k=5):
    id_type = df["id"].dtype
    last_20_ids = [id_type.type(x) if hasattr(id_type, 'type') else id_type(x) for x in last_20_ids]
    mask = df["id"].isin(last_20_ids).values
    last_vecs = embeddings[mask]

    if last_vecs.size == 0 or embeddings.size == 0:
        print("Warning: last_vecs or embeddings is empty. Returning empty recommendations.")
        return df.head(0)
    if last_vecs.ndim != 2 or embeddings.ndim != 2:
        print(f"Warning: invalid shape. last_vecs: {last_vecs.shape}, embeddings: {embeddings.shape}")
        return df.head(0)

    sim_scores = cosine_similarity(last_vecs, embeddings).mean(axis=0)

    sim_df = df.copy()
    sim_df["similarity"] = sim_scores
    sim_df = sim_df[~sim_df["id"].isin(last_20_ids)]

    return sim_df.sort_values("similarity", ascending=False).head(top_k)

def hybrid_recommend(df, last_20_ids, top_k=5):
    last_20 = df[df["id"].isin(last_20_ids)]
    sim_df = recommend(df, last_20_ids, top_k=len(df))

    counts = last_20["difficulty"].value_counts(normalize=True)
    target_mix = {
        "Easy": counts.get("Easy", 0.3),
        "Medium": counts.get("Medium", 0.4),
        "Hard": counts.get("Hard", 0.3),
    }

    recs = []
    for diff, frac in target_mix.items():
        n_pick = max(1, int(top_k * frac))
        pool = sim_df[sim_df["difficulty"] == diff]
        if len(pool) > 0:
            recs.append(pool.head(n_pick))

    if recs:
        out = pd.concat(recs)
        if len(out) < top_k:
            extra = sim_df[~sim_df["id"].isin(out["id"])].head(top_k - len(out))
            out = pd.concat([out, extra])
        return out.head(top_k)
    else:
        return sim_df.head(top_k)
