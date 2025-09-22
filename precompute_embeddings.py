import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "LeetCode Questions.csv")

# Load dataset
df = pd.read_csv(csv_path)
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace("\n", "_")
              .str.replace(" ", "_")
)

# Preprocess text features
df["text_features"] = df["difficulty"].astype(str) + " " + df["topics"].astype(str)

# Load lightweight model (local run only)
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

# Generate embeddings
embeddings = model.encode(
    df["text_features"].tolist(),
    normalize_embeddings=True
)
embeddings = np.array(embeddings)

# Save outputs
df.to_csv(os.path.join(BASE_DIR, "processed.csv"), index=False)
np.save(os.path.join(BASE_DIR, "embeddings.npy"), embeddings)

print("✅ Precomputed embeddings and saved processed.csv + embeddings.npy")
