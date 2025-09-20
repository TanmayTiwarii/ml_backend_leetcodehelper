# from fastapi import FastAPI, Query
# from typing import List
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer

# # -----------------------
# # Load & preprocess dataset once at startup
# # -----------------------
# def load_dataset(path):
#     df = pd.read_csv(path)
#     df.columns = (
#         df.columns.str.strip()
#                   .str.lower()
#                   .str.replace("\n", "_")
#                   .str.replace(" ", "_")
#     )
#     return df

# def preprocess_dataset(df):
#     df["text_features"] = df["difficulty"].astype(str) + " " + df["topics"].astype(str)
#     return df

# path = "LeetCode Questions.csv"   # put your dataset here
# df = preprocess_dataset(load_dataset(path))

# # -----------------------
# # Sentence-BERT model & embeddings (precompute once!)
# # -----------------------
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # Precompute embeddings for all problems
# embeddings = model.encode(
#     df["text_features"].tolist(),
#     normalize_embeddings=True
# )
# embeddings = np.array(embeddings)

# # -----------------------
# # Recommendation helpers
# # -----------------------
# def recommend(df, last_20_ids, top_k=5):
#     # indices of last_20_ids in df
#     mask = df["id"].isin(last_20_ids).values
#     last_vecs = embeddings[mask]

#     # mean similarity against all problems
#     sim_scores = cosine_similarity(last_vecs, embeddings).mean(axis=0)

#     sim_df = df.copy()
#     sim_df["similarity"] = sim_scores
#     sim_df = sim_df[~sim_df["id"].isin(last_20_ids)]  # exclude already solved

#     return sim_df.sort_values("similarity", ascending=False).head(top_k)

# def hybrid_recommend(df, last_20_ids, top_k=5):
#     last_20 = df[df["id"].isin(last_20_ids)]
#     sim_df = recommend(df, last_20_ids, top_k=len(df))

#     counts = last_20["difficulty"].value_counts(normalize=True)
#     target_mix = {
#         "Easy": counts.get("Easy", 0.3),
#         "Medium": counts.get("Medium", 0.4),
#         "Hard": counts.get("Hard", 0.3),
#     }

#     recs = []
#     for diff, frac in target_mix.items():
#         n_pick = max(1, int(top_k * frac))
#         pool = sim_df[sim_df["difficulty"] == diff]
#         if len(pool) > 0:
#             recs.append(pool.head(n_pick))

#     if recs:
#         out = pd.concat(recs)
#         # if fewer than top_k, fill with remaining from sim_df
#         if len(out) < top_k:
#             extra = sim_df[~sim_df["id"].isin(out["id"])].head(top_k - len(out))
#             out = pd.concat([out, extra])
#         return out.head(top_k)
#     else:
#         return sim_df.head(top_k)

# # -----------------------
# # FastAPI app
# # -----------------------
# app = FastAPI()

# @app.get("/recommend")
# def get_recommendations(last_ids: List[int] = Query(..., description="List of last problem IDs")):
#     if len(last_ids) < 1:
#         return {"error": "You must provide at least 1 problem id"}
    
#     recs = hybrid_recommend(df, last_ids, top_k=5)
#     recs["similarity"] = recs["similarity"].round(4)  # clean scores
#     return recs[["id", "problem_name", "difficulty", "topics", "similarity"]].to_dict(orient="records")

from fastapi import FastAPI, Query
from typing import List
from routes.recommend_similar import df as df_sim, hybrid_recommend
from routes.recommend_diff import df as df_diff, recommend_diff


app = FastAPI()

@app.get("/recommend/similar")
def get_similar(last_ids: List[int] = Query(..., description="List of last problem IDs")):
    if len(last_ids) < 1:
        return {"error": "You must provide at least 1 problem id"}
    
    recs = hybrid_recommend(df_sim, last_ids, top_k=5)
    recs["similarity"] = recs["similarity"].round(4)
    return recs[["id", "problem_name", "difficulty", "topics", "similarity"]].to_dict(orient="records")

@app.get("/recommend/diff")
def get_diff(last_ids: List[int] = Query(..., description="List of last problem IDs")):
    if len(last_ids) < 1:
        return {"error": "You must provide at least 1 problem id"}
    
    recs = recommend_diff(df_diff, last_ids, top_k=5)
    return recs[["id", "problem_name", "difficulty", "topics"]].to_dict(orient="records")
