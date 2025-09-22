from fastapi import FastAPI, Query
from typing import List
import httpx
import os
from routes.recommend_similar import df as df_sim, hybrid_recommend
from routes.recommend_diff import df as df_diff, recommend_diff
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

BASE_URL = os.getenv("BASE_URL")

async def fetch_recent_problems(username: str) -> List[int]:
    url = f"{BASE_URL}/{username}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()  # already a list of IDs
            return []
    except httpx.ReadTimeout:
        print(f"Timeout when fetching recent problems for {username} from {url}")
        return []
    except Exception as e:
        print(f"Error fetching recent problems for {username}: {e}")
        return []


@app.get("/recommend/similar")
async def get_similar(username: str = Query(..., description="LeetCode username")):
    last_ids = await fetch_recent_problems(username)
    print(f"DEBUG: last_ids for {username}: {last_ids}")
    if not last_ids:
        print("DEBUG: No recent problems found or user has no submissions.")
        return {"error": "Could not fetch recent problems or user has no submissions"}

    recs = hybrid_recommend(df_sim, last_ids, top_k=5)
    print(f"DEBUG: recs DataFrame columns: {recs.columns.tolist()}")
    print(f"DEBUG: recs DataFrame shape: {recs.shape}")
    if "similarity" in recs.columns:
        print(f"DEBUG: similarity values: {recs['similarity'].values}")
        recs["similarity"] = recs["similarity"].round(4)
        return recs[["id", "problem_name", "difficulty", "topics", "similarity"]].to_dict(orient="records")
    else:
        print("DEBUG: No similarity column in recs DataFrame.")
        return []


@app.get("/recommend/diff")
async def get_diff(username: str = Query(..., description="LeetCode username")):
    last_ids = await fetch_recent_problems(username)
    if not last_ids:
        return {"error": "Could not fetch recent problems or user has no submissions"}
    
    recs = recommend_diff(df_diff, last_ids, top_k=5)
    return recs[["id", "problem_name", "difficulty", "topics"]].to_dict(orient="records")

@app.get("/health")
async def health_check():
    return {"status": "ok"}
