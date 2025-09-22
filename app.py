from fastapi import FastAPI, Query
from typing import List
import httpx
import os
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()

BASE_URL = os.getenv("BASE_URL")

# globals to be initialized at startup
df_sim = None
df_diff = None
hybrid_recommend = None
recommend_diff = None


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


@app.on_event("startup")
async def load_resources():
    """
    Import heavy modules and load data after FastAPI starts,
    so Render detects the open port quickly.
    """
    global df_sim, df_diff, hybrid_recommend, recommend_diff
    from routes import recommend_similar, recommend_diff as diff_module

    df_sim = recommend_similar.df
    hybrid_recommend = recommend_similar.hybrid_recommend
    df_diff = diff_module.df
    recommend_diff = diff_module.recommend_diff

    print("DEBUG: Resources loaded at startup ✅")


@app.get("/recommend/similar")
async def get_similar(username: str = Query(..., description="LeetCode username")):
    if not hybrid_recommend or df_sim is None:
        return {"error": "Resources not ready yet, please try again shortly."}

    last_ids = await fetch_recent_problems(username)
    if not last_ids:
        return {"error": "Could not fetch recent problems or user has no submissions"}

    recs = hybrid_recommend(df_sim, last_ids, top_k=5)
    if "similarity" in recs.columns:
        recs["similarity"] = recs["similarity"].round(4)
        return recs[["id", "problem_name", "difficulty", "topics", "similarity"]].to_dict(orient="records")
    return []


@app.get("/recommend/diff")
async def get_diff(username: str = Query(..., description="LeetCode username")):
    if not recommend_diff or df_diff is None:
        return {"error": "Resources not ready yet, please try again shortly."}

    last_ids = await fetch_recent_problems(username)
    if not last_ids:
        return {"error": "Could not fetch recent problems or user has no submissions"}

    recs = recommend_diff(df_diff, last_ids, top_k=5)
    return recs[["id", "problem_name", "difficulty", "topics"]].to_dict(orient="records")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
