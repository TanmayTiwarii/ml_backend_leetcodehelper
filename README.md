# 🧠 UpSolve – Intelligent LeetCode Problem Recommender

UpSolve is an intelligent analytics platform that analyzes your **LeetCode profile** and recommends new or related problems based on your **submission history**, **problem-solving patterns**, and **topic preferences**.

---

## 🚀 Features

- 🔍 **Smart Problem Analysis** – Fetches your recent LeetCode submissions and learns your solving habits.  
- 🤖 **Hybrid Recommendation Engine** – Combines semantic similarity (via cosine similarity on embeddings) with difficulty balancing for meaningful suggestions.  
- ⚡ **FastAPI Backend** – Asynchronous architecture using `httpx` and `FastAPI` for scalable, low-latency responses.  
- 🧩 **Data-Driven Insights** – Precomputed embeddings of 2,000+ LeetCode problems for fast, semantic similarity lookup.  
- 🔄 **Multi-Mode Recommendations**  
  - `/recommend/similar` → Find problems similar to your recent submissions.  
  - `/recommend/diff` → Explore new problems of different difficulty levels.  
- 🌐 **CORS-Enabled API** – Seamlessly connects with a React frontend for an interactive dashboard experience.

---

## 🏗️ System Architecture
```bash
     ┌───────────────────────────┐
     │        Frontend (React)   │
     │  - User enters username   │
     │  - Displays suggestions   │
     └────────────┬──────────────┘
                  │ REST API
                  ▼
     ┌───────────────────────────┐
     │        FastAPI Backend    │
     │  - /recommend/similar     │
     │  - /recommend/diff        │
     │  - /health                │
     └────────────┬──────────────┘
                  │
                  ▼
     ┌───────────────────────────┐
     │  Embeddings + CSV Dataset │
     │  - LeetCode Questions.csv │
     │  - embeddings.npy         │
     └───────────────────────────┘
```

---

## ⚙️ Tech Stack

- **Backend:** FastAPI, httpx, pandas, numpy, scikit-learn  
- **Data:** Preprocessed LeetCode dataset with topics, difficulty, and embeddings  
- **Deployment:** Render / Docker-ready  
- **Frontend (optional):** React + Tailwind (CORS-compatible)

---

## 🧩 Project Structure

```bash
upsolve/
│
├── main.py # FastAPI entrypoint with endpoints
├── routes/
│ ├── recommend_similar.py # Similarity-based recommender
│ └── recommend_diff.py # Difficulty-based recommender
│
├── processed.csv # Cleaned dataset of problems
├── embeddings.npy # Precomputed text embeddings
├── LeetCode Questions.csv # Original dataset
├── requirements.txt
└── .env # Contains BASE_URL for user data fetch
```

---

## 🔧 Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/UpSolve.git
   cd UpSolve
   ```
2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the FastAPI server**
   ```bash
   uvicorn main:app --reload
   ```

## 🧠 Recommendation Logic

- Extracts recent problem IDs from user submission history (via BASE_URL).
- Computes cosine similarity between problem embeddings.
- Filters out solved problems and rebalances results by difficulty.
- Returns top-K ranked problems (configurable).


## 👨‍💻 Author

- **Tanmay Tiwari**
🔗 [LinkedIn](https://www.linkedin.com/in/tanmay-tiwari-30b1212a6/)
