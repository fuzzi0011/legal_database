# Pakistan Legal AI — Case Law Research System

Search and analyze judgments from SHC, LHC, and IHC using free AI (HuggingFace Mistral-7B).

---

## Setup in 4 Steps

### Step 1 — Get a Free HuggingFace Token

1. Go to https://huggingface.co and create a free account
2. Click your profile → Settings → Access Tokens
3. Click "New Token" → name it anything → Role: Read → Generate
4. Copy the token (starts with `hf_...`)

### Step 2 — Set the token

**Windows:**
```
set HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

**Mac/Linux:**
```bash
export HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

### Step 3 — Install and run

```bash
pip install -r requirements.txt
python -m api.server
```

Server starts at http://localhost:8000
Automatically loads sample Pakistani cases on first run.

### Step 4 — Open the UI

Open `frontend/index.html` in your browser.

---

## How to Scrape Real Cases

```bash
# Scrape all three courts
python scrapers/scraper.py

# Scrape by keyword
python scrapers/scraper.py "NHA road accident"
```

Or use the "Scrape Courts" button in the sidebar of the UI.

---

## Free Hosting (deploy online)

| Part         | Service       | Cost    |
|--------------|---------------|---------|
| Frontend     | GitHub Pages  | Free    |
| Backend API  | Railway.app   | Free    |
| AI (LLM)     | HuggingFace   | Free    |
| Database     | Supabase      | Free    |

### Deploy Backend to Railway
1. Push project to GitHub
2. Go to railway.app → New Project → Deploy from GitHub
3. Add environment variable: `HF_API_TOKEN = hf_xxx...`
4. Railway gives you a public URL automatically

### Deploy Frontend to GitHub Pages
1. Upload `frontend/index.html` to a GitHub repo
2. Settings → Pages → Deploy from main branch
3. Update `const API = 'https://your-app.railway.app'` in index.html

---

## Files

```
pakistan-legal-ai/
├── api/
│   ├── server.py          FastAPI backend
│   └── llm_engine.py      HuggingFace Mistral-7B
├── embeddings/
│   └── vector_db.py       TF-IDF search engine (offline)
├── scrapers/
│   └── scraper.py         SHC + LHC + IHC scrapers
├── frontend/
│   └── index.html         Web UI
└── requirements.txt
```

---

## LLM Used

**Mistral-7B-Instruct-v0.3** — free on HuggingFace, strong legal reasoning in English.
Model page: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
