# Deploy (free) and get a public link

## Option A — Streamlit Community Cloud (easiest, recommended)
1. Create a GitHub account and a **public** repo (e.g. `enterprise-intelligence-copilot`).
2. Push this folder:
   ```bash
   git init && git add . && git commit -m "Enterprise Intelligence Copilot"
   git branch -M main
   git remote add origin https://github.com/<you>/enterprise-intelligence-copilot.git
   git push -u origin main
   ```
   `data/olist.db.gz` is ~12 MB (under GitHub's 100 MB limit; GitHub may show a size notice, that is fine).
3. Go to https://share.streamlit.io → sign in with GitHub → **Create app** →
   repo = yours, branch = `main`, main file = `ui/streamlit_app.py`.
4. Deploy. First boot takes a few minutes. You get a public URL like `https://<name>.streamlit.app`.
5. Put the URL at the top of your README, in your LinkedIn Featured section and on your resume.

## Option B — Hugging Face Spaces (Docker)
New Space → SDK **Docker** → upload the repo. In the Space's `README.md` add the front-matter
`sdk: docker` and `app_port: 7860`. The included `Dockerfile` already listens on 7860.

## Optional: turn on an LLM for better answers
Streamlit Cloud → app → Settings → **Secrets**:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
LLM_DAILY_CAP = "100"   # hard cap on LLM calls per day so a public link cannot run up a bill
```
The app copies Streamlit secrets into environment variables automatically.
**Never commit the key.** The LLM path is implemented but I could not test it against the live API from my
sandbox — test it yourself and measure faithfulness before claiming anything about it.

## Before you share the link
- Open it in a private window and click every example question.
- The app sleeps on free tiers after inactivity; open it yourself right before interviews.
- Keep the data attribution in the sidebar (Olist is CC BY-NC-SA 4.0; verify on Kaggle).
