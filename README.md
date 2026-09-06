# SupportPilot – Milestone 1 (Ticket Processing & Classification)

This is a working implementation of Milestone 1 from the SupportPilot deck:
ticket submission, AI-based classification, severity prediction, and
priority assignment — built and trained on the real
`IT_Support_Ticket_Data.csv` dataset (29,650 real support tickets)

## No external AI API is used

This project does **not** call OpenAI, Anthropic, or any external LLM API.
"AI" here means classical machine learning trained locally with
**scikit-learn** (TF-IDF vectorizer + Logistic Regression). The only "API" in the project is the
**REST API that SupportPilot itself exposes** (`POST /api/ticket`). If you later want LLM-based classification (mentioned
as a future upgrade in the deck's intro), that would need an API key and
is a Milestone-2+ concern, not part of this build.

## How to run it

```bash
cd SupportPilot
pip install -r requirements.txt

# 1. Train the models (only needed once, or after changing the data)
python train_model.py

# 2. Start the web app
python app.py
```

Then open **http://127.0.0.1:5000** in a browser.

- `/` — submit a ticket via the web form, see the AI classification, severity, and priority instantly
- `/tickets` — table of all tickets submitted so far
- `POST /api/ticket` — REST endpoint, send JSON `{"description": "..."}` and get back category/severity/priority
- `GET /api/tickets` — JSON list of all stored tickets


## Authentication 

Real login is now enforced — `/`, `/tickets`, `/dashboard` all require a
logged-in session; anyone not logged in is redirected to `/login`.

- **Register**: `/register` — creates a `users` row with a salted password
  hash (`werkzeug.security.generate_password_hash`), never a plain-text password.
- **Login**: `/login` — checks the hash, then stores `user_id`/`username` in
  a signed Flask session cookie.
- **Logout**: `/logout` — clears the session.
- The REST API (`/api/ticket`, `/api/tickets`, `/api/stats`) is **not**
  behind login yet — it's meant for machine-to-machine use. If you need to
  lock that down too, the natural next step is an API key header, not the
  same session-cookie login.

**Before deploying anywhere real:** the app currently falls back to a
hardcoded `SECRET_KEY` if you don't set one. Set a real one as an
environment variable:
```bash
export SECRET_KEY="some-long-random-string"
```
Sessions are only as secure as this key — don't commit a real one to GitHub.

## Milestone 2: Knowledge Retrieval & Resolution Generation (new)

Full RAG pipeline, matching the Milestone 2 deck:

1. **Ticket analysis** — reuses the Milestone 1 classifier's cleaned text as the retrieval query.
2. **Knowledge base retrieval** — `knowledge_base.py` holds 17 real troubleshooting articles across every department the classifier predicts. `KnowledgeRetriever` (TF-IDF + cosine similarity, `ngram_range=(1,2)`) finds the top-3 most relevant articles.
3. **Context augmentation** — `rag_pipeline.py`'s `build_context()` formats retrieved articles with their relevance scores.
4. **Resolution generation** — `generate_resolution()` extracts the actual numbered steps from the retrieved articles (capped at 8 steps) and cites the source KB article per step. Nothing is invented — every line is traceable back to a real article, exactly the point of RAG over asking a model to answer from memory.
5. **Confidence threshold** — articles below `MIN_RELEVANCE` are dropped; if nothing clears the bar, the ticket gets an `INSUFFICIENT_KNOWLEDGE` response instead of a low-quality guess.
6. **Workflow status + metrics** — the UI shows the same 4-stage status the deck's mockup shows, plus three metrics:
   - **Retrieval accuracy** — genuinely measured via `evaluate_retrieval.py` against 31 hand-labeled test queries: **90.3%** (28/31), comparable to the deck's illustrative 92% but actually computed, not hardcoded.
   - **Resolution rate** — real user feedback. Each resolution has "✓ This resolved it" / "✗ Still need help" buttons; the rate is computed from actual responses, not a fixed demo number. Shows "No feedback yet" until at least one ticket gets a response.
   - **Avg. response time** — measured per-request from the actual pipeline execution, averaged across all stored tickets.

See it live at `/ai-agent` after logging in.

**Not implemented (documented as future work, per the deck's own roadmap):** semantic embeddings (deck slide 43-46 flags this as the production upgrade beyond TF-IDF), a real LLM generator (deck slide 40 — the resolution generator here extracts from KB content rather than calling an LLM), and a vector database.

## Authentication (email-based, plus real Google/Facebook sign-in)

Login/registration use **email**, not username:
- **Register**: `/register` — email + password (name optional). Passwords are salted+hashed (`werkzeug.security`), never stored in plain text.
- **Login**: `/login` — email + password.
- **Google / Facebook sign-in** — real OAuth via [Authlib](https://docs.authlib.org/), not a fake button. See setup below.
- All ticket pages (`/`, `/tickets`, `/dashboard`, `/ai-agent`) require login; unauthenticated visits redirect to `/login`.
- The REST API (`/api/ticket`, `/api/tickets`, `/api/stats`) is not behind login — it's for machine-to-machine use.
- The ticket form no longer asks for your email — it uses the email from your logged-in session automatically.

### Setting up real Google sign-in

Google requires you to register your own app — I can't generate these credentials for you, they're tied to your Google account.

1. Go to [console.cloud.google.com](https://console.cloud.google.com) → create a project (or use an existing one).
2. Go to **APIs & Services → OAuth consent screen** → set it up as "External" → add your email as a test user if it stays in testing mode.
3. Go to **APIs & Services → Credentials** → **Create Credentials → OAuth client ID** → Application type: **Web application**.
4. Under **Authorized redirect URIs**, add exactly: `http://127.0.0.1:5000/auth/google/callback` (and `http://localhost:5000/auth/google/callback` too, to be safe).
5. Copy the **Client ID** and **Client Secret** it gives you.
6. Set them as environment variables before running the app:
   ```bash
   export GOOGLE_CLIENT_ID="your-client-id.apps.googleusercontent.com"
   export GOOGLE_CLIENT_SECRET="your-client-secret"
   python app.py
   ```
7. Restart the app. The "Continue with Google" button will now appear on `/login` and `/register` automatically — it's hidden until these variables are set, rather than showing a broken button.

### Setting up real Facebook sign-in

1. Go to [developers.facebook.com](https://developers.facebook.com) → **My Apps → Create App** → choose "Consumer" or "Other" → give it a name.
2. Add the **Facebook Login** product to the app.
3. Under Facebook Login → Settings, add this to **Valid OAuth Redirect URIs**: `http://127.0.0.1:5000/auth/facebook/callback`
4. Copy the **App ID** and **App Secret** from Settings → Basic.
5. Set them as environment variables:
   ```bash
   export FACEBOOK_CLIENT_ID="your-app-id"
   export FACEBOOK_CLIENT_SECRET="your-app-secret"
   python app.py
   ```
6. **Important Facebook-specific limitation:** while your Facebook app is in "Development" mode (the default for a new app), only accounts you've added as *Test Users* or *Developers/Admins* on the app can actually log in. Making it work for arbitrary users requires Facebook's App Review process. For a college project demo, log in with your own Facebook account (as the app owner, you can always use it) or add a test user under **App Roles → Test Users**.

### What happens without any of this configured

If you don't set any of the four environment variables above, the app works exactly as before: email + password only, no broken buttons, nothing crashes. This is intentional — I didn't want your submission to break if you demo it on a machine without these set up.

**Before deploying anywhere real**, also set a real secret key instead of the dev fallback:
```bash
export SECRET_KEY="some-long-random-string"
```

## Project structure

```
SupportPilot/
├── app.py                  # Flask web app + REST API
├── train_model.py          # Trains category & severity ML models on real data
├── classifier.py           # Pre-processing, prediction, priority logic
├── database.py             # SQLite tickets table
├── requirements.txt
├── evaluation_report.json  # Auto-generated accuracy report (see below)
├── data/
│   └── IT_Support_Ticket_Data.csv
├── models/                 # Saved .pkl models (created by train_model.py)
├── templates/
│   └── index.html
└── tickets.db               # SQLite database (created on first run)
```

## Evaluation results — read this before you submit

The deck's slides 51–53 quote **90% classification accuracy / 85% severity
accuracy** — but slide 53 itself says in fine print: *"these 92%/88%
figures are illustrative examples, not actual results from your
project... your final report should use the results obtained from your
test dataset."* That's exactly what happened here.

On the real 29,650-ticket dataset (held-out 20% test split):

| Metric | Target | Actual (this run) |
|---|---|---|
| Category classification accuracy | ≥ 90% | **67.8%** |
| Severity prediction accuracy | ≥ 85% | **70.1%** |

**Why it's below the illustrative target, and why that's a normal, reportable
finding rather than a bug:**
- The dataset has **10 overlapping department labels** (e.g. "Technical
  Support" vs "IT Support" vs "Product Support" vs "Customer Service") that
  genuinely describe similar issues — even a human would mislabel some of
  these consistently.
- Real support emails are long, share a lot of generic boilerplate ("Dear
  Support Team... Thank you..."), and the actual signal (the real problem)
  is a small fraction of the text. The code already strips this boilerplate
  before vectorizing, which is what took accuracy from ~56% to ~68%.
- The toy 10-row example in the deck (2 rows per category, all short and
  unambiguous) is trivially separable — that's why it can imply high
  accuracy. It is not representative of a real 29K-row dataset.

**What you can say in your submission:** you trained on the real dataset
instead of a toy one, measured genuine held-out accuracy, and can explain
*why* it differs from the deck's illustrative numbers — that's a stronger,
more credible Milestone 1 report than simply hitting a target number.

**If you want to try pushing accuracy higher** before submitting, options
that are reasonable next steps (not already applied here): merging
near-duplicate departments (e.g. combine "Technical Support" + "IT
Support" + "Product Support" into one class), using the `Tags` column as
additional model input, or trying a stronger model (e.g. linear SVM, or a
small transformer). None of these are required for a legitimate Milestone 1
submission — the current pipeline is fully functional end-to-end.
