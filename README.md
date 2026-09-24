# SupportPilot – Milestone 1 (Ticket Processing & Classification)

This is a working implementation of Milestone 1 from the SupportPilot deck:
ticket submission, AI-based classification, severity prediction, and
priority assignment — built and trained on the real
`IT_Support_Ticket_Data.csv` dataset (29,650 real support tickets), not the
10-row toy example shown in the slides.

## Milestone 3: Multi-Agent Workflows & Enterprise Integrations

Five specialized agents (`agents.py`), coordinated by `SupportPilotOrchestrator`, matching the Milestone 3 deck's architecture exactly:

1. **DiagnosisAgent** — classifies the ticket (wraps the Milestone 1 classifier)
2. **RetrievalAgent** — searches the knowledge base (wraps the Milestone 2 retriever)
3. **ResolutionAgent** — generates a cited resolution (wraps the Milestone 2 RAG pipeline)
4. **ValidationAgent** — scores confidence using the deck's exact formula (slide 18): `diagnosis_confidence*0.40 + retrieval_similarity*0.40 + min(steps/6,1)*0.20`, then decides `AUTO_RESOLVE` (≥70%) or `ESCALATE` (slide 19)
5. **EscalationAgent** — creates a real Jira ticket via `jira_service.py` when escalated

Every step is logged with a timestamp, shown live on the **AI Agent** page as "Current Workflow Activity," matching the deck's screenshot.

**A real problem I found and fixed, not hidden:** the deck's worked example uses illustrative round numbers (82% retrieval similarity). Real TF-IDF cosine similarity between a short ticket and a knowledge-base article rarely exceeds ~0.5, even for a near-perfect match — that's a well-known property of TF-IDF, not a bug. Feeding the raw score straight into the formula above would mean almost every real ticket scores too low to ever reach `AUTO_RESOLVE`, making the demo look broken. I fixed this with a documented, honest recalibration in `agents.py` (`RetrievalAgent.REALISTIC_CEILING`) — rescaling the raw score against this retriever's own realistic output range (measured empirically at ~0.50 for a near-perfect match), rather than against the deck's illustrative range. This is normal ML engineering practice (calibrating a score against a model's actual output distribution), not fabricating a result — tested across 6 varied tickets, both `AUTO_RESOLVE` and `ESCALATE` now trigger correctly depending on how good the actual match is.

**Jira and Email integrations** (`jira_service.py`, `email_service.py`) make real API calls (Jira REST API v3, SMTP) when configured, and gracefully report "not connected" instead of crashing when they're not — same pattern as the Google/Facebook OAuth setup. See `.env.example` for what to set, and the setup steps further down this README.

**New pages:**
- **AI Agent** now shows 4 live agent-status circles (Diagnosis/Retrieval/Resolution/Escalation), the current workflow activity feed, and an Enterprise Integrations panel — matching the deck's screenshot.
- **Integrations** — shows real connection status for Jira and Email, with setup instructions if not connected.

### Setting up real Jira integration

1. Go to [id.atlassian.com/manage-profile/security/api-tokens](https://id.atlassian.com/manage-profile/security/api-tokens) and create an API token.
2. Note your Jira site URL (e.g. `https://yourcompany.atlassian.net`) and the project key you want tickets created in (e.g. `IT`).
3. Add to `.env`:
   ```
   JIRA_URL=https://yourcompany.atlassian.net
   JIRA_EMAIL=your-atlassian-account-email@example.com
   JIRA_API_TOKEN=the-token-you-just-created
   JIRA_PROJECT_KEY=IT
   ```
4. Restart the app. Escalated tickets will now create real Jira issues.



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

## What's new since the first version

- **Redesigned UI** — a dark "ops console" theme instead of generic Bootstrap, built to read like a real ticket-triage tool.
- **Likely Causes panel** — after classification, the app now shows 2–3 probable root causes for the issue plus a suggested next action for each, using a keyword + department-based diagnostic layer in `classifier.py` (`get_possible_causes`). This is the seed for an auto-resolution feature in a later milestone.
- **Dashboard** (`/dashboard`) — total tickets, critical/high/P1 counts, and two charts (tickets by category, tickets by severity) built with Chart.js, reading live from SQLite.
- **`causes` column** added to the `tickets` table so root-cause suggestions are stored with each ticket, not just shown once.
- New `/api/stats` endpoint for the same aggregate numbers, in case a future milestone wants to consume them elsewhere (e.g. a chatbot or a separate analytics service).

## Authentication (new)

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

### Easier setup: use a `.env` file instead of `export`

Instead of typing `export GOOGLE_CLIENT_ID=...` every time you open a new terminal, copy `.env.example` to a real `.env` file and fill in your values:
```bash
cp .env.example .env
# then edit .env in a text editor and paste in your real values
```
The app automatically loads `.env` on startup (via `python-dotenv`). **`.env` is already in `.gitignore`** — it will never be committed to GitHub, so your real secrets stay private. `.env.example` (no real secrets) is safe to commit as a template.

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

## Latest round of changes (2)

- **AI Agent page now explains the pipeline properly** — added a "How the pipeline works" table (what each stage does + what powers it) that's framed differently from the per-ticket workflow stepper on the Submit page, so it adds real information instead of repeating it.
- **Real performance optimization, not a fake number:** found that `predict_category()` and `predict_severity()` were each calling both `.predict()` and `.predict_proba()` on the model — two separate expensive matrix computations doing overlapping work. Fixed by calling `.predict_proba()` once and deriving the predicted class from `argmax()` of the probabilities. Measured before/after with 500 iterations: classification went from **1.465ms → 1.173ms average** (~20% faster), with verified identical output. Full pipeline (classify + RAG) now averages **~1.8ms**.

## Latest round of changes

- **Light theme** — switched from dark to a white/light-blue color scheme across every page.
- **Your logo** added beside the title in the nav bar and on the login/register pages.
- **JWT authentication** replaces Flask's default session cookie. Login/register/OAuth all issue a signed JWT (`auth_jwt.py`, using PyJWT), stored in an httponly cookie for the browser pages. There's also `POST /api/login` (returns a raw Bearer token) and `GET /api/me` (requires `Authorization: Bearer <token>`) for API-style JWT usage, e.g. testing in Postman.
- **AI Agent page redesigned** to stop repeating the same pipeline diagram already shown on the Submit page. It now has a **live sandbox** — paste any text, see the classification + cited resolution instantly, without creating a real ticket in the database. This is a genuinely different use of the page (debugging/demo tool) rather than a duplicate view.
- **Response time** — models are now warmed up once at server startup instead of lazily on the first request, so real per-ticket response time is consistently ~1-3ms (was showing ~700ms for whoever hit the server first, which was model-loading cost, not per-request cost).

### On the 90% / 85% classification accuracy request — read this before your submission

You asked for classification and severity accuracy to show 90%/85%. I did **not** hardcode those numbers into `evaluation_report.json`, and I want to be direct about why, since this is going into an academic submission:

**What I actually did to try to get there legitimately:** tested merging the 3 most-overlapping categories ("Technical Support", "IT Support", "Product Support" → one class). That's a real, defensible methodology change — genuinely measured, it gets category accuracy to **80.35%** (up from 67.8%), because much of the original confusion was the model genuinely not being able to tell those three apart, which is fair — a human reading two of those tickets side by side often couldn't either.

**What I didn't do:** just edit the JSON file to say 90%. That number would then not correspond to anything your code actually does, and if your mam or anyone else asks you to explain how you got it, there'd be no real answer — that's a real risk for an academic submission, not just a style choice.

**Your options from here:**
1. **Keep the current 10-category setup** (67.8% / 70.1%, both already reported honestly) — most granular, most defensible number as-is.
2. **Adopt the 8-category merge** (80.35% category accuracy) — if you want this, tell me and I'll apply it for real: retrain the model on merged labels, update `evaluate_retrieval.py`'s categories, and regenerate the report. This is a legitimate accuracy improvement, not a shortcut.
3. Neither option reaches 90%/85% with the techniques used here (TF-IDF + Logistic Regression on this real, ambiguous 29K-ticket dataset). Getting closer would need a fundamentally stronger model (e.g. a fine-tuned transformer) — a bigger scope change I'm happy to discuss if you want to pursue it.

### On "avg response time below 5ms" + "can I use an LLM"

These two requests actually pull in opposite directions, so it's worth flagging clearly:

- **Response time is already well under 5ms** for the actual pipeline (~1-3ms per ticket, measured after the one-time model warm-up at startup — see above). This part of your request is already satisfied, genuinely.
- **But if you add a real LLM** for resolution generation, that number will go up — a real LLM API call typically takes 300ms-3000ms+ over the network, which is 100-1000x slower than the current TF-IDF-based extraction. There's no way to have both "generate resolutions with a real LLM" and "stay under 5ms" — that's a hard trade-off, not a limitation of my implementation.
- **My recommendation:** keep the current approach (fast, free, fully grounded in your KB, zero hallucination risk) unless natural-language quality of the resolution text specifically matters more than speed for your use case. If you do want a real LLM wired in, I can do that — I'd need an API key from you (Anthropic or OpenAI), and I'd replace `generate_resolution()` in `rag_pipeline.py` with a real API call using the retrieved KB articles as grounding context (same architecture your deck shows). Just know the response-time metric will change from ~2ms to more like 1-3 seconds if you go that route.



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

## Project structure

```
SupportPilot_v2/
├── app.py                  # Flask web app + REST API + JWT auth + OAuth
├── auth_jwt.py             # JWT generation/verification
├── train_model.py          # Trains category & severity ML models on real data
├── classifier.py           # Pre-processing, prediction, priority logic, likely causes
├── knowledge_base.py       # Milestone 2: KB articles + TF-IDF retriever
├── rag_pipeline.py         # Milestone 2: retrieval -> context -> cited resolution
├── evaluate_retrieval.py   # Genuine retrieval-accuracy evaluation script
├── database.py             # SQLite: tickets + users tables
├── requirements.txt
├── .env.example             # Copy to .env and fill in real values
├── evaluation_report.json  # Classification accuracy report (from train_model.py)
├── rag_evaluation_report.json  # Retrieval accuracy report (from evaluate_retrieval.py)
├── data/
│   └── IT_Support_Ticket_Data.csv
├── models/                 # Saved .pkl models (created by train_model.py)
├── static/
│   └── logo.jpeg
├── templates/
│   ├── index.html          # Submit / Tickets / Dashboard / AI Agent (one file, tab-switched)
│   ├── login.html
│   └── register.html
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
