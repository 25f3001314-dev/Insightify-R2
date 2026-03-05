# Insightify-R2

> **Customer Upgrade Growth Command Center** — a production-ready Streamlit dashboard that turns raw customer behavior data into actionable campaign decisions: who to target, which offer to send, and what ROI to expect.

---

## Project Structure

```
Insightify-R2/
├── dashboard_app.py            # Main Streamlit dashboard application
├── Insightify_R2.ipynb         # ML notebook (training, evaluation, artifact export)
├── cleaned_customer_data.csv   # Feature-ready customer dataset
├── models/
│   ├── model.pkl               # Trained RandomForest classifier
│   ├── feature_columns.pkl     # Ordered feature column list (26 features)
│   ├── feature_medians.pkl     # Median fallback values per feature
│   └── model_metadata.json     # Model version, metrics, timestamp
├── requirements.txt            # Python dependencies
├── .streamlit/config.toml      # Streamlit server config (headless, port binding)
├── start_dashboard.sh          # Auto free-port launcher script
├── render.yaml                 # Render.com deployment manifest
├── .env.example                # Environment variable template
├── .gitignore                  # Security: excludes .env, secrets, __pycache__
└── README.md                   # This file
```

---

## How It Works

### 1. Data & Model Pipeline

The ML notebook (`Insightify_R2.ipynb`) loads `cleaned_customer_data.csv`, engineers features (spending score, purchase frequency, order value, weekend ratio, discount usage, complaint flags, etc.), trains a **RandomForest classifier** to predict membership upgrade probability, and exports four artifacts into `models/`.

The dashboard **never retrains at runtime** — it loads pre-built artifacts for instant, reproducible predictions.

### 2. Dashboard Modules

When you launch the dashboard, it scores every customer in real time and powers these modules:

| Module | What It Does |
|---|---|
| **KPI Cards** | Total rows, columns, labeled rows, and overall upgrade rate at a glance. |
| **Behavior Overview** | Interactive Plotly charts — spending score distribution, purchase frequency by upgrade status, order value vs weekend ratio scatter. |
| **Model Info** | Displays model name, version, holdout accuracy, and PR-AUC from saved metadata. |
| **Live Upgrade Prediction** | Form-based single-customer scorer — enter spending, frequency, order value, etc. and get instant upgrade probability with confidence band (High / Medium / Low). |
| **Membership Tier Opportunity** | Groups customers by tier (Basic → Platinum), shows average propensity, order value, and purchase frequency per tier with a bar chart. |
| **Targeting and Offer Planner** | The core campaign engine. Segments customers into **Premium Upsell**, **Value Seekers**, and **Service Recovery** buckets. For each segment it shows: target audience size, recommended offer, primary channel, expected upgrades, projected revenue lift, campaign cost, and **expected ROI**. Configurable planning horizon (7–60 days) and budget cap. Includes a **Download Targeting Plan DF (.csv)** button — one row per segment, ready for marketing ops spreadsheets. |
| **Offer Impact Simulator** | What-if model: adjust discount %, free delivery days, bonus points, priority support, and propensity range. Instantly see expected upgrades, revenue lift, and estimated ROI. |
| **Next-Best-Action Engine** | Assigns every customer to an action segment (Upsell Now / Nurture Campaign / Rescue & Recover / Low Priority) with a recommended offer. Shows segment split pie chart and a sortable top-25 customer preview table. |
| **AI Marketing Summary (Mistral)** | Connects to Mistral API for on-demand AI-generated summaries. Choose from Executive Summary, Campaign Plan, Persona Deep Dive, or Custom Q&A templates. Download results as **markdown (.md)** or **structured dataframe (.csv)**. |

### 3. Scoring Flow

```
CSV data → load_data() → score_customers()
                              ↓
                    model.predict_proba()  ← model.pkl + feature_columns.pkl + feature_medians.pkl
                              ↓
                   Upgrade_Probability column added to every row
                              ↓
              All modules consume scored dataframe
```

### 4. Export Options

- **Targeting Plan CSV** — one row per campaign segment with target size, offer, channel, cost, ROI.
- **AI Summary Markdown** — formatted report with model snapshot and AI-generated insights.
- **AI Summary CSV** — structured single-row dataframe with all context + summary text.

---

## Quick Start

### Run the Dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard_app.py
```

Or use the auto free-port launcher (recommended in dev containers):

```bash
bash start_dashboard.sh
```

### Run the Notebook

Open `Insightify_R2.ipynb` and run all cells in order to retrain or inspect the model.

---

## Mistral API Setup (AI Summary)

Do not hardcode API keys. Use one of these methods:

**Option A: Environment variable**
```bash
export MISTRAL_API_KEY="your_api_key_here"
streamlit run dashboard_app.py
```

**Option B: Streamlit secrets**

Create `.streamlit/secrets.toml`:
```toml
MISTRAL_API_KEY = "your_api_key_here"
```

In the dashboard:
- Click **Generate AI Summary** for on-demand output.
- Toggle **Auto AI Summary** in the sidebar for automatic generation on each rerun.
- Use download buttons for `.md` or `.csv` exports.

---

## Deployment

### Streamlit Community Cloud (Recommended)

1. Push this repo to GitHub.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and create an app from this repo.
3. Set main file path: `dashboard_app.py`.
4. Add secret: `MISTRAL_API_KEY = "your_api_key_here"` in app settings.
5. Deploy.

### Render

Use a Web Service with start command:

```bash
streamlit run dashboard_app.py --server.port $PORT --server.address 0.0.0.0
```

A `render.yaml` manifest is included in the repo.

### Vercel

Vercel is optimized for Next.js/static sites. For Streamlit apps, Streamlit Cloud or Render is the better fit.

---

## Troubleshooting

### Port Forwarding Issues

If the app runs but VS Code shows a forwarding error:

```bash
# Check busy ports
lsof -i :8501

# Start on explicit free port
streamlit run dashboard_app.py --server.address 0.0.0.0 --server.port 8502 --server.headless true

# Kill stale process if needed
pkill -f "streamlit run dashboard_app.py"
```

Then forward port `8502` from the VS Code Ports panel.

---

## License

See [LICENSE](LICENSE) for details.
