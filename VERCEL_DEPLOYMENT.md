# Vercel Production Deployment Guide

This document provides a professional, step-by-step procedure to deploy the **SafeRoute AI** system onto Vercel's serverless infrastructure. Vercel is highly optimized for frontend frameworks but has strict limits for Python backends, making this specific configuration critical for a successful deployment.

## Architecture Overview on Vercel
To guarantee a 100% free tech stack, we utilize Vercel's advanced routing:
1. **Frontend (Static Host)**: Vercel instantly deploys the HTML/CSS/JS maps inside the `/frontend` directory via its global CDN (`@vercel/static`).
2. **Backend (Serverless API)**: Vercel spins up temporary micro-containers using `@vercel/python` specifically to execute `app.py`. When a user requests `/predict`, Vercel boots the Flask app, loads the `accident_model.pkl` Random Forest model, returns the coordinates, and shuts down instantly.

---

## Critical Dependency Optimization
Serverless Functions on Vercel's Free-Tier possess a rigid **250MB size limit**. 

Machine learning applications are notoriously heavy. A full training environment utilizing `xgboost`, `osmnx`, `geopandas`, and `lightgbm` will **fail to build** on Vercel due to memory limits.
To solve this, the `requirements.txt` has been strictly minimized to contain **only inference-level dependencies**:
*   `Flask`
*   `flask-cors`
*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `joblib`

*If you plan to train the model locally again, you must reinstall the broader data science suite separately.*

---

## Deployment Steps

### Step 1: Version Control (GitHub)
Push the current state of your repository to GitHub. Ensure the `/models` directory (containing `.pkl` files) and `/frontend` directory are pushed upstream.

```bash
git init
git add .
git commit -m "Initialize Vercel deployment structure"
git branch -M main
git remote add origin https://github.com/YourUsername/safe-route-ai.git
git push -u origin main
```

### Step 2: Vercel Dashboard Import
1. Navigate to your [Vercel Dashboard](https://vercel.com/dashboard).
2. Click **Add New** > **Project**.
3. Import the `safe-route-ai` repository from your linked GitHub account.

### Step 3: Deployment Configuration
Vercel's advanced builder will detect the `vercel.json` file in the root directory. This acts as the absolute source of truth for the deployment map.
*   **Framework Preset**: Leave as `Other`.
*   **Root Directory**: Leave as `./`.
*   **Build Command**: Vercel will auto-override this based on `@vercel/python`. Leave blank.

Click **Deploy**.

---

## Troubleshooting Build Errors

### "Max Serverless Function Size Exceeded"
If Vercel throws an error regarding function sizes being greater than 250MB:
1. Double-check that `requirements.txt` does not contain `scipy` or `xgboost`. `scikit-learn` natively relies on `scipy`, which adds ~100MB to the build. Vercel usually natively accommodates this alongside Pandas if kept barebones.
2. If the build continuously rejects the size, the **industry-standard fallback** is to decouple the system: Host the `/frontend` directory on Vercel, and push the Flask `app.py` script to **Render.com** (which natively supports gigabytes of free tier ML environments).

### 500 Internal Server Error (Model Not Found)
Vercel Serverless executes from abstract read-only pathing. The system relies on absolute dynamic paths utilizing `os.path.dirname(__file__)` which we have already integrated into `app.py`. If this occurs, verify that `/models/accident_model.pkl` successfully uploaded to Github, as git `.gitignore` files notoriously ignore large binaries by default.
