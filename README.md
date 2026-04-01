# SafeRoute AI - ML Powered Navigation

![SafeRoute Cover](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Flask](https://img.shields.io/badge/Backend-Flask-black) ![Leaflet](https://img.shields.io/badge/Frontend-Leaflet.js-lightgreen)

SafeRoute AI is an end-to-end Machine Learning navigation system built to identify high-risk traffic zones and compute the *absolute safest* routes between two destinations. By utilizing historical US Accident data, our proprietary machine learning models evaluate route safety dynamically in real-time, functioning entirely on open-source, free-tier tech stacks.

## Key Features
* **Google Maps-Style UI**: A sleek, beautifully designed frontend using `Leaflet.js` and OpenStreetMap HOT tiles.
* **Smart Autocomplete**: Native location search powered by the free Nominatim API.
* **Risk Heatmap**: Visualize accident-prone regions dynamically overlaid on the map.
* **ML Route Evaluation**: Instead of just calculating the shortest path, the system calculates multiple permutations of routes, tests them against a Random Forest classification model, and factors in `Cost = Distance + (Severity * Weight)` to highlight the inherently safest route in vibrant green. 

## Tech Stack
* **Frontend**: HTML5, Vanilla JavaScript, CSS3, Leaflet.js, Leaflet Routing Machine.
* **Backend**: Python, Flask, Flask-CORS.
* **Machine Learning**: Scikit-Learn, Pandas, Numpy (Random Forest Classifier).
* **Geospatial Processing**: Nominatim Geocoding API.

---

## Local Development Setup

### 1. Environment Parsing
Ensure you have Python 3.9+ installed. Set up your virtual environment cleanly:
```bash
python3 -m venv myenv
source myenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Model Training (Optional)
If you wish to retrain the model on new data, execute the training pipeline.
```bash
python src/train.py "data/Your_New_Data.csv"
```
*Note: This generates `accident_model.pkl` and preprocessing configurations in the `/models/` directory.*

### 3. Running the Server
Launch the Flask application bridging the backend ML to the frontend UI:
```bash

```
Visit `http://localhost:5000` in your browser.

---

## Vercel Production Deployment

This application is structurally engineered to be deployed securely onto **Vercel's** serverless hosting environment.

### Prerequisites:
1. Initialize a Git repository and push this code to GitHub.
2. Create an account on [Vercel](https://vercel.com).

### Deployment Steps:
1. Log into Vercel and select **"Add New Project"**.
2. Import the SafeRoute AI GitHub repository.
3. Vercel will automatically detect the `vercel.json` configuration provided in the root directory.
4. Keep the framework preset to `Other`. 
5. Click **Deploy**. Vercel will process the serverless functions via `@vercel/python` and statically host the `frontend/` directory.

### Important Note on Vercel Serverless ML Limits:
*Vercel's free tier imposes a 50MB execution limit and a 250MB total uncompressed size limit for Serverless Functions. Since Scikit-Learn, Pandas, and the 30MB `.pkl` model are extremely heavy, deployment on Vercel may occasionally exceed this limit during the build phase depending on your exact dependency tree. If you encounter a `Max Serverless Function Size Exceeded` error on Vercel, it is highly recommended to deploy the backend (`app.py`) separately via **Render** (which natively supports heavy Python binaries), while retaining Vercel exclusively for the static `/frontend` architecture.*

## License
MIT License. Open-source and Free to use.
# US-Accidents-Pred
# US-Accidents-Pred
# US-Accidents-Pred
# US-Accidents-Pred
# US-Accidents-Pred
