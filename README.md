# SafeRoute AI - ML Powered Navigation

![SafeRoute Cover](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Flask](https://img.shields.io/badge/Backend-Flask-black) ![Leaflet](https://img.shields.io/badge/Frontend-Leaflet.js-lightgreen)

SafeRoute AI is an end-to-end Machine Learning navigation system built to identify high-risk traffic zones and compute the *absolute safest* routes between two destinations. By utilizing historical US Accident data, our proprietary machine learning models evaluate route safety dynamically in real-time, functioning entirely on open-source, free-tier tech stacks.

## Key Features
* **Google Maps-Style UI**: A sleek, beautifully designed frontend using `Leaflet.js` and OpenStreetMap HOT tiles.
* **Smart Autocomplete**: Native location search powered by the free Nominatim API.
* **Risk Heatmap**: Visualize accident-prone regions dynamically overlaid on the map.
* **ML Route Evaluation**: Instead of just calculating the shortest path, the system calculates multiple permutations of routes, tests them against a Random Forest classification model, and factors in `Cost = Distance + (Severity * Weight)` to highlight the inherently safest route in vibrant green. 

## Flow of the Program
1. **User Input:** The user types their source and destination in the frontend. 
2. **Geocoding:** The native search bar queries the Nominatim API to get coordinates for the selected locations.
3. **Route Generation:** The `Leaflet Routing Machine` queries a routing server to find possible route permutations between the start and end coordinates.
4. **Data Sampling:** The frontend scripts sample coordinates along each route and send a sequence of waypoints to the Flask backend's `/predict` API.
5. **Model Evaluation:** The Flask application feeds the coordinates and current environmental details into the pre-trained `RandomForest` machine learning model.
6. **Continuous Scoring:** The backend uses `.predict_proba()` to evaluate probabilities of accident severity, calculating a highly precise continuous expected severity score for each waypoint. The average score for the entire route is calculated.
7. **Safest Path Selection:** The frontend compares the overall predicted severity across all permutations and ranks the safest route on the map in a prominent layout.

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
python src/train.py 
```
*Note: This generates `accident_model.pkl` and preprocessing configurations in the `/models/` directory.*

### 3. Running the Server
Launch the Flask application bridging the backend ML to the frontend UI:
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

---

## Production Deployment (Important)

### The Large File Issue
Machine learning models (such as `accident_model.pkl` at roughly ~284MB) are too substantial to be tracked on GitHub out-of-the-box. We have untracked `models/*.pkl` from `.gitignore` so they register with Git. 

**Because the model is over 100MB, you MUST use [Git LFS](https://git-lfs.com/) (Large File Storage) to push this repository to GitHub:**
```bash
git lfs install
git lfs track "models/*.pkl"
git add .gitattributes models/*.pkl
git commit -m "Add ML models via LFS"
git push
```

### Vercel vs Render
- **Vercel** is extremely limited for heavy Python servers (250MB size limit for serverless functions). If you push the 284MB model file here, you will hit a `Max Serverless Function Size Exceeded` error and the `/predict` API will drop requests, causing the map search to break.
- **Render** natively supports heavier Python dependencies. **We highly recommend deploying the Flask App via Render:**
  1. Login to Render and connect your GitHub repository.
  2. Create a **Web Service**.
  3. Environment: `Python 3`. Build Command: `pip install -r requirements.txt`. Start Command: `gunicorn app:app`.
  4. Render will handle the heavy model file correctly.

## License
MIT License. Open-source and Free to use.
