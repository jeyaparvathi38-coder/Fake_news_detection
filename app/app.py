from flask import Flask, render_template, request, jsonify
import os
import time
import re
import requests as http_requests

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Hugging Face Inference API config
# Set these as environment variables in Vercel dashboard:
#   HF_API_URL   = https://api-inference.huggingface.co/models/YOUR_USERNAME/YOUR_MODEL
#   HF_API_TOKEN = your HuggingFace read token
# ---------------------------------------------------------------------------
HF_API_URL   = os.environ.get("HF_API_URL", "")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def predict_via_hf_api(text):
    """Call Hugging Face Inference API and return (result, confidence)."""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text[:512]}
    try:
        resp = http_requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # HF returns: [[{"label": "LABEL_0", "score": 0.99}, ...]]
        if isinstance(data, list) and isinstance(data[0], list):
            scores = data[0]
        elif isinstance(data, list):
            scores = data
        else:
            return None, None

        label_0 = next((x["score"] for x in scores if x["label"] == "LABEL_0"), 0)
        label_1 = next((x["score"] for x in scores if x["label"] == "LABEL_1"), 0)

        if label_1 > label_0:
            return "FAKE", round(label_1 * 100, 2)
        else:
            return "REAL", round(label_0 * 100, 2)
    except Exception as e:
        print("HF API Error:", e)
        return None, None

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/insights')
def insights():
    return render_template('insights.html')

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    start_time = time.time()
    data = request.json
    content = data.get('content', '')

    if not content:
        return jsonify({"error": "Content is required"}), 400

    cleaned = clean_text(content)

    # Try HF Inference API first
    if HF_API_URL and HF_API_TOKEN:
        result, confidence = predict_via_hf_api(cleaned)
        if result:
            risk_level = ("High" if confidence > 80 else "Medium") if result == "FAKE" else "Low"
            processing_time = round(time.time() - start_time, 2)
            return jsonify({
                "result": result,
                "confidence": confidence,
                "processing_time": processing_time,
                "risk_level": risk_level,
                "model_used": "DistilBERT (v2 Indian Dataset) via HF API",
                "top_positive": [],
                "top_negative": []
            })

    # Fallback: demo mode (no model configured)
    time.sleep(1.5)
    processing_time = round(time.time() - start_time, 2)
    return jsonify({
        "result": "DEMO",
        "confidence": 0,
        "processing_time": processing_time,
        "risk_level": "N/A",
        "model_used": "Demo Mode — Configure HF_API_URL & HF_API_TOKEN in Vercel dashboard",
        "top_positive": [],
        "top_negative": []
    })

@app.route('/api/notebook_stats', methods=['GET'])
def notebook_stats():
    return jsonify({
        "dataset_info": {
            "total_records": 41336,
            "real_news": 21576,
            "fake_news": 19760,
            "features_used": 2
        },
        "model_performance": {
            "accuracy": 99.90,
            "precision": 99.82,
            "recall": 99.97,
            "f1_score": 99.89,
            "roc_auc": 99.95
        },
        "training_history": {
            "epochs": ["Epoch 1", "Epoch 2"],
            "training_loss": [0.006160, 0.006759],
            "validation_loss": [0.008715, 0.008320]
        },
        "confusion_matrix": {
            "true_negative": 4316,
            "false_positive": 0,
            "false_negative": 8,
            "true_positive": 3944
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
