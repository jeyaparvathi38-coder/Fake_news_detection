# NewsShield: AI-Powered Fake News Detection

![NewsShield Logo](app/static/images/hero-glow.png) *(Representative)*

**NewsShield** is a state-of-the-art web application powered by a fine-tuned **DistilBERT** Transformer neural network. Designed to combat misinformation, it analyzes text patterns, context, and linguistic cues to classify news articles as **Real** or **Fake** with up to 99.9% accuracy.

## 🚀 Features

- **Advanced NLP Engine:** Built on HuggingFace's DistilBERT, trained on a massive database of 40,000+ Indian News articles.
- **Explainable AI (SHAP):** Doesn't just give a verdict—it highlights the exact words and phrases that contributed to the classification.
- **Modern UI/UX:** A stunning, fully responsive dark-themed dashboard featuring:
  - Frosted glassmorphism design (Glass Cards).
  - Neural constellation and floating data node background animations.
  - A mathematically precise 3D orbital feature ring.
- **Real-Time Analysis:** Sub-second inference speeds utilizing lightweight transformer models.
- **In-Depth Insights:** Detailed dashboards showcasing model training loss, confusion matrices, and dataset label distributions.

## 🛠️ Technology Stack

- **Frontend:** HTML5, CSS3, Bootstrap 5, FontAwesome, Vanilla JS
- **Backend:** Python 3.11, Flask
- **Machine Learning & NLP:** 
  - `PyTorch` (Deep Learning Framework)
  - `Transformers` (HuggingFace DistilBERT)
  - `SHAP` (Explainability)
  - `Scikit-Learn`, `Pandas`, `NumPy` (Data Processing)

## 📂 Project Structure

```text
fake_news_detection/
├── app/
│   ├── app.py                 # Main Flask application
│   ├── templates/             # HTML templates (index, detect, about, insights, methodology)
│   └── static/                # CSS, JS, and Images
├── distilbert_v2_indian_model/# Fine-tuned DistilBERT model weights (ignored in git)
├── fake_news_v2_indian.ipynb  # Jupyter notebook for EDA and Model Training
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/jeyaparvathi38-coder/Fake_news_detection.git
   cd Fake_news_detection
   ```

2. **Create a Virtual Environment (Optional but recommended)**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train/Load the Model**
   - *Note: Due to GitHub file size limits, the 200MB+ `model.safetensors` is not included.*
   - To generate the model, run the cells in `fake_news_v2_indian.ipynb` using the `Fake.csv` and `True.csv` datasets. The notebook will automatically save the model to `distilbert_v2_indian_model/`.

5. **Run the Flask Web Server**
   ```bash
   cd app
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000`.

## 🧠 How It Works

1. **Text Cleaning:** Raw text input is stripped of URLs, special characters, and excessive whitespace.
2. **Tokenization:** Text is converted into mathematical subword tokens that the neural network can interpret.
3. **Self-Attention:** DistilBERT's attention layers map relationships between words to understand the deep contextual meaning.
4. **Classification:** A Softmax layer calculates confidence scores for `REAL` vs `FAKE`.
5. **SHAP Interpretation:** The model highlights the top 5 positive and negative contributing words to explain its logic.

---

*Built with ❤️ to combat misinformation.*
