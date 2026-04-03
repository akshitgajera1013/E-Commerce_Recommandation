# 🛒 RetailNexus: Omni-Channel Recommendation Engine

Deployment Link :- https://e-commercerecommandation-kfxvyhpheck66gawzktysp.streamlit.app/

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![NLP](https://img.shields.io/badge/Machine%20Learning-NLP%20%2F%20TF--IDF-orange?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Data%20Viz-Plotly-purple?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Frontend-Python_Engine-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

An advanced, monolithic Python application built for institutional-grade e-commerce product recommendation and retail analytics. This system leverages a highly optimized **Content-Based Filtering Architecture** utilizing TF-IDF vectorization and Cosine Similarity mapping to generate highly relevant product slates based on semantic spec overlap.

---

## 🧠 System Architecture & Capabilities

This platform transforms unstructured product catalogs into a high-dimensional mathematical space to predict semantic similarity without relying on user purchase history.

### 1. Dual-Pass Search & Resolution Engine
Users input a target product to seed the recommendation engine. The system utilizes a dual-pass architecture:
* **Pass 1 (Keyword Anchor):** Scans the catalog for broad category matches (e.g., "Laptop") and dynamically anchors the vector math to the most foundational product in that category.
* **Pass 2 (Fuzzy Match):** If no keyword is found, it utilizes a Levenshtein Distance algorithm to resolve user typos and autocorrect the search query.

### 2. Natural Language Processing Pipeline
The engine mathematically analyzes a concatenated metadata string (`Name`, `Brand`, `Category`, `Details`) for every product in the inventory. It converts this text into mathematical vectors using a Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer, penalizing generic marketing words and heavily weighting unique, identifying product specifications.

### 3. Cosine Similarity Analytics
Calculates the dot product between the user's target vector and all other vectors in the high-dimensional matrix. It outputs the top 10 matches based on the smallest cosine angle, visualized alongside pricing distribution charts and similarity decay curves.

### 4. Conversion Forecasting
Forecasts customer cart-addition probability via a logistic growth algorithm. Simulates time-on-page interaction vs. purchase intent curves for the top recommended products based on their respective similarity scores.

### 5. Click-Through Rate (CTR) Volatility (Monte Carlo)
Executes a 1,000-iteration stochastic mathematical simulation to model behavioral variance. Applies simulated human-behavioral volatility against the base similarity score to map the probable Click-Through Rate distribution of the #1 recommendation.

### 6. Secure Data Export (JSON / CSV)
Generates an official Retail Dossier tagged with a unique cryptographic Vector ID. Enables base64-encoded, secure local downloads of the entire recommendation payload in both programmatic (JSON) and ledger (CSV) formats.

---

## 🛠️ Technical Stack

* **Core Logic & Computation:** `Python 3.x`, `numpy`
* **Data Processing & Pipelines:** `pandas`
* **Machine Learning Architecture:** `scikit-learn` (TF-IDF Vectorizer, Linear Kernel)
* **Interactive Data Visualization:** `plotly.express`, `plotly.graph_objects`
* **Frontend Delivery:** Custom Python-rendered UI engine with over 350 lines of injected, dynamic CSS.

---

## 📂 Repository Structure

```text
├── app.py                         # Main monolithic Python application interface
├── recommendation_system.pkl      # Bundled ML Assets (DataFrame, TF-IDF Vectorizer, Matrix)
├── requirements.txt               # Python package dependencies
└── README.md                      # System documentation




⚙️ Installation & Deployment

1. Clone the Repository

git clone [https://github.com/akshitgajera1013/E-Commerce-Recommandation.git](https://github.com/akshitgajera1013/E-Commerce-Recommandation.git)

cd E-Commerce-Recommandation

2. Install Dependencies

pip install -r requirements.txt

3. Initialize the Application Server

streamlit run app.py
