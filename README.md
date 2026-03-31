# NLP: Supervised vs. Unsupervised Learning 🧠📊

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-lightgrey.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00.svg)](https://www.tensorflow.org/)

## 📖 Overview
This repository provides a comprehensive comparative analysis of Natural Language Processing (NLP) techniques, explicitly contrasting **Supervised Text Classification** with **Unsupervised Clustering and Topic Modeling**. 

Following the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology, this project evaluates how both traditional Machine Learning and Deep Learning architectures handle two fundamentally different types of text data: subjective emotional sentiment and objective technical jargon.

## 📂 Data Sources
The analysis utilizes two distinct datasets from Kaggle to test the versatility of our models:
1. **[TripAdvisor Hotel Reviews](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews)** (~20k rows): Subjective, sentiment-heavy text. Used to predict and discover a 1–5 star rating scale.
2. **[Wine Tasting Reviews](https://www.kaggle.com/datasets/mysarahmadbhat/wine-tasting)** (~130k rows): Objective, highly technical text. Used to predict and discover quality tiers based on complex, lexical flavor profiles.

---

## 📓 Notebook 1: Supervised Learning (Text Classification)
**File:** `Supervised_Learning_Text_Classification.ipynb`

**Goal:** Build and test several machine learning models to categorize text into predefined labels (Star Ratings for hotels, engineered Quality Tiers for wines).

### Methodology & Models
* **Preprocessing Pipeline:** Lowercasing, punctuation removal, NLTK tokenization, stopword removal, and WordNet lemmatization.
* **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency) for ML models; Tokenization and Padding for Deep Learning.
* **Models Evaluated:**
  * Multinomial Naive Bayes (MNB) - *Baseline*
  * Support Vector Machine (LinearSVC)
  * Random Forest Classifier
  * Deep LSTM (Long Short-Term Memory network)

### Key Findings
* 🏆 **Best Performer:** **LinearSVC** consistently outperformed all other models across both datasets (Macro F1 of 0.51 for TripAdvisor, 0.64 for Wine). It proved highly capable of finding optimal decision boundaries in high-dimensional, sparse TF-IDF feature spaces, overcoming severe class imbalances.
* 📉 **The Imbalance Trap:** The baseline MNB struggled significantly with minority classes, completely failing to predict the "Neutral" hotel ratings or "Excellent" wine categories.
* ⚠️ **Deep Learning Overkill:** The Deep LSTM architecture underperformed compared to traditional linear models. It quickly overfit the training data and suffered from mode collapse, indicating that for short, polarized text, word presence (feature weighting) is more predictive than word sequence.
* 🌳 **Tree-based Limitations:** Random Forest delivered the weakest performance, confirming that non-linear, tree-based models are poorly suited for sparse text matrices.

---

## 📓 Notebook 2: Unsupervised Learning (Clustering & Topic Modeling)
**File:** `Unsupervised Learning_Segmentation_Clustering.ipynb`

**Goal:** Utilize clustering and dimensionality reduction to discover hidden structures, semantic segments, and thematic patterns without relying on predefined labels or human ratings.

### Methodology & Models
* **Feature Engineering:** `Word Count` and `Sentiment Polarity` (via TextBlob) were added as latent descriptors to profile generated clusters mathematically.
* **Algorithms Evaluated:**
  * K-Means Clustering (Optimized via the Elbow Method)
  * Agglomerative Hierarchical Clustering
  * Latent Dirichlet Allocation (LDA)
  * Non-Negative Matrix Factorization (NMF)
  * LSTM-Based Next-Word Prediction (Semantic Pattern Discovery)

### Key Findings
* 🧠 **Subjective vs. Technical Text:** Subjective data (TripAdvisor) forms distinct clusters based on emotional valence, whereas technical data (Wine) is highly standardized and dense, relying strictly on specific jargon (flavor profiles) rather than sentiment.
* 🏆 **Topic Modeling Success:** **LDA** proved to be the most "human-like" model, achieving the highest alignment with original human ratings (Highest NMI score). It successfully identified logical overlapping themes (e.g., Service, Location, Cleanliness).
* 🌫️ **Fuzzy Boundaries:** Silhouette scores across the board remained low. Text data naturally has overlapping vocabulary (e.g., words like "hotel" or "wine" appear everywhere), resulting in "fuzzy" semantic clusters rather than perfectly isolated geometric groups.
* 💡 **Business Application:** For automated text analysis pipelines, **LDA** should be prioritized for deep thematic discovery, while **K-Means** serves best for rapid, broad customer segmentation.

---

## 🛠️ Technologies & Libraries
* **Language:** Python 3
* **Data Manipulation & EDA:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `TextBlob`
* **NLP Processing:** `nltk` (Stopwords, WordNetLemmatizer, Tokenizer), `wordcloud`
* **Machine Learning:** `scikit-learn` (TF-IDF, LinearSVC, MNB, RandomForest, K-Means, Agglomerative, LDA, NMF)
* **Deep Learning:** `tensorflow` / `keras` (Sequential, LSTM, Embedding, EarlyStopping)

---

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/nlp-supervised-vs-unsupervised.git](https://github.com/yourusername/nlp-supervised-vs-unsupervised.git)
   cd nlp-supervised-vs-unsupervised

## Install the required dependencies:

Bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk tensorflow textblob wordcloud
Download the Data:
Ensure you have downloaded the datasets from Kaggle (links above) and placed the .csv files in the root directory.

Run the Notebooks:
Launch Jupyter Notebook or Jupyter Lab:

Bash
jupyter notebook
Open Supervised_Learning_Text_Classification.ipynb or Unsupervised Learning_Segmentation_Clustering.ipynb and run the cells sequentially.

✍️ Author
Femi James
