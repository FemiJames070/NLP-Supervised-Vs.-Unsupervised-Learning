# NLP-Supervised-Vs.-Unsupervised-Learning

NLP Supervised Learning Vs. Unsupervised Learning
This repository provides a comprehensive comparative analysis of Natural Language Processing (NLP) techniques, contrasting supervised text classification with unsupervised clustering and topic modeling. Following the CRISP-DM methodology, the project applies both traditional machine learning and deep learning models to two distinct types of text data: subjective customer sentiment and objective technical descriptions.

📊 Data Sources
The analysis utilizes two high-quality datasets from Kaggle to compare model performance across different linguistic contexts:

TripAdvisor Hotel Reviews (~20k rows): Subjective, sentiment-heavy text. Used to predict/discover a 1–5 star rating scale.

Wine Tasting Reviews (~130k rows): Objective, highly technical text. Used to predict/discover quality tiers based on complex lexical flavor profiles.

📓 Notebook 1: Supervised Learning (Text Classification)
Goal: Build and test several machine learning models to categorize text into predefined labels (Star Ratings for hotels, engineered Quality Tiers for wines).

Methodology & Models
Preprocessing: Lowercasing, punctuation removal, NLTK tokenization, stopword removal, and WordNet lemmatization.

Vectorization: TF-IDF (Term Frequency-Inverse Document Frequency) and Token Padding (for deep learning).

Models Evaluated:

Multinomial Naive Bayes (MNB) - Baseline

Support Vector Machine (LinearSVC)

Random Forest Classifier

Deep LSTM (Long Short-Term Memory network)

Key Findings
The Best Performer: LinearSVC consistently outperformed all other models across both datasets (Macro F1 of 0.51 for TripAdvisor, 0.64 for Wine). It proved highly capable of finding optimal decision boundaries in high-dimensional, sparse TF-IDF feature spaces, even amidst severe class imbalance.

The Imbalance Trap: The baseline MNB struggled significantly with minority classes, often completely failing to predict the "Neutral" hotel ratings or "Excellent" wine categories.

Deep Learning Overkill: The Deep LSTM architecture underperformed compared to traditional linear models. It quickly overfit the training data and suffered from mode collapse, indicating that for short, polarized text, sequence (word order) is less important than simple feature weighting.

Tree-based Limitations: Random Forest delivered the weakest performance, confirming that non-linear, tree-based models are poorly suited for sparse text data.

📓 Notebook 2: Unsupervised Learning (Clustering & Topic Modeling)
Goal: Utilize clustering and dimensionality reduction to discover hidden structures, semantic segments, and thematic patterns without relying on predefined labels.

Methodology & Models
Feature Engineering: Word Count and Sentiment Polarity (TextBlob) were added as latent descriptors to profile clusters.

Clustering Algorithms:

K-Means Clustering (tested via the Elbow Method)

Agglomerative Hierarchical Clustering

Topic Modeling & Sequence Discovery:

Latent Dirichlet Allocation (LDA)

Non-Negative Matrix Factorization (NMF)

LSTM-Based Next-Word Prediction (Semantic Pattern Discovery)

Key Findings
Subjective vs. Technical: Unsupervised algorithms perform very differently depending on the text type. Subjective data (TripAdvisor) forms distinct clusters based on emotional valence, whereas technical data (Wine) is highly standardized and dense, relying strictly on specific jargon (flavor profiles) rather than sentiment.

Topic Modeling Success: LDA proved to be the most "human-like" model, achieving the highest alignment with original human ratings (Highest NMI score). It successfully identified logical overlapping themes (e.g., Service, Location, Cleanliness).

Fuzzy Boundaries: Silhouette scores across the board remained low. Text data naturally has overlapping vocabulary (e.g., words like "hotel" or "wine" appear everywhere), resulting in "fuzzy" clusters rather than perfectly isolated groups.

Business Application: For automated text analysis pipelines, LDA should be prioritized for deep thematic discovery, while K-Means serves best for rapid, broad customer segmentation.

🛠️ Technologies & Libraries
Language: Python 3

Data Manipulation & EDA: Pandas, NumPy, Matplotlib, Seaborn, TextBlob

NLP Processing: NLTK (Stopwords, WordNetLemmatizer, Tokenizer), WordCloud

Machine Learning: Scikit-learn (TF-IDF, LinearSVC, MNB, RandomForest, K-Means, Agglomerative, LDA, NMF)

Deep Learning: TensorFlow / Keras (Sequential, LSTM, Embedding, EarlyStopping)

🚀 How to Run
Clone the repository.

Ensure you have the datasets downloaded from Kaggle and placed in the root directory (or update the file paths in the notebooks).

Install the required dependencies: pip install pandas numpy matplotlib seaborn scikit-learn nltk tensorflow textblob wordcloud

Run the Jupyter Notebooks sequentially to observe the data pipeline from preprocessing to model evaluation.

Author: Femi James
