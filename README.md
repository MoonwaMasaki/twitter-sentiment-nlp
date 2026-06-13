# Tweet Sentiment Analysis
### An NLP Sentiment Classifier for Apple's Marketing & Product Design Team

**Author:** Angela Masaki

**Date:** 08/06/2026 - 13/06/2026

**Dataset:** CrowdFlower via Kaggle — Apple Twitter Sentiment

CrowdFlower. (2016). Apple Twitter Sentiment. Retrieved [2026] from https://www.kaggle.com/datasets/slythe/apple-twitter-sentiment-crowdflower. Original source: https://data.world/crowdflower/brands-and-product-emotions.

---

## Project Overview

This project builds an NLP sentiment classifier for tweets about Apple products. The dataset consists of approximately 3,886 tweets collected from 2014, hand-labeled by human annotators using a numeric sentiment scale: 1 (negative), 3 (neutral), and 5 (positive).

The business context is Apple's Marketing & Product Design team, who need a way to monitor public sentiment on social media at scale — without manually reading every tweet. A deployed classifier could sit inside a brand monitoring dashboard, automatically flagging negative sentiment during product launches or major events in real time.

The project follows the full data science lifecycle: business understanding, data preparation, iterative modeling, evaluation, and model explainability. The final model is a tuned LinearSVC classifier that achieves a macro F1-score of 0.79 on the binary (positive vs. negative) task and 0.67 on the three-class task.

---

## Techniques Employed

To build and evaluate the sentiment classifier, the following NLP and machine learning techniques were used:

- **Text preprocessing** with Python's `re` library — removing URLs, @mentions, hashtags, punctuation, and digits
- **Stopword removal** and **lemmatization** using NLTK's English stopword list and `WordNetLemmatizer`
- **Bag-of-words vectorization** using scikit-learn's `CountVectorizer` (baseline)
- **TF-IDF vectorization** with unigrams and bigrams using scikit-learn's `TfidfVectorizer`- **Feature engineering** with hand-crafted numeric features (tweet length, word count, exclamation count, question count, caps ratio, emoji proxy) extracted from raw tweet text using a custom scikit-learn `TransformerMixin`
- **Multinomial Naive Bayes** as a baseline classifier
- **Logistic Regression** with balanced class weights for class imbalance handling
- **Linear Support Vector Classification (LinearSVC)** — the strongest performer for high-dimensional text data
- **Hyperparameter tuning** via `GridSearchCV` with 5-fold stratified cross-validation
- **FeatureUnion** combining TF-IDF vectors and hand-crafted features into a single feature matrix for Model 5
- **Model explainability** via LinearSVC coefficient inspection to surface the most predictive words per sentiment class
- **Macro F1-score** as the primary evaluation metric, chosen due to class imbalance

---

## Project Outline

The project is structured as a single end-to-end Jupyter Notebook with the following sections, each building on the previous:

1. **Business Understanding** — Defining the stakeholder, the problem, and how the model will be used
2. **Data Understanding** — Exploring the dataset, inspecting class distribution, and identifying limitations
3. **Data Preparation** — Cleaning tweet text, mapping sentiment labels, removing stopwords, lemmatizing, and vectorizing
4. **Modeling** — Building five models iteratively, from a Naive Bayes baseline to a tuned LinearSVC, on both binary and multiclass tasks
5. **Model Comparison** — Evaluating and comparing all models on the held-out test set using macro F1
6. **Final Model Evaluation** — Confusion matrix, classification report, and LinearSVC coefficient-based explainability
7. **Conclusion** — Results summary, limitations, and recommendations for next steps

---

## Dataset

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/slythe/apple-twitter-sentiment-crowdflower). Once downloaded, place the CSV file (`Apple-Twitter-Sentiment-DFE.csv`) in the root directory of the repository alongside the notebook.

The dataset contains 3,886 tweets with the following key columns:
- `text` — the raw tweet content
- `sentiment` — numeric label: `1` = negative, `3` = neutral, `5` = positive, `not_relevant` = excluded

The original source is available at [data.world/crowdflower/brands-and-product-emotions](https://data.world/crowdflower/brands-and-product-emotions).

---

## Repository Structure

```
twitter-sentiment-nlp/
├── phase4_nlp_sentiment.ipynb   # Main notebook (full analysis)
├── presentation.pptx            # Non-technical stakeholder presentation
├── README.md                    # This file
└── .gitignore                   # Excludes data files and system files
```

---

## How to Reproduce

1. Clone this repository
```bash
git clone https://github.com/MoonwaMasaki/twitter-sentiment-nlp.git
cd twitter-sentiment-nlp
```

2. Install dependencies
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/slythe/apple-twitter-sentiment-crowdflower) and place `Apple-Twitter-Sentiment-DFE.csv` in the root directory

4. Open and run the notebook
```bash
jupyter notebook phase4_nlp_sentiment.ipynb
```

> NLTK corpora (stopwords, wordnet) are downloaded automatically on first run.

