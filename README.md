# 📰 Spotting the Fake News — NLP Classification Pipeline

An end-to-end Natural Language Processing (NLP) and Machine Learning pipeline built to automatically classify news articles as **Real (1)** or **Fake (0)** across a dataset of over 44,000 articles.

## 📌 Project Overview & Workflow
1. **Data Preparation & Leakage Prevention:** Merged `True.csv` and `Fake.csv`, handled missing/duplicate records, and stripped agency tags (such as `Reuters`) from authentic articles so the models learn actual writing styles rather than memorizing source names.
2. **Text Preprocessing & Feature Engineering:** Combined headlines and article bodies, normalized text via lowercasing, stopword removal, and Porter Stemming (`NLTK`), and mapped URLs and emotional punctuation (`!`, `?`) into dedicated tokens (`urllink`, `exclammark`, `questionmark`).
3. **Feature Extraction:** Transformed cleaned text into numerical vectors using three distinct techniques: **Count Vectorizer**, **TF-IDF**, and **Word2Vec** (`Gensim`).
4. **Model Benchmarking & Optimization:** Evaluated four classification algorithms (**Logistic Regression**, **Naive Bayes**, **Linear SVM**, and **Random Forest**) across all vectorizers (13 total experiments), followed by hyperparameter tuning and Bigram `(1,2-gram)` expansion for Random Forest.
5. **Evaluation & Live Inference:** Assessed performance using **Accuracy**, **F1-Score**, and a **Confusion Matrix**, achieving the highest score with **Linear SVM + TF-IDF (98.85% F1-Score | 98.75% Accuracy)**, and tested the final model on unseen custom news samples.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** `Pandas`, `NumPy`, `NLTK`, `Gensim`, `Scikit-learn`, `Matplotlib`, `Seaborn`
