
# 🧠 Reddit Mental Health Prediction

This repository presents an end-to-end machine learning pipeline to **predict mental health conditions** (Stress, Anxiety, Depression) using **Reddit posts**. The goal is to explore how **natural language processing (NLP)** and supervised learning can be applied to unstructured social media text to assist in early detection of mental health issues.

---

## 📁 Project Structure

```

reddit-mental-health-prediction/
│
├── results/ # 📊 Evaluation Visuals
│   ├── confusion_matrix_svm.png
│   ├── roc_curve_svm_multiclass.png
│   ├── model_accuracy_comparison.png
│   └── model_performance_comparison.png
|
├── README.md # 📘 Project overview
├── best_model.pkl # 💾 Trained best-performing model
├── reddit_mental_health_prediction.ipynb # 🧪 Full ML pipeline in one notebook
├── reddit_dataset.csv # dataset which used for project
├── requirements.txt # 📦 Python dependencies


---

## 🎯 Project Objective

The aim is to:
- Use **Reddit mental health posts** as unstructured data.
- Apply **NLP preprocessing** (tokenization, stopword removal, lemmatization).
- Perform Exploratory Data Analysis (EDA) to understand class distribution, word frequency, and insights.
- Extract features using **TF-IDF**
- Split data using an 80/20 train-test ratio for model training and evaluation.
- Train and evaluate models like **Support Vector Machine (SVM)**, **Random Forest**, **XGBoost**.**Naïve Bayes**, **Random Forest**, and **Logistic Regression**.
- 💠 Support Vector Machine (SVM)
- 🌳 Random Forest
- ⚡ XGBoost
- 🧠 Artificial Neural Network (ANN)
- 📈 Logistic Regression
- 📊 Naïve Bayes  

- Identify which algorithm performs best in predicting mental health categories.

---

## 📚 Dataset

📥 **Download Reddit Dataset**: [`reddit_dataset.csv`](./reddit_dataset.csv)
---

## 📊 Target Labels
According to research....
- **0 → Stress**
- **1 → Depression**
- **2 → Anxiety**

---


📦 Libraries used:

* pandas
* numpy
* scikit-learn
* nltk
* matplotlib
* seaborn
* joblib


---

## 📈 Model Evaluation

The models are evaluated using:

* Confusion Matrix
* Accuracy Comparison
* Precision / Recall / F1-Score
* ROC-AUC

Results are saved under the `results/` folder.

---


## 📦 Output

* Evaluation report in `results/mo.csv`
* Confusion matrix image in `results/confusion_matrix_svm.png`
* Best model saved as `best_model.pkl`


---

## 🚀 Future Work

🚀 Expand to include structured survey data
🌐 Deploy as an interactive web app
💬 Integrate sentiment analysis and topic modeling


---

## ✨ Author

**Muhammad Zain Mushtaq**
[LinkedIn](https://linkedin.com/in/your-link) • [GitHub](https://github.com/your-username)

---

