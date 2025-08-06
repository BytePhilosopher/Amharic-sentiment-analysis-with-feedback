```markdown
# 🇪🇹 QalAnalyzer (ቃል Analyzer)

**QalAnalyzer** is a machine learning-based sentiment analysis tool for the Amharic language. It classifies Amharic text into three categories: **Positive**, **Negative**, and **Neutral**. Built with a focus on low-resource language NLP, this project uses classical ML methods and a simple web app interface for practical use in Ethiopia and beyond.

---

## 🌟 Features

- 🔤 Amharic text preprocessing and cleaning
- 📊 TF-IDF feature extraction
- 🤖 Sentiment classification using Logistic Regression
- 🧪 Model evaluation and performance reports
- 🌐 Streamlit web app for real-time sentiment analysis
- 📁 Modular code and organized project structure

---

## 📂 Project Structure

```

qal-analyzer/
│
├── data/               # Raw and cleaned datasets
├── notebooks/          # Jupyter notebooks (EDA, training, evaluation)
├── models/             # Saved models (TF-IDF, Logistic Regression)
├── app/                # Streamlit app
├── utils/              # Text cleaning utilities
├── README.md
├── requirements.txt
└── LICENSE

````

---

## 📊 Dataset

We used the publicly available [LiyaSileshi Amharic Sentiment Dataset](https://github.com/liyaSileshi/amharic-sentiment-analysis), containing over 9,000 Amharic tweets labeled as:
- Positive
- Negative
- Neutral

---

## 🛠️ Tech Stack

| Purpose              | Tool/Library            |
|----------------------|-------------------------|
| Programming Language | Python                  |
| ML Framework         | Scikit-learn            |
| Web App              | Streamlit               |
| Visualization        | Matplotlib, Seaborn     |
| Text Processing      | NLTK, Regex             |
| Deployment Ready     | GitHub + Streamlit      |

---

## 🚀 How to Run Locally

1. **Clone the repo:**
```bash
git clone https://github.com/your-username/qal-analyzer.git
cd qal-analyzer
````

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the app:**

```bash
streamlit run app/streamlit_app.py
```

---

## 🧠 Model Overview

* **Vectorizer:** TF-IDF
* **Classifier:** Logistic Regression (baseline)
* **Evaluation:** Confusion matrix, precision, recall, F1-score

---

## 📸 Screenshots

*Add screenshots of your Streamlit app or example predictions here*

---

## 📚 Learn More

* [TF-IDF Explained](https://monkeylearn.com/blog/what-is-tf-idf/)
* [Scikit-learn Text Classification Guide](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
* [Streamlit Docs](https://docs.streamlit.io/)

---

## 🤝 Contributing

Contributions are welcome — especially around improving preprocessing, adding a transformer-based model, or expanding the dataset.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🧠 Author

Created by **Yostina Abera**
Inspired by the need for Amharic NLP tools.

---

> *“ቃል ኃይል ነው።”* — **Words have power.**

```
# Amharic-sentiment-analysis-with-feedback
# Amharic-sentiment-analysis-with-feedback
