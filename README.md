# Intelligent Sentiment Detection Engine 

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Handling-lightblue?style=flat-square)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-orange?style=flat-square)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-green?style=flat-square)](https://seaborn.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-lightgreen?style=flat-square)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-lightgrey?style=flat-square)](https://www.nltk.org/)
[![SpaCy](https://img.shields.io/badge/SpaCy-NLP-darkgreen?style=flat-square)](https://spacy.io/)
[![Transformers](https://img.shields.io/badge/Transformers-BERT-lightblue?style=flat-square)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)


---

## 1. Project Overview

The **Intelligent Sentiment Detection Engine** leverages **Natural Language Processing (NLP)** and **Machine Learning/Deep Learning** to automatically analyze **user reviews of ChatGPT**. By classifying feedback into **Positive, Neutral, and Negative** categories, it provides actionable insights to improve **user experience**, **product features**, and **brand reputation**.

The project combines **data preprocessing**, **EDA**, **modeling**, and **interactive dashboards** (Streamlit + Flask) for real-time insights.

---

## 2. Problem Statement

Analyzing thousands of user reviews manually is **time-consuming and error-prone**.  

This project automates sentiment detection to:

- Understand **overall user satisfaction**.
- Detect recurring **complaints or pain points**.
- Help product and marketing teams make **data-driven decisions**.

---

## 3. Business Use Cases

- **Customer Feedback Analysis:** Identify common pain points and areas for improvement.  
- **Brand Reputation Management:** Track sentiment trends to evaluate public perception.  
- **Feature Enhancement:** Focus on product features with negative or neutral feedback.  
- **Automated Customer Support:** Prioritize complaints for faster resolution.  
- **Marketing Strategy Optimization:** Align campaigns with sentiment trends.

---

## 4. Dataset Description

**Dataset Name:** `chatgpt_reviews.csv`  

| Column Name         | Description                                |
| ------------------- | ------------------------------------------ |
| `date`              | Date when the review was submitted         |
| `title`             | Short summary of the review                |
| `review`            | Full user feedback text                    |
| `rating`            | Numerical rating (1–5 stars)               |
| `username`          | Anonymized reviewer name                   |
| `helpful_votes`     | Number of helpful votes                    |
| `review_length`     | Character length of review                 |
| `platform`          | Web or Mobile                              |
| `language`          | ISO code of review language                |
| `location`          | Reviewer’s country                         |
| `version`           | ChatGPT version (e.g., 3.5, 4.0)           |
| `verified_purchase` | Indicates if user is a verified subscriber |

---

## 5. Data Preprocessing

Steps followed:

I. Removed punctuation, special symbols, and emojis.  
II. Tokenized sentences into words.  
III. Lemmatized words to their base form.  
IV. Removed stopwords for noise reduction.  
V. Handled missing and duplicate entries.  
VI. Balanced the dataset using SMOTE/undersampling techniques.  

---

## 6. Technologies & Libraries

- **Languages:** Python  
- **Frameworks:** Streamlit, Flask  
- **Libraries:**  
  - `pandas`, `numpy` (Data Handling)  
  - `nltk`, `spacy` (Text Preprocessing)  
  - `scikit-learn` (ML Models)  
  - `matplotlib`, `seaborn`, `wordcloud` (Visualization)  
  - `transformers`, `torch` (BERT / DL Models)  

## 7. Exploratory Data Analysis (EDA)

### 1. Sentiment Overview

ChatGPT Review Sentiment Analysis
<img width="1440" height="800" alt="ChatGPT Review Sentiment Analysis" src="https://github.com/user-attachments/assets/06b6bbe8-1dad-42d0-99db-e71915ad0531" />

Analyze Sentiment

<img width="1440" height="801" alt="Analyze Sentiment" src="https://github.com/user-attachments/assets/df6170aa-04d5-469d-8b06-60f1347e2bae" />


### 2. Rating Distribution
Rating Analysis
<img width="1440" height="833" alt="Images1" src="https://github.com/user-attachments/assets/2eade5fb-f4e1-400a-bbed-e271c1a8243b" />
<img width="1440" height="828" alt="Images2" src="https://github.com/user-attachments/assets/ceaac955-c7b7-47af-8ff1-2e8278e3689e" />




### 3. Platform & Language Insights
Platform and Language Analysis

<img width="1433" height="526" alt="Image1" src="https://github.com/user-attachments/assets/541f2f47-b996-403c-833d-437d8a703577" />
<img width="1435" height="762" alt="Image2" src="https://github.com/user-attachments/assets/5ab19ff6-aeb8-4460-ac67-4f0a1657ef26" />



### 4. Sentiment Distribution (Detailed)
Sentiment Distribution
<img width="1438" height="800" alt="Sentiment Distribution" src="https://github.com/user-attachments/assets/973e0225-cce3-474d-bd88-f7ef37d25ad4" />


---

## 8. Model Building

| Model               | Type        | Description                                          |
| ------------------- | ----------- | ---------------------------------------------------- |
| Logistic Regression | ML          | Baseline classifier using TF-IDF features           |
| Naïve Bayes         | ML          | Fast, probabilistic model for text                  |
| Random Forest       | ML          | Ensemble method for improved accuracy               |
| LSTM                | DL          | Captures long-term text dependencies                |

**Confusion Matrix:**  
<img width="800" height="600" alt="confusion_matrix copy" src="https://github.com/user-attachments/assets/b8dbc066-1321-4acd-9af0-60cc247ce0e7" />


---

## 9. Outcomes / Results

This section highlights the key findings and results obtained from EDA, modeling, and sentiment analysis.

### 1. Sentiment Distribution
- Majority of reviews are **Positive (4–5 stars)**.
- Neutral and negative reviews are fewer but highlight improvement areas.  


### 2. Rating vs Sentiment Analysis
- Not all 1-star reviews are negative; some 3-star reviews contain neutral sentiment.
- Positive correlation observed between **verified users** and higher ratings.  

### 3. Keyword Analysis
- Positive reviews frequently mention: `"helpful"`, `"easy"`, `"fast"`, `"accurate"`.
- Negative reviews frequently mention: `"slow"`, `"bug"`, `"error"`, `"confusing"`.
- 
### 4. Platform & Version Insights
- Web platform reviews slightly better than Mobile.
- Newer versions of ChatGPT (4.0+) have higher average sentiment.  


### 5. Review Length & Helpfulness
- Longer reviews tend to be more detailed; negative reviews are slightly longer.
- Reviews with **high helpful votes** mostly align with extreme sentiments (very positive or very negative).  

### 6. Model Performance Outcome
- **Best model:** BERT Transformer achieved highest F1-score and accuracy.
- Confusion matrix shows most misclassifications in the **Neutral** class.  

### 7. Actionable Insights
- Focus on improving **response speed** and **accuracy** to reduce negative feedback.
- Encourage **verified users** to post feedback for better quality insights.
- Optimize **mobile experience** to match Web platform satisfaction.



## 10. Features

✅ Sentiment Classification (Positive / Neutral / Negative)  
✅ Word Cloud Generation  
✅ Platform-wise Analysis (Web vs Mobile)  
✅ Regional Sentiment Insights  
✅ Verified vs Non-Verified User Analysis  
✅ ChatGPT Version Sentiment Trends  

---

## 11. Future Work

- Integrate **topic modeling (LDA/BERT)** for negative feedback analysis  
- Implement **multilingual support** for global reviews  
- Deploy model API using **AWS Lambda / EC2**  
- Add **real-time review monitoring**  
- Use **Explainable AI (SHAP/LIME)** for interpretability  

---

##  12. Key Insights

- Majority of users gave **4–5 star reviews**, showing high satisfaction  
- Negative reviews often mentioned **response speed** and **accuracy**  
- Verified users tend to post **more positive feedback**  
- Web platform received slightly better ratings than Mobile  
- Sentiment trends improved over time with newer versions (4.0+)  

---

### 13. Conclusion

This project demonstrates how Natural Language Processing (NLP) and Machine Learning/Deep Learning can analyze customer feedback at scale. By classifying sentiments into **positive, neutral, and negative categories**, it provides valuable insights into **user satisfaction, pain points, and feature performance**. The results help product teams make **data-driven decisions** and improve overall user experience. Furthermore, deployment through interactive dashboards and APIs ensures that insights are **accessible in real-time**, enabling continuous monitoring and improvement of ChatGPT’s platform.


---


---

---

##  14. How to Run Locally

```bash
# Clone the repository
git clone https://github.com/PrashantKumar39/Intelligent-Sentiment-Detection-Engine.git
cd Intelligent-Sentiment-Detection-Engine

# Install dependencies
pip install -r requirements.txt

# Run Streamlit Dashboard
streamlit run streamlit_app.py

# (Optional) Run Flask API
python app.py
