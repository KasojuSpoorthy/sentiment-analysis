# AI Sentiment Analyzer

A sleek and interactive **Sentiment Analysis** web application built with **Flask**, **Hugging Face Transformers**, and **Matplotlib**. This app allows users to analyze customer reviews either by entering text manually or uploading a CSV file. It provides star ratings with positive/negative sentiments and visualizes sentiment distribution with charts.

---

## ✅ Features

✔ Analyze customer reviews using a pre-trained multilingual sentiment model  
✔ Support for manual text entry and CSV uploads  
✔ Detects language and translates non-English text automatically  
✔ Shows star rating with polarity (Positive / Neutral / Negative)  
✔ Generates sentiment distribution charts  
✔ Clean, responsive, and mobile-friendly UI  
✔ Easy deployment on platforms like Render, Heroku, Railway, etc.

---

## ✅ Technologies Used

- **Backend**: Flask  
- **Machine Learning**: `transformers` (Hugging Face) – `nlptown/bert-base-multilingual-uncased-sentiment`  
- **Language Detection**: `langdetect`  
- **Translation**: `deep-translator` (Google Translate)  
- **Data Handling**: Pandas  
- **Visualization**: Matplotlib  
- **Frontend**: HTML, CSS (Google Fonts), Jinja Templates

---

## ✅ Setup Instructions

### 🔹 1. Clone the repository

```bash
git clone https://github.com/yourusername/sentiment_flask_app.git
cd sentiment_flask_app
