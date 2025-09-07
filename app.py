import os
import pandas as pd
from flask import Flask, render_template, request, send_file
from transformers import pipeline
from langdetect import detect
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Initialize Flask
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load sentiment model
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def analyze_text(text):
    """Detect language, translate if needed, analyze sentiment and add polarity"""
    try:
        lang = detect(text)
    except:
        lang = "unknown"

    text_en = GoogleTranslator(source='auto', target='en').translate(text) if lang != "en" and lang != "unknown" else text
    res = classifier(text_en)[0]

    # Determine sentiment polarity based on star rating
    rating = int(res['label'][0])  # Extract numeric rating from "4 stars"
    if rating >= 4:
        polarity = "Positive"
    elif rating <= 2:
        polarity = "Negative"
    else:
        polarity = "Neutral"

    return {
        "original_text": text,
        "sentiment": f"{res['label']} ({polarity})",
        "confidence": round(res['score'], 2),
    }

@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    chart = None
    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        if user_input:
            texts = user_input.split("\n")
            results = [analyze_text(t) for t in texts if t.strip()]

            # Sentiment distribution chart
            sentiments = [r["sentiment"].split()[0] for r in results]  # Use star rating only
            dist = pd.Series(sentiments).value_counts().to_dict()
            plt.figure(figsize=(6,4))
            plt.bar(dist.keys(), dist.values(), color='skyblue')
            plt.title("Sentiment Distribution")
            plt.xlabel("Sentiment")
            plt.ylabel("Count")
            chart_path = os.path.join("static", "chart.png")
            plt.savefig(chart_path)
            plt.close()
            chart = "chart.png"
    return render_template("index.html", results=results, chart=chart)

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    if file:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        df = pd.read_csv(path)
        if "review" not in df.columns:
            return "CSV must have a 'review' column"
        df["analysis"] = df["review"].apply(lambda x: analyze_text(str(x)))
        df_results = pd.DataFrame(df["analysis"].tolist())
        output_path = os.path.join(UPLOAD_FOLDER, "results.csv")
        df_results.to_csv(output_path, index=False)
        return send_file(output_path, as_attachment=True)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

