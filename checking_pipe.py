import joblib, hashlib, pathlib

p = pathlib.Path("artifacts/final_sentiment_pipe.pkl")
pipe = joblib.load(p)

print("Has tfidf:", "tfidf" in pipe.named_steps)
tfidf = pipe.named_steps["tfidf"]
print("Has vocabulary_:", hasattr(tfidf, "vocabulary_"))
print("Has idf_:", hasattr(tfidf, "idf_"))
print("Predict works:", pipe.predict(["this dress is great"]))

sha = hashlib.sha256(p.read_bytes()).hexdigest()
print("Local sha256 first 16:", sha[:16])