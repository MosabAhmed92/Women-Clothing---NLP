# quick_model_test.py
import joblib, sys
PIPE_PATH = "artifacts/sentiment_pipe.joblib"

try:
    pipe = joblib.load(PIPE_PATH)
except Exception as e:
    print(f"[FAIL] Could not load pipeline at {PIPE_PATH}: {e}")
    sys.exit(1)

if not hasattr(pipe.named_steps["tfidf"], "idf_"):
    print("[FAIL] Vectorizer inside pipeline is NOT fitted (missing idf_).")
    sys.exit(2)

try:
    y = pipe.predict(["This fabric is soft but sizing is off."])[0]
    print(f"[PASS] Predict works. Example label: {y}")
except Exception as e:
    print(f"[FAIL] Predict raised: {type(e).__name__}: {e}")
    sys.exit(3)