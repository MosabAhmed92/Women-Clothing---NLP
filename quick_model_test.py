import os, sys, joblib

PROJ_ROOT = os.path.abspath('.')
ART_DIR = os.path.join(PROJ_ROOT, 'artifacts')
pipe_path = os.path.join(ART_DIR, 'final_sentiment_pipe.pkl')

pipe = joblib.load(pipe_path)

print('Loaded : ', type(pipe))

sample = ["Love the fit and the fabric, would buy again"]
print('the sample is : ', sample[0])
print('sentiment prediction is : ', pipe.predict(sample)[0])
print("model Loaded and prediction made on a sample")
