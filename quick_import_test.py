import os, sys

print("PYTHONPATH OK:", any("Women" in p or "src" in p for p in sys.path))

import streamlit, sklearn, numpy, pandas, yaml, joblib, nltk

print("streamlit", streamlit.__version__)
print("sklearn", sklearn.__version__)
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("yaml", yaml.__version__)
print('joblib', joblib.__version__)
print('nltk', nltk.__version__)


print("Smoke test passed")
