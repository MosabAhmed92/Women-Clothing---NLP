import os
import pickle
import joblib



def save_model(model, save_dir, name):
        # Ensuring that the save_dir exist 
        os.makedirs(save_dir, exist_ok=True)
        
        file_path = os.path.join(save_dir, f"{name}.pkl")

        with open(file_path, "wb") as f:
            joblib.dump(model, f)

        print('The model Has been Saved Sucessfuly')
        return file_path



def load_model (path, name):
      # Ensuring the path exist
        file_path = os.path.join(path, f"{name}.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No model found at {file_path}")
        with open(file_path, "rb") as f:
            model = joblib.load(f)
        return model
