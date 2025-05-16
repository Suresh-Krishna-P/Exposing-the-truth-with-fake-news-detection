from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
import os
import zipfile

app = Flask(__name__)

# Paths for saving model, vectorizer, and dataset
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
ZIP_PATH = "data/dataset.csv.zip"  # Path to your zipped CSV
CSV_PATH = "data/dataset.csv"      # Path to unzipped CSV

# Load and preprocess dataset
def load_dataset():
    try:
        # Ensure data/ directory exists
        os.makedirs("data", exist_ok=True)
        
        # Unzip the dataset if CSV doesn't exist
        if not os.path.exists(CSV_PATH):
            if os.path.exists(ZIP_PATH):
                with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                    zip_ref.extractall("data/")
                print(f"Unzipped {ZIP_PATH} to {CSV_PATH}")
            else:
                raise FileNotFoundError(f"Zip file not found at {ZIP_PATH}")
        
        # Load the dataset
        df = pd.read_csv(CSV_PATH)
        
        # Basic preprocessing
        df = df.dropna(subset=['text', 'label'])  # Remove rows with missing values
        df['label'] = df['label'].astype(int)     # Ensure labels are 0 or 1
        df['text'] = df['text'].str.strip()       # Remove extra whitespace
        
        return df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")

# Train the model
def train_model():
    try:
        df = load_dataset()
        X = df['text']
        y = df['label']
        
        # Vectorize text using TF-IDF
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_vec = vectorizer.fit_transform(X)
        
        # Train Logistic Regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_vec, y)
        
        # Save model and vectorizer
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(VECTORIZER_PATH, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        return {"status": "success", "message": "Model trained successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Load trained model and vectorizer
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    return None, None

# Flask route to train the model
@app.route('/train', methods=['POST'])
def train():
    result = train_model()
    return jsonify(result)

# Flask route to predict news authenticity
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        news_text = data.get('text', '')
        
        if not news_text:
            return jsonify({"status": "error", "message": "No text provided"})
        
        model, vectorizer = load_model()
        if model is None or vectorizer is None:
            return jsonify({"status": "error", "message": "Model not trained"})
        
        # Vectorize input text
        X_vec = vectorizer.transform([news_text])
        
        # Predict
        prediction = model.predict(X_vec)[0]
        confidence = model.predict_proba(X_vec)[0][prediction]
        
        label = "True" if prediction == 1 else "Fake"
        return jsonify({
            "status": "success",
            "label": label,
            "confidence": round(confidence * 100, 2)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
