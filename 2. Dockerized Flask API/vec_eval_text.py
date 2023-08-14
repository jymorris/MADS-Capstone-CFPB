"""
Description:
This code provides an API service using Flask. 
The purpose of this API is to evaluate and classify customer complaints into different product types and issues using pre-trained machine learning models.

When a user submits a complaint via the API endpoint /complaint_eval/, the complaint text is first normalized and tokenized. 
Words with repeating characters are cleaned up, and numerical tokens are removed. 
The cleaned text is then converted into a 100-dimensional vector using GloVe embeddings. 
This vector is then fed into two separate models - one to predict the product category and the other to predict the issue. 
The results are returned to the user as a JSON response.

The code also provides a couple of basic endpoints (/ and /hello) for testing and basic interaction.
"""
from flask import Flask, jsonify
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
import re
import pickle


# Load pre-trained word embeddings from GloVe (100-dimensional version)
glove_6B_100D_txt_file = "glove_file, something like glove.6B.100d.txt"
embeddings_dict_6B_100D = {}
with open(glove_6B_100D_txt_file, 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = ' '.join(values[:-100]).lower().strip()
        vector = np.asarray(values[-100:], "float32")
        embeddings_dict_6B_100D[word] = vector

# Convert a given text into a 100-dimensional vector using the loaded word embeddings
def vectorize_text_100(text):
    # Get the embeddings for each word, ignore words that aren't in the embeddings dictionary
    vectors = [embeddings_dict_6B_100D.get(word) for word in str(text).split() if word in embeddings_dict_6B_100D]
    
    # Filter out None values
    vectors = [v for v in vectors if v is not None]
    
    # Average the vectors or return a zero-vector if no vectors are found
    if vectors:
        vectorized = np.mean(vectors, axis=0)
    else:
        vectorized = np.zeros(100)
    return vectorized

# Normalize and clean a given text
def text_normalizer(text):
    if text:
        # Tokenize text and retain words like "wasn't" as single tokens
        tokenizer = RegexpTokenizer(r'\b\w[\w\'-]*\w\b|\w')
        words = tokenizer.tokenize(text)
        
        # Remove tokens with repeating characters (e.g., "!!!!!!", "666")
        words = [re.sub(r'(\w)\1{2,}', '', word) if re.search(r'(\w)\1{2,}', word) else word for word in words]
        
        # Convert words to lowercase and strip any white space
        words = [word.lower().strip() for word in words]
        
        # Remove tokens that are just numbers
        words = ['' if word.isdigit() else word for word in words]
        
        # Join words into a cleaned text string
        text = ' '.join([word for word in words if word])
    
    return text

# Load a pickled model from a given filename
def load_model(filename):
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

# Predict using a set of models on a given text vector
def model_prediction(model_dict,vectorized_complaint):
    result_dict = {}
    for key, model in model_dict.items():
        result_dict[key] = model.predict_proba([vectorized_complaint])[0][1]
    return result_dict

# Initialize Flask app
app = Flask(__name__)

# Basic routes for the API
@app.route("/")
def home():
    return "Welcome to the MADS CFPB Complaint Evaluation API!", 200

# Tell the API (Google Cloud Run) to "Get Your Lazy Butt Up And Start Rebuild Yourself For Work"
@app.route("/hello", methods=["GET"])
def hello():
    return "Hello", 200

@app.route("/complaint_eval/<path:complaint>", methods=["GET"])
def eval_complaint(complaint):
    eval = {}
    
    # Normalize and vectorize the complaint text
    vectorized_complaint = vectorize_text_100(text_normalizer(complaint))

    product_clf_pkl = "your model.pkl"
    issue_clf_pkl = "your model.pkl"
    
    # Predict the "Product" and "Issue" using pre-trained models
    eval["Product"] = model_prediction(load_model("product_clf_pkl"),vectorized_complaint)
    eval["Issue"] = model_prediction(load_model("issue_clf_pkl.pkl"),vectorized_complaint)

    return jsonify(eval)

# Start the Flask application on port 5000
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
