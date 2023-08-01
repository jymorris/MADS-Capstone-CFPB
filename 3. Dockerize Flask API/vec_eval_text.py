from flask import Flask, jsonify
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
import re
import pickle


# some preperation
embeddings_dict_6B_100D = {}
with open("glove.6B.100d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = ' '.join(values[:-100]).lower().strip()
        vector = np.asarray(values[-100:], "float32")
        embeddings_dict_6B_100D[word] = vector
        
def vectorize_text_100(text):
    vectors = [embeddings_dict_6B_100D.get(word) for word in str(text).split() if word in embeddings_dict_6B_100D]
    vectors = [v for v in vectors if v is not None]  # remove any None values
    if vectors:
        vectorized = np.mean(vectors, axis=0)
    else:
        vectorized = np.zeros(100)  # if there are no vectors, return a zero-vector
    return vectorized

def text_normalizer(text):
    if text:
        # Use NLTK RegexpTokenizer for tokenization. 
        # This tokenizer splits the text by white space and also keeps tokens like "wasn't" and "don't".
        tokenizer = RegexpTokenizer(r'\b\w[\w\'-]*\w\b|\w')
        words = tokenizer.tokenize(text)

        # Clean up any token with repeating characters like '666', 'aaa', '!!!!!!', substitute them with empty string ''.
        # This includes 'XXXX' maskings in the text created by CFPB.
        words = [re.sub(r'(\w)\1{2,}', '', word) if re.search(r'(\w)\1{2,}', word) else word for word in words]

        # Convert to lowercase and remove punctuations.
        words = [word.lower().strip() for word in words]

        # Substitute the tokens with "" where they are just numbers.
        words = ['' if word.isdigit() else word for word in words]

        # Join the words back into a single string.
        text = ' '.join([word for word in words if word])
    
    return text

def load_model(filename):
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

def model_prediction(model_dict,vectorized_complaint):
    result_dict = {}
    for key, model in model_dict.items():
        result_dict[key] = model.predict_proba([vectorized_complaint])[0][1]
    # returns a probability in float
    return result_dict

# Initialise flask app
app = Flask(__name__)

@app.route("/")
def home():
    """Method 0: Return a welcome message"""
    return "Welcome to the MADS CFPB Complaint Evaluation API!", 200

@app.route("/hello", methods=["GET"])
def hello():
    """Method 1: Return a simple hello"""
    return "Hello", 200

# @app.route("/hello/<my_name>", methods=["GET"])
# def hello_name(my_name):
#     """Method 2: Return hello with name, given in URL"""
#     return f"Hello from URL, {my_name}", 200

@app.route("/complaint_eval/<path:complaint>", methods=["GET"])
def eval_complaint(complaint):
    eval = {}
    vectorized_complaint = vectorize_text_100(text_normalizer(complaint))
    eval["Product"] = model_prediction(load_model("_product_svm_long_models_dict.pkl"),vectorized_complaint)

    eval["Issue"] = model_prediction(load_model("_issue_svm_long_models_dict.pkl"),vectorized_complaint)

    # use jsonify to return the dictionary as a JSON response
    return jsonify(eval)
# gcloud init
# gcloud auth login
# gcloud auth configure-docker us-central1-docker.pkg.dev
# docker-credential-gcloud list
#  docker build -t flaskapp_cr:v1 -f Dockerfile .
#  docker tag flaskapp_cr:v1 us-central1-docker.pkg.dev/cfpb-391621/cfpb-docker-repo/flaskapp_cr:v1
#  docker push us-central1-docker.pkg.dev/cfpb-391621/cfpb-docker-repo/flaskapp_cr:v1
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

