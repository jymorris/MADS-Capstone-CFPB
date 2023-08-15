# Customer Complaints Analysis and API, A UMICH MADS Capstone Project 

This repository contains code and tools for processing and analyzing the CFPB customer complaints dataset, creating a Flask API, and deploying a Streamlit dashboard.

## Directory Structure

1. **Set Up and ML Development**
    - `Processing and Storing Consumer Complaint Data in GCP PostgreSQL.ipynb`
    - `Mark Duplicative Complaints For ML Model Training.ipynb`
    - `Train SVM Binary Classifiers For Customer Complaints.ipynb`
    - `CFPB_complaint_place_holder.csv`

2. **Dockerized Flask API**
    - `Dockerfile`
    - `requirements.txt`
    - `vec_eval_text.py`
    - `your_glove.6B.100d_place_holder.txt`
    - `your_issue_clf_model.pkl`
    - `your_product_clf_model.pkl`

3. **Dockerized CFPB Streamlit Tool**
    - `Dockerfile`
    - `States_shapefile.cpg`
    - `States_shapefile.dbf`
    - `States_shapefile.prj`
    - `States_shapefile.shp`
    - `States_shapefile.shx`
    - `main.py`
    - `requirements.txt`
    - `wrapper_script.sh`

## Setup Instructions

### 0. Install Requirements
* run `pip install -r requirements.txt` to install all needed libraries.

### 1. Set Up and ML Development

1. Download the CFPB customer complaint dataset from [here](https://www.consumerfinance.gov/data-research/consumer-complaints/). Navigate to the "Get the data" section for the CSV file.
2. Download the GloVe pre-trained word vectors from [here](https://nlp.stanford.edu/projects/glove/). We used the Wikipedia versions with 50 and 100 dimensions.

**Important**: Adjust the vectorizing functions in the file if you opt for different GloVe versions.

3. Execute `Processing and Storing Consumer Complaint Data in GCP PostgreSQL .ipynb`:
   - Make sure you update your PostgreSQL credentials within the notebook.
   - This notebook guides you through storing the CFPB data into a PostgreSQL service.

4. Run `Mark Duplicative Complaints For ML Model Training.ipynb` followed by `Train SVM Binary Classifiers For Customer Complaints.ipynb`. 
   - The first script marks records with shared IDs, and the second one uses these IDs to filter duplicative complaints and generate pickled classification models.

### 2. Dockerized Flask API

1. Move the pickled models from the first folder to this directory.
2. Update the model filenames in `vec_eval_text.py`.
3. Copy your choice of GloVe vector file to this directory for text vectorization.

**Testing**:
   - To test the API locally, run the appropriate command for your platform (given in the repository).
   - For Docker testing:
     1. Install Docker Desktop.
     2. Build the Docker image: 
        ```bash
        docker build -t app_name:v1 -f Dockerfile .
        ```
     3. Run the container:
        ```bash
        docker run -p 5000:5000 app_name:v1
        ```
     4. Access the Flask API at `localhost:5000`.

If you're deploying to a cloud platform (like Google Cloud Run), follow their specific deployment instructions.

### 3. Dockerized CFPB Streamlit Tool

Ensure your database connection and API are accessible.

1. Run locally using:
   ```bash
   python3 -m streamlit run main.py --server.port=8501
   ```
2. For Docker:
   - Build and run instructions are similar to the Flask API Docker steps mentioned above.

**Note**: Remember to update any credentials or API links in the code files as needed.

### Additional Resources & Information

- **PostgreSQL Version**: We used version 14. Functionality might differ for other versions.
- Ensure that credentials and links to the API are updated across all notebooks and scripts as required.
- The provided shapefiles are crucial for the geopandas library in the Streamlit tool.

---

We hope this guide helps you navigate and utilize the tools efficiently. Should you encounter any issues, please refer to the respective documentation or seek support from relevant community forums.
