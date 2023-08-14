#!/bin/bash
mkdir -p ~/.streamlit/
echo "[general]" > ~/.streamlit/credentials.toml
echo "email = \"ricardojackwu@gmail.com\"" >> ~/.streamlit/credentials.toml
python3 -m streamlit run main.py --server.port=8501
