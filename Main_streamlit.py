import streamlit as st
import txtai
import pandas as pd
import numpy as np

@st.cache_resource
def load_embeddings():
    try:
        # Load the embeddings
        embeddings = txtai.Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
        embeddings.load("embeddings.tar.gz")
        
        # Load the titles from the CSV again for retrieval
        df = pd.read_csv("/home/User_Name/Project/Search engine/train.csv")
        np.random.seed(1)
        titles = df.dropna().sample(10000).TITLE.values
        
        return titles, embeddings
    except Exception as e:
        st.error(f"An error occurred while loading embeddings: {e}")
        return None, None

titles, embeddings = load_embeddings()

st.title("Tech-Product Search Engine")

query = st.text_input("Enter search query", '')

if st.button("Search"):
    if query:
        result = embeddings.search(query, limit=5)
        actual_results = [titles[x[0]] for x in result]
        for res in actual_results:
            st.write(res)
    else:
        st.warning("Please enter a search query")
