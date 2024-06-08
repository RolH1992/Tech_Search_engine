import txtai
import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv("/home/user_name/search engine/file.csv")

# Set random seed for reproducibility
np.random.seed(1)

# Sample titles from the dataframe
titles = df.dropna().sample(10000).TITLE.values

# Create an embeddings object
embeddings = txtai.Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})

# Index the titles
embeddings.index([(i, title, None) for i, title in enumerate(titles)])

# Save the indexed embeddings
embeddings.save("embeddings.tar.gz")

# Perform a search query
query = 'protector for cam'
result = embeddings.search(query, limit=5)

# Print search results
print(result)

# Get the actual results from the titles using the indices from the search result
actual_results = [titles[x[0]] for x in result]

# Print the actual titles of the search results
print(actual_results)
