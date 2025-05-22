# Movie Search Engine

A search engine for movie plots developed as part of the Information Retrieval & Natural Language Processing course.

## Table of Contents

- [About](#about)  
- [Dataset](#dataset)  
- [Features](#features)  
- [Tech Stack](#tech-stack)
- [Tasks](#tasks)
- [Directory Structure](#directory-structure)  
- [Installation](#installation)  


## About

This project implements multiple IR and NLP techniques to build a movie-plot search engine. It demonstrates:

- **TF–IDF** vector space search  
- **Word2Vec** semantic similarity search  
- **BERT** embeddings for deep semantic retrieval  

Originally created for an IR & NLP university course, it showcases end‑to‑end data preprocessing, indexing, and querying pipelines. 

## Dataset

Movie plots are sourced from the [Wikipedia Movie Plots dataset on Kaggle](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots).  
You can download it directly and place the CSV file in the project root. 

## Features

- **Boolean & ranked retrieval** using TF–IDF  
- **Semantic search** with Word2Vec embeddings  
- **Contextual search** leveraging BERT-based sentence embeddings  
- **Jupyter notebook** walkthrough for exploration and evaluation  

## Tech Stack

- Python 3.x  
- Jupyter Notebook  
- Libraries: `scikit-learn`, `gensim`, `transformers`, `pandas`, `numpy` ,`faiss`,`os`,`time`,`re`,`nltk`,`pickle`

## Tasks

- Define text representations (e.g., TF-IDF, Word2Vec, BERT)
- Implement index mechanism to store and access vectorized documents efficiently
- Implement a retrieval and ranking mechanism for query results
- Evaluate the time taken to answer 30 queries
- Evaluate the relevance of the top 10 results returned for each query

## Directory Structure

<pre>
Movie-SearchEngine/ 
 ├── NLP_SearchEngine.ipynb  : Notebook that explains the steps for each task
 ├── TF_IDF_SearchEngine.py  : Implementation of TF_IDF_SearchEngine class
 ├── Word2VecSearchEngine.py : Implementation of Word2VecSearchEngine class
 ├── BertSearchEngine.py     : Implementation of BertSearchEngine class
 ├── requirements.txt        : libraries used in the project
 ├── README.md 
 └── .gitignore </pre>

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/JohnSkouloudis/Movie-SearchEngine.git
   cd Movie-SearchEngine
   ```

2.  **Set up a virtual environment (optional but recommended)**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. **Install dependencies** 
    ```bash
    pip install -r requirements.txt
    ```


