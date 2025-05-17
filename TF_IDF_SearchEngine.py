from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pandas as pd
import nltk
import re
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity


class TF_IDF_SearchEngine:

    def __init__(self,df,vectorizer=None,tfidf_matrix=None,inverted_index=None):
        self.df = df
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.inverted_index = inverted_index
        self.features = self.vectorizer.get_feature_names_out() if vectorizer else None

        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('wordnet')


    #Preprocess the text by removing new lines, extra spaces, and converting to lowercase and lemmatizing
    def preprocess_text(self,plot,lemmatize = True):
        plot = re.sub(r'[\r\n]+', ' ', plot)  
        plot = re.sub(r'\s+', ' ', plot) 
        plot = re.sub(r"'s\b", "", plot) 

        plot = plot.lower()

        tokens = word_tokenize(plot)

        stop_words = set(stopwords.words('english'))
        puncts = set(string.punctuation)
        puncts.update(["``", "''", '"', "'", "`","--"])
    

        tokens = [word for word in tokens if word not in stop_words]

        tokens = [word for word in tokens if word not in puncts]

        tokens = [word for word in tokens if not word.isnumeric()]

        if lemmatize:
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]

        return ' '.join(tokens)
    
    # Create the TF-IDF vectorizer and fit it to the data
    def create_tfidf_vectorizer(self):

        if self.vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(preprocessor=self.preprocess_text,min_df=2,max_df=0.85)

            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['Plot'])

            self.tfidf_features = self.vectorizer.get_feature_names_out()
        else:
            raise ValueError("TF-IDF vectorizer is already created. Please create it only once.")
        
    
    # Create the Inverted index from the TF-IDF matrix
    # The inverted index is a dictionary where the keys are terms and the values are lists of tuples (doc_id, tf_idf_score) sorted by tf_idf_score
    def create_inverted_index(self):

        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix is not created. Please create it first.")
        if self.features is None:
            raise ValueError("Features are not created. Please create them first.")
        
        self.inverted_index = defaultdict(list)

        for term_idx , term in enumerate(self.features):
            for doc_idx in self.tfidf_matrix[:, term_idx].nonzero()[0]:

                tf_idf_score = self.tfidf_matrix[doc_idx, term_idx] 
                self.inverted_index[term].append((doc_idx, tf_idf_score))

        for term in self.inverted_index:
            self.inverted_index[term].sort(key=lambda x: x[1], reverse=True)    

    # Search the inverted index for a given query
    def search_inverted_index(self,query,top_k=10,depth=None,term_count_score=0.0):

        if self.inverted_index is None:
            raise ValueError("Inverted index is not created. Please create it first.")
        
        query_vector = self.vectorizer.transform([query])

        query_terms = [self.features[i] for i in query_vector.nonzero()[1]]
        term_count = len(query_terms)
        
        matching_docs = defaultdict(int)

        for term in query_terms:
            if term in self.inverted_index:
                if depth is not None:
                    matches = self.inverted_index[term][:depth]
                else:
                    matches = self.inverted_index[term]
                for doc_id ,score in matches:
                        matching_docs[doc_id] += 1


        matching_docs_matrix = self.tfidf_matrix[list(matching_docs.keys())]

        similarities = cosine_similarity(query_vector, matching_docs_matrix).flatten()


        results =[]
        for i, doc_id in enumerate(list(matching_docs.keys())):
            score = similarities[i]  + ( term_count_score * ( matching_docs[doc_id] / term_count ) )
            results.append((doc_id,score,similarities[i]))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    # Use the search_inverted_index method to search for a query and return the title and plot of the top_k results
    def search(self,query,top_k=10,depth=None,term_count_score=0.0):

       searches = self.search_inverted_index(query,top_k,depth,term_count_score)
       results = []
       for doc_id, score, similarity in searches:
            results.append((doc_id,self.df.iloc[doc_id]['Title'], self.df.iloc[doc_id]['Plot'], score, similarity))
        
       return results
