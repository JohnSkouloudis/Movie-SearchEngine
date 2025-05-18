import pandas as pd
import nltk
import re
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from gensim.models import KeyedVectors
import faiss


class Word2VecSearchEngine:

    def __init__(self, model_path,df,faiss_index=None):
        
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.index = faiss_index
        self.df = df

        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('wordnet')



    # Preprocess the plot text
    def preprocess_text(self,plot ,lemmatize = True,return_tokens = False):



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

        if return_tokens:
            return tokens    

        return ' '.join(tokens)
    

    
    # Create the vector for the plot
    def get_plot_vector(self,plot):
        tokens = self.preprocess_text(plot, return_tokens=True)
        
        if not tokens:
            return np.zeros(self.model.vector_size)
        
        vectors = [self.model[token] for token in tokens if token in self.model]

        if not vectors:
            return np.zeros(self.model.vector_size)

        plot_vector = np.mean(vectors, axis=0)
        return plot_vector
    
    # Create the FAISS index
    def create_faiss_index(self):

        if self.index is None:
            print("Creating faiss index:")
            self.index = faiss.IndexFlatL2(self.model.vector_size)

            for plot in self.df['Plot']:
                plot_vector = self.get_plot_vector(plot)
                plot_vector = plot_vector.astype('float32')
                self.index.add(plot_vector.reshape(1, -1))
    
            print(f"Added plot vector to index: {self.index.ntotal} vectors")
            print(f"Index dimension: {self.index.d}")
            
        else:
            print("Index has already been initialized or created.")
    
    # Search the FAISS index
    def search_faiss_index(self,query, top_k=10):

        query_vector = self.get_plot_vector(query)
        query_vector = query_vector.astype('float32')
        query_vector = query_vector.reshape(1, -1)
        D, I = self.index.search(query_vector, top_k)
        
        results = []
        for distance, idx in zip(D[0], I[0]):
            results.append((idx, distance))
        
        results.sort(key=lambda x: x[1])

        return results
    
    # returns a list of tuples (doc_id, title, plot, distance)
    def search(self,query, top_k=10):
        self.create_faiss_index()
        answers = self.search_faiss_index(query, top_k)

        results = []
        for doc_id, distance in answers:
            results.append((doc_id,self.df.iloc[doc_id]['Title'], self.df.iloc[doc_id]['Plot'], distance))

        return results
    


# Save the Word2VecSearchEngine object to a pickle file

if __name__ == "__main__":
    
    df = pd.read_csv('wiki_movie_plots_deduped_updated.csv')
    search_engine = Word2VecSearchEngine(model_path='models/GoogleNews-vectors-negative300.bin',df=df)
    search_engine.create_faiss_index()

    results = search_engine.search("jack and the beanstalk", top_k=10)
    for doc_id,title, plot,distance in results:
        print(f"Document ID: {doc_id}, Title: {title}, Distance: {distance:.4f}, Plot: {plot[:50]}...")

    with open('word2vec_search_engine.pkl', 'wb') as f:
        pickle.dump(search_engine, f)

    