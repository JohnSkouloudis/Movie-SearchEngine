from sentence_transformers import SentenceTransformer
import faiss
import torch
import pandas as pd
import numpy as np

class BertSearchEngine:
    def __init__(self, model_path,df,index=None,device='cpu'):
       
       self.device = device 
       self.model = SentenceTransformer(model_path, device=self.device)
       self.df = df
       self.index = index

    def get_bert_embedding(self,plot):

        if not plot:
            return np.zeros(768)
    
        embedding = self.model.encode(plot,batch_size=64)
        return embedding
    
    def create_faiss_index(self):

        if self.index is not None:
            raise ValueError("Index already exists. Please delete the existing index before creating a new one.")
        
        self.index = faiss.IndexFlatIP(768)

        for plot in self.df['Plot']:
            plot_embedding = self.get_bert_embedding(plot,self.model)
            plot_embedding = plot_embedding.astype('float32')
            self.index.add(plot_embedding.reshape(1, -1))

    def search_faiss_index(self,query, top_k=10):

        if self.index is None:
            raise ValueError("Index has not been created yet. Please create the index first.")
        
        query_vector = self.get_bert_embedding(query)
        query_vector = query_vector.astype('float32')
        query_vector = query_vector.reshape(1, -1)

        similarities , indices = self.index.search(query_vector, top_k)

        results = []

        for similarity, index in zip(similarities[0], indices[0]):
            results.append((index, similarity))
            
        results.sort(key=lambda x: x[1], reverse=True)  
        return results
    
    def search(self,query, top_k=10):

        answers = self.search_faiss_index(query, top_k)
        results = []
        for index, similarity in answers:
            results.append((index,similarity,self.df.iloc[index]['Title'], self.df.iloc[index]['Plot']))
        
        return results
        

   