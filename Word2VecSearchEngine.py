from gensim.models import Word2Vec

class Word2VecSearchEngine:

    def __init__(self, model_path):
        from gensim.models import Word2Vec
        self.model = Word2Vec.load(model_path)

    