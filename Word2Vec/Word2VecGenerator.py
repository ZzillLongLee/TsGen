from gensim.models import Word2Vec


class Word2VecGenerator():
    def __init__(self):
        self.__model = None

    def generateModel(self, sentences=None, vector_size=20, window=2, min_count=1, Type=None):
        self.__model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=min_count)
        self.__model.save("Outcome/Models/" + Type + ".model")

    def getVectorValue(self, word=None):
        vector = self.__model.wv[word]
        return vector

    def loadModel(self, modelPath):
        self.__model = Word2Vec.load(modelPath)
        return self.__model
