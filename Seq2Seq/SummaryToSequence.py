from keras_preprocessing.text import Tokenizer
import numpy as np
import pickle

class SummaryToSequence():

    def __init__(self):
        self.__tar_tokenizer = None

    def convertAsSequence(self, decoder_input_train, decoder_target_train):
        self.__tar_tokenizer = Tokenizer()
        self.__tar_tokenizer.fit_on_texts(decoder_input_train)
        self.__tar_tokenizer.fit_on_texts(decoder_target_train)
        decoder_input_train = self.__tar_tokenizer.texts_to_sequences(decoder_input_train)
        decoder_target_train = self.__tar_tokenizer.texts_to_sequences(decoder_target_train)
        return decoder_input_train, decoder_target_train

    def getVocabularySize(self):
        return len(self.__tar_tokenizer.word_index) + 1


    def seq2text(self, input_seq):
        sentence = ''
        for i in input_seq:
            if ((i != 0 and i != self.__tar_word_to_index['sostoken']) and i != self.__tar_word_to_index['eostoken']):
                sentence = sentence + self.__tar_index_to_word[i] + ' '
        return sentence

    
    def getTokenizer(self):
        return self.__tar_tokenizer

    def saveDecoder_Tokenizer(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self.__tar_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def loadDecoder_Tokenizer(self, path):
        with open(path, 'rb') as handle:
            self.__tar_tokenizer = pickle.load(handle)
