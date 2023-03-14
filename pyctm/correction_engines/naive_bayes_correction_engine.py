import numpy as np

from sklearn.naive_bayes import GaussianNB

class NaiveBayesCorrectorEngine(object):

    def __init__(self, dictionary):
        self.model_words = GaussianNB()
        self.model_values = GaussianNB()
        self.model_base = GaussianNB()
        self.model_signal = GaussianNB()

        self.trained = False

        self.dictionary = dictionary

        self.fit(dictionary)
    
    def fit(self, dictionary):
        x_words = np.array(list(dictionary.words.values()))
        y_words = np.arange(len(dictionary.words))

        self.model_words.fit(x_words, y_words)

        x_values = np.array(list(dictionary.values.values()))
        y_values = np.arange(len(dictionary.values))

        self.model_values.fit(x_values, y_values)

        x_base = np.array(list(dictionary.baseValues.values()))
        y_base = np.arange(len(dictionary.baseValues))

        self.model_base.fit(x_base, y_base)

        x_signal = np.array(list(dictionary.signalValues.values()))
        y_signal = np.arange(len(dictionary.signalValues))

        self.model_signal.fit(x_signal, y_signal)

        self.trained = True
    
    def make_word_correction(self, word_sdr):
        if self.__is_trained():
            word_class = self.model_words.predict([np.array(word_sdr)])
            word = list(self.dictionary.words.values())[word_class[0]]
            
            return word
    
    def make_value_correction(self, value_sdr):
        if self.__is_trained():
            value_class = self.model_values.predict([np.array(value_sdr)])
            value = list(self.dictionary.values.values())[value_class[0]]
            
            return value

    def make_base_correction(self, base_sdr):
        if self.__is_trained():
            base_class = self.model_base.predict([np.array(base_sdr)])
            base = list(self.dictionary.baseValues.values())[base_class[0]]
            
            return base

    def make_signal_correction(self, signal_sdr):
        if self.__is_trained():
            signal_class = self.model_signal.predict([np.array(signal_sdr)])
            signal = list(self.dictionary.signalValues.values())[signal_class[0]]
            
            return signal

    def __is_trained(self):
        return self.trained