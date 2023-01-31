import numpy as np

from sklearn.naive_bayes import GaussianNB

class NaiveBayesCorrectorEngine:

    def __init__(self, dictionary, values):
        self.model_dictionary = GaussianNB()
        self.model_values = GaussianNB()
        self.model_dictionary_trained = False
        self.model_values_trained = False
        self.dictionary = dictionary
        self.values = values

        self.fit(dictionary, values)
    
    def fit(self, dictionary, values):
        x_dictionary = np.array(list(dictionary.values()))
        y_dictionary = np.arange(len(dictionary))

        self.model_dictionary.fit(x_dictionary, y_dictionary)
        self.model_dictionary_trained = True

        x_values = np.array(list(values.values()))
        y_values = np.arange(len(values))

        self.model_values.fit(x_values, y_values)
        self.model_values_trained = True

    def clear_dictionary_model(self):
        self.model_dictionary = GaussianNB()
        self.model_dictionary_trained = False

    def clear_values_model(self):
        self.model_values = GaussianNB()
        self.model_values_trained = False
    
    def make_word_correction(self, word_sdr):
        if self.__is_trained():
            word_class = self.model_dictionary.predict([np.array(word_sdr)])
            word = list(self.dictionary.values())[word_class[0]]
            
            return word
    
    def make_value_correction(self, value_sdr):
        if self.__is_trained():
            value_class = self.model_values.predict([np.array(value_sdr)])
            value = list(self.values.values())[value_class[0]]
            
            return value

    def __is_trained(self):
        return self.model_dictionary_trained and self.model_values_trained