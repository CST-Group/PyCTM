from pyctm.representation.idea import Idea
from pyctm.representation.idea_metadata_values import IdeaMetadataValues
from pyctm.representation.value_validation import ValueValidation
import numpy as np

class SDRIdeaDeserializer:

    def __init__(self, dictionary, values):
        self.dictionary = dictionary
        self.values = values
        self.value_validation = ValueValidation()
    
    def deserializer(self, sdr_idea):

        idea_list = []

        self.__generate_idea_graph(sdr_idea, idea_list)

        return idea_list[0] if len(idea_list) > 0 else None
    
    def __generate_idea_graph(self, sdr_idea, idea_list):

        sdr = sdr_idea.sdr

        idea_relationship = {}

        for i in range(len(sdr)):

            sdr_channel = sdr[i]

            if self.__is_nullable_sdr(sdr_channel):
                continue
            
            parent_id = None

            if i != 0:
                parent_id = int(self.__extract_value(sdr_channel, 0))
            
            id = int(self.__extract_value(sdr_channel, 2))
            name = self.__extract_word(sdr_channel, 4)
            type = int(self.__extract_value(sdr_channel, 5))
            metadata = int(self.__extract_value(sdr_channel, 7))
            length = int(self.__extract_value(sdr_channel, 9))

            idea = Idea(_id=id, name=name, _type=type)

            self.__set_value(idea, sdr_channel, metadata, length)
        
            if parent_id is not None:
                idea_relationship[id] = parent_id                
                
            idea_list.append(idea)

        for idea in idea_list:

            if idea.id in idea_relationship:

                parent_id  = idea_relationship[idea.id]
                parent_idea  = self.__get_idea_in_list(parent_id, idea_list)

                if parent_idea is not None:
                    parent_idea.add(idea)

    def __set_value(self, idea, sdr_channel, metadata, length):

        if IdeaMetadataValues.is_array(metadata=metadata):
            self.__set_array_value(idea, sdr_channel, int(length), metadata)
        elif IdeaMetadataValues.is_primitive(metadata=metadata):
            idea.value = self.__extract_value(sdr_channel, 11)
        elif IdeaMetadataValues.is_bool(metadata=metadata):
            idea.value = bool(self.__extract_word(sdr_channel, 11))
        else:
            idea.value = self.__extract_word(sdr_channel, 11)
        
    
    def __set_array_value(self, idea, sdr_channel, length, metadata):

        if IdeaMetadataValues.is_string_array(metadata):
            string_list = []

            for i in range(length):
                string_list.append(self.__extract_word(sdr_channel, 11+i))
            
            idea.value = string_list

        elif IdeaMetadataValues.is_bool_array(metadata):
            bool_list = []

            for i in range(length):
                bool_list.append(bool(self.__extract_word(sdr_channel, 11+i)))
            
            idea.value = bool_list
        
        else:
            value_list = []

            for i in range(length):
                value_list.append(self.__extract_value(sdr_channel, 11+i*2))
            
            idea.value = value_list

    def __get_idea_in_list(self, id, idea_list):

        for idea in idea_list:
            if idea.id == id:
                return idea
        
        return None

    def __is_nullable_sdr(self, sdr):

        sum_check = 0

        for i in range(len(sdr)):
            for j in range(len(sdr[i])):
                sum_check += sdr[i][j]
        
        return sum_check == 0

    def __extract_word(self, sdr_channel, row):

        word = self.__get_word(sdr_channel[row])

        return word if word is not None else ''

    
    def __extract_value(self, sdr_channel, row):

        length = len(sdr_channel[row])
        x_range = int(length/4)

        value_string = ""

        for i in range(4):

            value_sdr = self.__build_sdr(x_range, sdr_channel[row], i)

            value_key = self.__get_value(value_sdr)

            if value_key is not None:
                value_string += str(value_key)
            
            if i == 0:
                value_string += '.'
        
        if len(value_string) == 1 or value_string == '':
            return 0
        
        base_sdr = self.__build_sdr(x_range, sdr_channel[row+1], 0)

        base = 0

        base_key = self.__get_value(base_sdr)

        if base_key is not None:
            base = base_key
        
        value_signal = 1 if sdr_channel[row+1][x_range] * -1 == 0 else -1
        base_signal = 1 if sdr_channel[row+1][x_range+1] * -1 == 0 else -1

        number = float(value_string) * (10 ** (base*base_signal)) * value_signal

        return number

    def __build_sdr(self, x_range, sdr_row, interval):
        
        sdr = np.full([x_range], 0)

        for i in range(x_range):
            sdr[i] = sdr_row[interval*x_range+i]
        
        return sdr
    
    def __get_value(self, value_sdr):

        for index in range(len(self.values.values())):
            if self.value_validation.compare_value(list(self.values.values())[index], value_sdr):
                return list(self.values.keys())[index]
        
        return None
    
    def __get_word(self, word_sdr):

        for index in range(len(self.dictionary.values())):
            if self.value_validation.compare_value(list(self.dictionary.values())[index], word_sdr):
                return list(self.dictionary.keys())[index]
        
        return None
