from pyctm.representation.idea import Idea
from pyctm.representation.idea_metadata_values import IdeaMetadataValues
from pyctm.representation.value_validation import ValueValidation
import numpy as np
from codecs import decode
import struct

class SDRIdeaDeserializer:

    def __init__(self, dictionary, corrector_engine=None, to_raw=False):
        self.dictionary = dictionary
        self.value_validation = ValueValidation()
        self.corrector_engine = corrector_engine
        self.to_raw = to_raw
    
    def deserialize(self, sdr):

        idea_list = []

        self.__generate_idea_graph(sdr, idea_list)

        return idea_list[0] if len(idea_list) > 0 else None
    
    def __generate_idea_graph(self, sdr, idea_list):

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
                sum_check = sum_check + sdr[i][j]
        
        return sum_check <= 10

    def __extract_word(self, sdr_channel, row):

        word_sdr = sdr_channel[row]

        if self.corrector_engine is not None:
            word_sdr = self.corrector_engine.make_word_correction(word_sdr)

        word = self.__get_word(word_sdr)

        return word if word is not None else ''

    
    def __extract_value(self, sdr_channel, row):
        
        if self.to_raw:
            return self.__extract_value_raw(sdr_channel, row)
        
        return self.__extract_value_sdr(sdr_channel, row)

    def __extract_value_raw(self, sdr_channel, row):

        string_value = ""
        for i in range(2):
            sdr_channel[row+i][sdr_channel[row+i] < 0.5] = 0
            sdr_channel[row+i][sdr_channel[row+i] >= 0.5] = 1   

            for j in range(len(sdr_channel)):
                string_value += str(int(sdr_channel[row + i, j]))
                
        return self.__bin_to_float(string_value)
    
    def __int_to_bytes(self, n, length):
        return decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]

    def __bin_to_float(self, b):
        bf = self.__int_to_bytes(int(b, 2), 8)
        return struct.unpack('>d', bf)[0]

    def __extract_value_sdr(self, sdr_channel, row):

        length = len(sdr_channel[row])
        x_range = int(length/2)

        offset = 0
        interval = 0

        value_string = ""
        for i in range(3):

            value_sdr = self.__build_sdr(x_range, sdr_channel[row+offset], interval)

            if self.corrector_engine is not None:
                value_sdr = self.corrector_engine.make_value_correction(value_sdr)

            value_key = self.__get_value(value_sdr)

            if value_key is not None:
                value_string += str(value_key)
            
            if i == 0:
                value_string += '.'
            
            if (i + 1) * x_range >= length:
                offset = offset + 1
                interval = 0
            else:
                interval = interval + 1
        
        if len(value_string) == 1 or value_string == '' or value_string == '0.00':
            return 0
        
        base_sdr = self.__build_sdr(int(x_range/2), sdr_channel[row+1], 2)

        if self.corrector_engine is not None:
            base_sdr = self.corrector_engine.make_base_correction(base_sdr)

        base = 0

        base_key = self.__get_base(base_sdr)
        if base_key is not None:
            base = base_key
        
        value_signal_sdr  = self.__build_sdr(int(x_range/4), sdr_channel[row+1], 6)
        base_signal_sdr  = self.__build_sdr(int(x_range/4), sdr_channel[row+1], 7)
        
        if self.corrector_engine is not None:
            value_signal_sdr = self.corrector_engine.make_signal_correction(value_signal_sdr)
            base_signal_sdr = self.corrector_engine.make_signal_correction(base_signal_sdr)

        value_signal_key = self.__get_signal(value_signal_sdr)
        base_signal_key = self.__get_signal(base_signal_sdr)
        
        value_signal = 1 if int(value_signal_key) * -1 == 0 else -1
        base_signal = 1 if int(base_signal_key) * -1 == 0 else -1

        number = float(value_string) * (10 ** (float(base)*base_signal)) * value_signal

        return number

    def __build_sdr(self, x_range, sdr_row, interval):        
        return sdr_row[interval*x_range:(interval+1)*x_range]
    
    def __get_value(self, value_sdr):

        for index in range(len(self.dictionary.values.values())):
            if self.value_validation.compare_value(list(self.dictionary.values.values())[index], value_sdr):
                return list(self.dictionary.values.keys())[index]
        
        return None
    
    def __get_word(self, word_sdr):

        for index in range(len(self.dictionary.words.values())):
            if self.value_validation.compare_value(list(self.dictionary.words.values())[index], word_sdr):
                return list(self.dictionary.words.keys())[index]
        
        return None

    
    def __get_base(self, base_sdr):

        for index in range(len(self.dictionary.baseValues.values())):
            if self.value_validation.compare_value(list(self.dictionary.baseValues.values())[index], base_sdr):
                return list(self.dictionary.baseValues.keys())[index]
        
        return None
    
    def __get_signal(self, signal_sdr):

        for index in range(len(self.dictionary.signalValues.values())):
            if self.value_validation.compare_value(list(self.dictionary.signalValues.values())[index], signal_sdr):
                return list(self.dictionary.signalValues.keys())[index]
        
        return None
