

from pyctm.representation.dictionary import Dictionary
from pyctm.representation.idea_metadata_values import IdeaMetadataValues
from pyctm.representation.sdr_idea_builder import SDRIdeaBuilder
from pyctm.representation.value_converter import ValueConverter

import random
import numpy as np
import struct


class SDRIdeaSerializer():

    def __init__(self, channels, rows, columns, default_value=0, active_value=1, dictionary=Dictionary(), to_raw=False):
        self.rows = rows
        self.columns = columns
        self.channels = channels
        self.default_value = default_value
        self.active_value = active_value
        self.dictionary = dictionary
        self.channel_counter = 1
        self.to_raw = to_raw

    def serialize(self, idea):

        if idea is not None:
            sdr_idea = SDRIdeaBuilder.build(
                self.channels, self.rows, self.columns, self.default_value, self.active_value)

            self.set_id_value(idea, sdr_idea.sdr, 0)
            self.set_name_value(idea, sdr_idea.sdr, 0)
            self.set_type_value(idea, sdr_idea.sdr, 0)
            self.set_metadata_value(idea, sdr_idea.sdr, 0)
            self.value_analyse(idea, sdr_idea.sdr, 0)

            self.channel_counter = 1
            self.generate_sdr(sdr_idea, idea)           
            
            return sdr_idea

        else:
            raise Exception('Idea Graph is null.')

    def generate_sdr(self, sdr_idea, idea):

        for child_idea in idea.child_ideas:

            self.set_parent_value(idea, sdr_idea.sdr, self.channel_counter)
            self.set_id_value(child_idea, sdr_idea.sdr, self.channel_counter)
            self.set_name_value(child_idea, sdr_idea.sdr, self.channel_counter)
            self.set_type_value(child_idea, sdr_idea.sdr, self.channel_counter)
            self.set_metadata_value(
                child_idea, sdr_idea.sdr, self.channel_counter)
            self.value_analyse(child_idea, sdr_idea.sdr, self.channel_counter)

            self.channel_counter += 1
            self.generate_sdr(sdr_idea, child_idea)

    def set_parent_value(self, idea, sdr, channel):
        self.set_numeric_value(sdr, channel, 0, self.columns, idea.id)

    def set_id_value(self, idea, sdr, channel):
        self.set_numeric_value(sdr, channel, 2, self.columns, idea.id)
    

    def set_name_value(self, idea, sdr, channel):
        if idea.name != None:
            self.set_value(sdr, channel, 4, self.get_array_from_words(idea.name))
    
    def set_value(self, sdr, channel, row, value):
        sdr[channel, row] = value

    def set_type_value(self, idea, sdr, channel):
        self.set_numeric_value(sdr, channel, 5, self.columns, idea.type)


    def get_type_name(self, value):
        return str(type(value)).replace('<class \'', '').replace('\'>', '')


    def set_metadata_value(self, idea, sdr, channel):

        if idea.value != None:

            metadata_value = 0
            idea_metadata_values = IdeaMetadataValues()

            if type(idea.value) is list:
                if len(idea.value) > 0:
                    list_type_name = self.get_type_name(idea.value)
                    element_type_name = self.get_type_name(idea.value[0])
                    metadata_value = idea_metadata_values.get_metadata_map()[list_type_name+'_'+element_type_name]
            else:
                variable_type_name = self.get_type_name(idea.value)
                metadata_value = idea_metadata_values.get_metadata_map()[variable_type_name]
            
            self.set_numeric_value(sdr, channel, 7, self.columns, metadata_value)

            length = 0

            if type(idea.value) is list:
                length = len(idea.value)
            

            self.set_numeric_value(sdr, channel, 9, self.columns, length)
            

    def value_analyse(self, idea, sdr, channel):
        if idea.value != None:
            if type(idea.value) is list:
                for i in range(0, len(idea.value)):
                    if type(idea.value[i]) is str:
                        self.set_value(sdr, channel, 11 + i, self.get_array_from_words(str(idea.value[i])))
                    else:
                        self.set_numeric_value(sdr, channel, 11+i*2, self.columns, idea.value[i])
            
            else:
                if type(idea.value) is str:
                    if idea != None:
                        self.set_value(sdr, channel, 11, self.get_array_from_words(str(idea.value)))
                else:
                    if type(idea.value) is bool:
                        self.set_numeric_value(sdr, channel, 11, self.columns, 1 if idea.value else 0)
                    else:
                        self.set_numeric_value(sdr, channel, 11, self.columns, idea.value)
                

    def set_numeric_value(self, sdr, channel, row, length, value):
        if self.to_raw:
            self.set_numeric_value_raw(sdr, channel, row, length, value)
        else:
            self.set_numeric_value_sdr(sdr, channel, row, length, value)

    def set_numeric_value_raw(self, sdr, channel, row, length, value):
        
        [d_value] = struct.unpack(">Q", struct.pack(">d", value))
        string_value = '{:064b}'.format(d_value)

        if string_value is not None or string_value != '':
            offset = 0
            for i in range(2):
                for j in range(length):
                    sdr[channel, row+i, j] = int(string_value[j+offset])
            
            offset = 32
        

    def set_numeric_value_sdr(self, sdr, channel, row, length, value):
        v_range = length//2

        value_converter = ValueConverter()

        base_ten_value = value_converter.convert_number_to_base_ten(abs(value))

        value_string = "%.2f" % base_ten_value[0]
        value_string = value_string.replace('.', '')
        value_string = value_string.replace('-', '')

        offset = 0
        interval = 0

        for i in range(0, min(len(value_string), 3)):
            value_int = abs(int(value_string[i]))

            value_sdr = self.get_array_from_values(value_int, v_range)

            for j in range(0, len(value_sdr)):
                sdr[channel, row+offset, interval*v_range+j] = value_sdr[j]
            
            if (i + 1) * v_range >= length:
                offset = offset + 1
                interval = 0
            else:
                interval = interval + 1

        base = base_ten_value[1]
        base_sdr = self.get_array_from_base_values(abs(int(base)), v_range//2)

        for i in range(0, len(base_sdr)):
            sdr[channel, row+1, v_range+i] = base_sdr[i]

        sinal_sdr = self.get_array_from_signal_values(1 if value < 0 else 0, v_range//4)

        for i in range(0, len(sinal_sdr)):
            sdr[channel, row+1, v_range + len(base_sdr) + i] = sinal_sdr[i]

        base_signal_sdr = self.get_array_from_signal_values(1 if base < 0 else 0, v_range//4)

        for i in range(0, len(base_signal_sdr)):
            sdr[channel, row+1, v_range + len(base_sdr) + len(sinal_sdr) + i] = base_signal_sdr[i]

    
    def get_array_from_base_values(self, base, length):
        if str(base) in self.dictionary.baseValues:
            return self.dictionary.baseValues[str(base)]
        else:
            
            array_base = self.generate_content(length, self.dictionary.baseValues)
            self.dictionary.baseValues[str(base)] = array_base

            return array_base

    def get_array_from_signal_values(self, signal, length):

        if str(signal) in self.dictionary.signalValues:
            return self.dictionary.signalValues[str(signal)]
        
        else:
            signal_sdr = np.full([int(length)], int(signal))
            self.dictionary.signalValues[str(signal)] = signal_sdr.tolist()

            return signal_sdr
    
    
    def get_array_from_words(self, word):

        if word in self.dictionary.words:
            return self.dictionary.words[word]
        
        else:
            array_word = self.generate_content(self.columns, self.dictionary.words)
            self.dictionary.words[word] = array_word

            return array_word 

    def get_array_from_values(self, value, length):
        if str(value) in self.dictionary.values:
            return self.dictionary.values.get(str(value))
        else:
            
            array_value = self.generate_content(length, self.dictionary.values)
            self.dictionary.values[str(value)] = array_value

            return array_value

    def generate_content(self, length, map):
        return self.generate_content_max_hamming_distance(map.values(), length, length//2)
    
    def generate_content_max_hamming_distance(self, existing_arrays, lenght_words, active_bits_in_words):
        
        new_array = [0] * lenght_words
        num_ones = 0
        while num_ones < random.randint(active_bits_in_words//2, active_bits_in_words):
            index_to_set = random.randint(0, lenght_words-1)
            if new_array[index_to_set] == 0:
                new_array[index_to_set] = 1
                num_ones += 1

        distances = []
        for array in existing_arrays:
            distance = sum([1 for i in range(lenght_words) if new_array[i] != array[i]])
            distances.append(distance)

        while min(distances) <= 1:
            index_to_flip = random.randint(0, lenght_words-1)
            new_array[index_to_flip] = 1 - new_array[index_to_flip]
            distances = []
            for array in existing_arrays:
                distance = sum([1 for i in range(lenght_words) if new_array[i] != array[i]])
                distances.append(distance)

        if new_array in existing_arrays:
            return self.generate_new_binary_array(existing_arrays, lenght_words, active_bits_in_words)
        else:
            return new_array
