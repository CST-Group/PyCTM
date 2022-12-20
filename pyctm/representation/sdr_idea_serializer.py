

from pyctm.representation.idea_metadata_values import IdeaMetadataValues
from pyctm.representation.sdr_idea_builder import SDRIdeaBuilder
from pyctm.representation.value_converter import ValueConverter
import random
import numpy as np


class SDRIdeaSerializer():

    def __init__(self, channels, rows, columns, default_value=0, active_value=1, dictionary={}, values={}):
        self.rows = rows
        self.columns = columns
        self.channels = channels
        self.default_value = default_value
        self.active_value = active_value
        self.value_converter = {}
        self.dictionary = dictionary
        self.values = values
        self.channel_counter = 1

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
            self.set_value(sdr, channel, 4, self.get_array_from_dictionary(idea.name))
    
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
        if type(idea.value) is list:
            for i in range(0, len(idea.value)):
                self.set_numeric_value(sdr, channel, 11+i*2, self.columns, idea.value[i])
        
        else:
            if type(idea.value) is str:
                if idea != None:
                    self.set_value(sdr, channel, 11, self.get_array_from_dictionary(str(idea.value)))
            else:
                if type(idea.value) is bool:
                    self.set_numeric_value(sdr, channel, 11, self.columns, 1 if idea.value else 0)
                else:
                    self.set_numeric_value(sdr, channel, 11, self.columns, idea.value)
                



    def set_numeric_value(self, sdr, channel, row, length, value):
        v_range = length//4

        value_converter = ValueConverter()

        base_ten_value = value_converter.convert_number_to_base_ten(abs(value))

        value_string = str(base_ten_value[0])
        value_string = value_string.replace('.', '')
        value_string = value_string.replace('-', '')

        for i in range(0, min(len(value_string), 4)):
            value_int = abs(int(value_string[i]))

            value_sdr = self.get_array_from_value(value_int, v_range)

            for j in range(0, len(value_sdr)):
                sdr[channel, row, i*v_range+j] = value_sdr[j]

        base = base_ten_value[1]
        base_sdr = self.get_array_from_value(abs(base), v_range)

        for i in range(0, len(base_sdr)):
            sdr[channel, row+1, i] = base_sdr[i]

        if value < 0:
            sdr[channel, row+1, len(base_sdr)] = 1
        else:
            sdr[channel, row+1, len(base_sdr)] = 0

        if base < 0:
            sdr[channel, row+1, len(base_sdr)+1] = 1
        else:
            sdr[channel, row+1, len(base_sdr)+1] = 0

    def get_array_from_value(self, value, length):
        if value in self.values:
            return self.values.get(value)
        else:
            while True:
                array_value = self.generate_content(length, True, {}, self.values)

                if self.check_compatibility(array_value, True):

                    self.values[value] = array_value

                    return array_value

    def generate_content(self, length, is_value, dictionary, values):

        retry = True

        while retry:

            value = self.generate_value(length)
            if is_value:
                
                if len(values.values()) == 0:
                    retry = False
                else:
                    for stored_value in values.values():
                        if len(stored_value) == len(value) and not self.compare_values(stored_value, value):
                            retry = False
            else:
                if len(dictionary.values()) == 0:
                    retry = False
                else:
                    for stored_value in dictionary.values():
                        if len(stored_value) == len(value) and not self.compare_values(stored_value, value):
                            retry = False                

            if retry == False:
                return value

    def generate_value(self, length):

        w = int(length/2)

        value = np.full([int(length)], int(self.default_value))

        for i in range(0, w):

            retry = True

            while retry:
                index = random.randint(0, length-1)
                if value[index] != 1:
                    value[index] = self.active_value
                    retry = False
        
        return value

    def get_array_from_dictionary(self, word):

        if word in self.dictionary:
            return self.dictionary[word]
        
        else:
            array_word = self.generate_content(self.columns, False, self.dictionary, {})

            while True:
                if self.check_compatibility(array_word, False):

                    self.dictionary[word] = array_word

                    return array_word

    def compare_values(self, new_value, value):

        if len(new_value) == len(value):

            for i in range(0, len(new_value)):
                if new_value[i] != value[i]:
                    return False

            return True

        return False
    
    def check_compatibility(self, new_value_word, is_value):

        if is_value:
            for value in self.values:
                if self.compare_values(new_value_word, self.values[value]):
                    return False
        else:
            for word in self.dictionary:
                if self.compare_values(new_value_word, self.dictionary[word]):
                    return False
        
        return True