# Assuming the existence of the Idea class and ValueValidation class
# from idea import Idea
# from value_validation import ValueValidation

from pyctm.representation.array_value_validation import ArrayValueValidation
from pyctm.representation.idea import Idea
from pyctm.representation.idea_metadata_values import IdeaMetadataValues
from pyctm.representation.array_value_converter import ArrayValueConverter
from pyctm.representation.value_validation import ValueValidation


class SDRIdeaArrayDeserializer:
    def __init__(self, dictionary):
        self.dictionary = dictionary

        if self.dictionary is not None:
            self.dictionary.words = {int(key): value for key, value in self.dictionary.words.items()}
            
        self.value_converter = ArrayValueConverter()
        self.index = 0
        self.start_word = 1
        self.end_word = 2

    def deserialize(self, sdr_idea_array):
        if sdr_idea_array is None or sdr_idea_array.sdr is None:
            raise Exception("SDR Idea Array is null or empty.")

        idea_relationship = {}
        idea_list = []

        sdr = sdr_idea_array.sdr
        self.index = 0

        if self.dictionary is not None:
            self.dictionary.words = {int(key): value for key, value in self.dictionary.words.items()}

        while self.index < len(sdr):
            if sdr[self.index] == self.start_word:
                self.index += 1
                continue
            elif sdr[self.index] == self.end_word:
                break

            idea = Idea()

            parent_id = None
            if idea_list:
                parent_id = self.get_value_according_type(self.get_numeric_value(sdr), 'long')

            idea.id = self.get_value_according_type(self.get_numeric_value(sdr), 'long')
            idea.name = self.get_string_value(sdr)
            idea.type = int(self.get_string_value(sdr))
            idea.value = self.get_value(sdr)

            if parent_id is not None:
                idea_relationship[idea.id] = parent_id

            idea_list.append(idea)

        for idea_element in idea_list:
            relations = [relation for relation in idea_relationship.items() if relation[1] == idea_element.id]
            for relation in relations:
                child_idea = next((idea for idea in idea_list if idea.id == relation[0]), None)
                if child_idea:
                    idea_element.child_ideas.append(child_idea)

        return idea_list[0] if idea_list else None

    def get_local_numeric_value(self, sdr):
        index = 0
        value_string = ""

        for i in range(3):
            value_string += str(int(self.dictionary.words[str(sdr[index + i])]))

        index += 3

        value_string = f"{value_string[0]}.{value_string[1]}{value_string[2]}"

        signal = self.dictionary.words[str(sdr[index])]
        index += 1
        base = int(self.dictionary.words[str(sdr[index])])
        index += 1
        base_signal = self.dictionary.words[str(sdr[index])]
        index += 1

        value = float(value_string) * (10 ** (base * (1 if base_signal == "+" else -1)))
        value = value if signal == "+" else -value
        value = round(value, 2)

        return value
    
    def get_local_string_value(self, index):
        value = self.dictionary.words[str(index)]
        return value
    
    def get_metadata_type(self, metadata_value):
        idea_metadata_values = IdeaMetadataValues()
        metadata_map = idea_metadata_values.get_metadata_map()
        for clazz, metadata in metadata_map.items():
            if metadata == metadata_value:
                if ArrayValueValidation.is_array(clazz):
                    if "list_str" == clazz:
                        return "STRING_ARRAY"
                    else:
                        return "NUM_ARRAY"
                elif ArrayValueValidation.is_primitive(clazz):
                    return "NUM_VALUE"
                elif ArrayValueValidation.is_string(clazz):
                    return "STRING_VALUE"

    def get_numeric_value(self, sdr):
        value_string = ""

        for i in range(3):
            value_string += str(int(self.dictionary.words[sdr[self.index + i]]))

        self.index += 3

        value_string = f"{value_string[0]}.{value_string[1]}{value_string[2]}"

        signal = self.dictionary.words[sdr[self.index]]
        self.index += 1
        base = int(self.dictionary.words[sdr[self.index]])
        self.index += 1
        base_signal = self.dictionary.words[sdr[self.index]]
        self.index += 1

        value = float(value_string) * (10 ** (base * (1 if base_signal == "+" else -1)))
        value = value if signal == "+" else -value
        value = round(value, 2)

        return value

    def get_string_value(self, sdr):
        value = self.dictionary.words[sdr[self.index]]
        self.index += 1
        return value

    def get_value(self, sdr):
        metadata_value = int(self.dictionary.words[sdr[self.index]])
        self.index += 1

        length = self.get_value_according_type(self.get_numeric_value(sdr), 'int')

        idea_metadata_values = IdeaMetadataValues()
        metadata_map = idea_metadata_values.get_metadata_map()
        for clazz, metadata in metadata_map.items():
            if metadata == metadata_value:
                if ArrayValueValidation.is_array(clazz):
                    return self.get_array_value(sdr, length, clazz)
                elif ArrayValueValidation.is_primitive(clazz):
                    return self.get_value_according_type(self.get_numeric_value(sdr), clazz)
                elif ArrayValueValidation.is_string(clazz):
                    return self.get_string_value(sdr)
        return None

    def get_array_value(self, sdr, length, clazz):
        array = [None] * length
        for i in range(length):
            if clazz == 'list_double' or clazz == 'list_float':
                array[i] = self.get_value_according_type(self.get_numeric_value(sdr), 'float')
            elif clazz == 'list_int':
                array[i] = self.get_value_according_type(self.get_numeric_value(sdr), 'int')
            elif clazz == 'list_short':
                array[i] = self.get_value_according_type(self.get_numeric_value(sdr), 'short')
            elif clazz == 'list_bool':
                array[i] = self.get_string_value(sdr) == 'True'
            elif clazz == 'list_long':
                array[i] = self.get_value_according_type(self.get_numeric_value(sdr), 'long')
            elif clazz == 'list_str':
                array[i] = self.get_string_value(sdr)
        return array

    def get_value_according_type(self, value, clazz):
        if clazz == 'int':
            return int(value)
        elif clazz == 'float':
            return float(value)
        elif clazz == 'short':
            return int(value)  # Python does not have a native 'short' type
        elif clazz == 'long':
            return int(value)
        elif clazz == 'double':
            return float(value)
        else:
            return value