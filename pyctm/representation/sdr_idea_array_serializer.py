from pyctm.representation.array_dictionary import ArrayDictionary
from pyctm.representation.array_value_converter import ArrayValueConverter
from pyctm.representation.idea_metadata_values import IdeaMetadataValues
from pyctm.representation.sdr_idea_array_builder import SDRIdeaArrayBuilder


class SDRIdeaArraySerializer:
    def __init__(self, total_of_ideas, total_of_values, default_value, dictionary=None):
        self.total_of_ideas = total_of_ideas
        self.total_of_values = total_of_values
        self.default_value = default_value
        self.value_converter = ArrayValueConverter()
        self.dictionary = dictionary if dictionary is not None else ArrayDictionary({})
        self.index = 0
        self.start_word = 1
        self.end_word = 2

    def serialize(self, idea):
        if idea is None:
            raise Exception("Idea Graph is null.")

        self.index = 0
        sdr_idea_array = SDRIdeaArrayBuilder().build(self.total_of_ideas, self.total_of_values, self.default_value, self.start_word)
        self.index += 1

        self.set_id_value(idea, sdr_idea_array.sdr)
        self.set_name_value(idea, sdr_idea_array.sdr)
        self.set_type_value(idea, sdr_idea_array.sdr)
        self.set_metadata_value(idea, sdr_idea_array.sdr)
        self.value_analyse(idea, sdr_idea_array.sdr)

        self.generate_sdr(sdr_idea_array, idea)

        sdr_idea_array.sdr[self.index] = self.end_word

        return sdr_idea_array

    def set_parent_value(self, idea, sdr):
        self.set_value(sdr, idea.id)

    def set_id_value(self, idea, sdr):
        self.set_value(sdr, idea.id)

    def value_analyse(self, idea, sdr):
        if isinstance(idea.value, list):
            for value in idea.value:
                if isinstance(value, (int, float, str)):
                    self.set_value(sdr, value)
                else:
                    self.set_word(sdr, self.get_value_from_dictionary(str(value)))
        else:
            if isinstance(idea.value, (int, float)):
                self.set_value(sdr, idea.value)
            elif isinstance(idea.value, str):
                self.set_word(sdr, self.get_value_from_dictionary(idea.value))

    def generate_sdr(self, sdr_idea_array, idea):
        for child_idea in idea.child_ideas:
            self.set_parent_value(idea, sdr_idea_array.sdr)
            self.set_id_value(child_idea, sdr_idea_array.sdr)
            self.set_name_value(child_idea, sdr_idea_array.sdr)
            self.set_type_value(child_idea, sdr_idea_array.sdr)
            self.set_metadata_value(child_idea, sdr_idea_array.sdr)
            self.value_analyse(child_idea, sdr_idea_array.sdr)

            self.generate_sdr(sdr_idea_array, child_idea)

    def set_metadata_value(self, idea, sdr):
        if idea.value is not None:
            
            idea_metadata_values = IdeaMetadataValues()

            if isinstance(idea.value, list) and len(idea.value) > 0:
                element_type = type(idea.value[0]).__name__
                metadata_key = f'list_{element_type}'
            else:
                metadata_key = type(idea.value).__name__

            metadata_value = idea_metadata_values.get_metadata_map().get(metadata_key, 0)
            self.set_word(sdr, self.get_value_from_dictionary(str(metadata_value)))

            length = len(idea.value) if isinstance(idea.value, list) else 0
            self.set_value(sdr, length)

    def set_name_value(self, idea, sdr):
        if idea.name is not None:
            self.set_word(sdr, self.get_value_from_dictionary(idea.name))

    def set_type_value(self, idea, sdr):
        self.set_word(sdr, self.get_value_from_dictionary(str(idea.type)))

    def set_word(self, sdr, value):
        sdr[self.index] = value
        self.index += 1

    def set_value(self, sdr, value):
        self.set_numeric_value(sdr, value)

    def set_numeric_value(self, sdr, value):
        base_ten_value = self.value_converter.convert_number_to_base_ten(abs(value))
        value_string = "{:.2f}".format(base_ten_value[0]).replace(".", "").replace("-", "")

        for i in range(min(len(value_string), 3)):
            value_int = int(value_string[i])
            sdr[self.index] = self.get_value_from_dictionary(value_int)
            self.index += 1

        signal = self.get_value_from_dictionary("+" if value >= 0 else "-")
        sdr[self.index] = signal
        self.index += 1

        base = self.get_value_from_dictionary(abs(int(base_ten_value[1])))
        sdr[self.index] = base
        self.index += 1

        base_signal = self.get_value_from_dictionary("+" if base_ten_value[1] >= 0 else "-")
        sdr[self.index] = base_signal
        self.index += 1

    def get_value_from_dictionary(self, value):
        for key, val in self.dictionary.words.items():
            if val == value:
                return key
        else:
            new_key = len(self.dictionary.words)
            self.dictionary.words[new_key] = value
            return new_key