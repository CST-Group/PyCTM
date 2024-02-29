from pyctm.representation.sdr_idea_array import SDRIdeaArray

class SDRIdeaArrayBuilder:
    def build(self, total_of_ideas, total_of_values, default_value, start_word):
        sdr_idea_array = SDRIdeaArray(total_of_ideas, total_of_values, default_value)
        self.initialize_matrix(sdr_idea_array, start_word)
        return sdr_idea_array

    def initialize_matrix(self, sdr_idea_array, start_word):
        sdr_idea_array.sdr[0] = start_word
        for k in range(1, len(sdr_idea_array.sdr)):
            sdr_idea_array.sdr[k] = sdr_idea_array.default_value