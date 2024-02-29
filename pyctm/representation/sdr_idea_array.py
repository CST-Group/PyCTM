class SDRIdeaArray:
    def __init__(self, total_of_ideas, total_of_values=None, default_value=0):
        if total_of_values is not None:
            self.sdr = [0] * (15 + 2 + total_of_values * 6 + ((total_of_ideas - 1) * (21 + total_of_values * 6)))
        else:
            self.sdr = [0] * (total_of_ideas * 72)
        self.default_value = default_value

    