

from pyctm.representation.sdr_idea import SDRIdea


class SDRIdeaBuilder():

    @staticmethod
    def build(channels, rows, columns, default_value, active_value):
        sdr_idea = SDRIdea(channels, rows, columns)
        return sdr_idea
