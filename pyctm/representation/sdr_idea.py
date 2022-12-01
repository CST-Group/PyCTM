import numpy as np

class SDRIdea():

    def __init__(self, channels, rows, columns):
        self.channels = channels
        self.rows = rows
        self.columns = columns
        self.default_value = 0
        self.active_value = 1
        self.sdr = np.full([channels, rows, columns], self.default_value)