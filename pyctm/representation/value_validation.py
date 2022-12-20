class ValueValidation:

    def __init__(self):
        pass

    def is_array(self, object):
        return object

    def compare_value(self, new_value, value):

        if len(new_value) == len(value):

            for i in range(len(new_value)):
                if new_value[i] != value[i]:
                    return False
            

            return True
        
        return False