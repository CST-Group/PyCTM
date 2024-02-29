class ArrayValueConverter:
    def extract_double_array(self, matrix, length, i):
        return [float(matrix[i][k + len(matrix) + 4]) for k in range(length)]

    def extract_int_array(self, matrix, length, i):
        return [int(matrix[i][k + len(matrix) + 4]) for k in range(length)]

    def extract_float_array(self, matrix, length, i):
        return [float(matrix[i][k + len(matrix) + 4]) for k in range(length)]

    def extract_long_array(self, matrix, length, i):
        return [int(matrix[i][k + len(matrix) + 4]) for k in range(length)]

    def extract_short_array(self, matrix, length, i):
        return [int(matrix[i][k + len(matrix) + 4]) for k in range(length)]

    def extract_boolean_array(self, matrix, length, i):
        return [bool(matrix[i][k + len(matrix) + 4]) for k in range(length)]

    def convert_to_double_array(self, obj):
        return list(map(float, self.convert_to_list(obj)))

    def convert_to_generic_array(self, obj, cls):
        return list(map(cls, self.convert_to_list(obj)))

    def convert_to_list(self, obj):
        if isinstance(obj, list):
            return obj
        elif isinstance(obj, (int, float, bool)):
            return [obj]
        else:
            raise ValueError("Unsupported type for conversion to list")

    def convert_number_to_base_ten(self, value):
        value = abs(value)
        base = 0
        value_divided = value
        while True:
            if value == 0 or value == 1:
                return [value, 0]
            if value > 1:
                if value_divided >= 1:
                    value_divided /= 10
                    base += 1
                else:
                    return [value_divided * 10, base - 1]
            elif value < 1:
                if value_divided <= 1:
                    value_divided *= 10
                    base -= 1
                else:
                    return [value_divided / 10, base + 1]