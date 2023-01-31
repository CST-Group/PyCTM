
class ValueConverter():

    def convert_number_to_base_ten(self, value):

        number_base_ten = []

        base = 0
        value_divided = value

        while True:

            if value == 0 or value == 1:
                number_base_ten.append(float(value))
                number_base_ten.append(float(0))

                return number_base_ten
            
            if value > 1:
                if value_divided >= 1:
                    value_divided/=10
                    base+=1
                
                else:
                    number_base_ten.append(float(value_divided*10))
                    number_base_ten.append(float(base-1))

                    return number_base_ten
            
            elif value < 1:
                if value_divided <= 1:
                    value_divided*=10
                    base-=1
                else:
                    number_base_ten.append(float(value_divided/10))
                    number_base_ten.append(float(base+1))

                    return number_base_ten
            elif value < 1:
                if value_divided <= 1:
                    value_divided*=10
                    base-=1
                else:
                    number_base_ten.append(float(value_divided/10))
                    number_base_ten.append(float(base+1))

                    return number_base_ten