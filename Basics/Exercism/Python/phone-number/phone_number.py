class PhoneNumber:
    def __init__(self, number):
        [self.area_code, self.number] = self.initialize(number)

    def initialize(self, number):
        temp_num = list(filter(lambda x: x.isdigit(), number))
        if len(temp_num) == 10:
            if temp_num[0] == '0' or temp_num[0] == '1' or temp_num[3] == '0' or temp_num[3] == '1':
                raise ValueError("The number has been deemed invalid.")
            else:
                return ''.join(temp_num[0:3]), ''.join(temp_num[0:len(temp_num)])
        else:
            if len(temp_num) > 10 and temp_num[0] == '1':
                if temp_num[1] == '0' or temp_num[1] == '1' or temp_num[4] == '0' or temp_num[4] == '1':
                    raise ValueError("The number has been deemed invalid.")
                else:
                    return ''.join(temp_num[1:4]), ''.join(temp_num[1:len(temp_num)])
            else:
                raise ValueError("The number has been deemed invalid.")

    def pretty(self):
        ex_code = ''.join(list(self.number)[3:6])
        rem_code = ''.join(list(self.number)[6:10])
        return "(" + self.area_code + ")" + " " + ex_code + "-" + rem_code
