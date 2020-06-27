def is_valid(isbn):
    if(len(isbn) == 0 or len(list(filter(lambda x: ((not x.isdigit()) and x != 'X' and x != '-'), isbn))) > 0):
        return False
    else:
        temp_num = list(filter(lambda x: (x.isdigit() or x == 'X'), isbn))
        if(len(temp_num) < 9):
            return False
        indexes = [i for i, x in enumerate(temp_num) if x == 'X']
        if(len(indexes) > 0 and (len(indexes) > 1 or indexes[0] != 9)):
            return False
        else:
            i = 9
            if(len(temp_num) == 10):
                i = i + 1
            if(len(indexes) > 0 and indexes[0] == 9):
                temp_num[9] = '10'
            temp_num = list(map(int, temp_num))
            check_sum = 0
            tot = i
            while i > 0:
                check_sum = check_sum + i*temp_num[tot-i]
                i = i - 1
        return (check_sum % 11) == 0
