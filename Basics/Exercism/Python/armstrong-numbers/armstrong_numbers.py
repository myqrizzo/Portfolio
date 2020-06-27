def is_armstrong(number):
    numToList = list(map(int, str(number)))
    return number == sum(list(map(lambda x:pow(x,len(numToList)),numToList)))
