def classify(number):
    if number < 0 or number == 0:
        myError = ValueError('The number is not a natural number.')
        raise myError
    
    sum = 0
    for i in range(1, number):
        if number % i == 0:
            sum = sum + i
    
    if (sum == number):
        return "perfect"
    if (sum < number):
        return "deficient"
    if (sum > number):
        return "abundant"