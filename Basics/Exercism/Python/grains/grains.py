def square(number):
    # raise number with 2 as base
    # check number in acceptable range
    # return to receive total number of grains on chess square
    if (number <= 0 or number >= 65):
        raise ValueError("The number is outside the acceptable range.")
    return 2**(number-1)


def total():
    # iterate over chess board
    # return sum over all function calls to return total grains
    return sum([square(i) for i in range(1, 65)])
