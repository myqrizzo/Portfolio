import numpy as np


def square_of_sum(number):
    return sum(range(1, number+1))**2


def sum_of_squares(number):
    return np.dot(range(1, number+1), range(1, number+1))


def difference_of_squares(number):
    return square_of_sum(number) - sum_of_squares(number)
