def steps(number):
    if number <= 0:
        raise ValueError("The number entered is invalid.")
    step_tot = 0
    while number > 1:
        if number % 2 == 0:
            number = number / 2
        else:
            number = 3*number + 1
        step_tot += 1

    return step_tot
