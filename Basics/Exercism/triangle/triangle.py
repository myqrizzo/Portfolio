def equilateral(sides):
    # Search through array, compare all values
    # If all zero return False
    if sum(sides) == 0:
        return False
    # if all are zero or not equal return False
    for side in sides:
        if side != sides[0]:
            return False
    # if all are equal return True
    return True


def isosceles(sides):
    # Search through array, compare all values
    # Check triangle inequality
    sides.sort(reverse=True)
    if (sides[0] > (sides[1]+sides[2])):
        return False
    # Check 2 or 3 sides are equal
    my_dict = {i: sides.count(i) for i in sides}
    for value in my_dict.values():
        if value >= 2:
            return True
    return False


def scalene(sides):
    # Search through array, compare all values
    # Check triangle inequality
    sides.sort(reverse=True)
    if (sides[0] > (sides[1]+sides[2])):
        return False
    # Check 2 or 3 sides are equal
    my_dict = {i: sides.count(i) for i in sides}
    for value in my_dict.values():
        if value > 1:
            return False
    return True
