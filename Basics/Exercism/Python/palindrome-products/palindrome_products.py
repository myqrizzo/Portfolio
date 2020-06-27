def is_palindrome(number):
    reverse = 0
    # calculate reverse of product to check
    # whether or not it is a palindrome
    while (number != 0):
        reverse = reverse * 10 + number % 10
        number = number // 10
    return reverse


def get_all_factors(n, min, max):
    factors = [[i, n // i]
               for i in range(min, max+1) if n % i == 0 and (n // i) <= max and (n // i) >= min]
    print(factors)
    if n > 1 and (n // 1) <= max and (n // 1) >= min:
        factors.pop()
    return factors


def largest(min_factor, max_factor):
    if min_factor > max_factor:
        raise ValueError("The minimum is greater than the maximum.")
    elif min_factor != max_factor:
        max_product = 0  # Initialize result
        for i in range(max_factor, min_factor-1, -1):
            for j in range(i, min_factor-1, -1):

                # calculating product of
                # two n-digit numbers
                product = i * j
                if (product < max_product):
                    break

                # Detemine if product is a palindrome by checking reverse
                reverse = is_palindrome(product)

                # update new product if exist and if
                # greater than previous one
                if (product == reverse and product > max_product):
                    max_product = product
    else:
        return None, []

    return max_product, get_all_factors(max_product, min_factor, max_factor)


def smallest(min_factor, max_factor):
    if min_factor > max_factor:
        raise ValueError("The minimum is greater than the maximum.")
    elif min_factor != max_factor:
        min_product = 1e9  # Initialize result
        for i in range(max_factor, min_factor-1, -1):
            for j in range(i, min_factor-1, -1):

                # calculating product of
                # two n-digit numbers
                product = i * j
                if (product > min_product):
                    break

                # Detemine if product is a palindrome by checking reverse
                reverse = is_palindrome(product)

                # update new product if exist and if
                # smaller than previous one
                if (product == reverse and product < min_product):
                    min_product = product
    else:
        return None, []

    if(min_product == 1e9):
        return None, []

    return min_product, get_all_factors(min_product, min_factor, max_factor)
