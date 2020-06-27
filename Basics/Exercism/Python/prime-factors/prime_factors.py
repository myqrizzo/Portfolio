def factors(value):
    # Func: Compute the prime factors of a given natural number
    itr = 2  # assume first prime is 2
    primes = []  # receptacle for discovered primes
    temp = value  # remaining after division by prime
    while temp > 1:
        # check value mod 2 is zero
        if (temp % itr) == 0:
            # if yes, append and continue with this number
            temp = temp / itr
            primes.append(itr)
        else:
            # if no, do not append again, continue with next
            itr += 1
    # return prime factors
    return primes
