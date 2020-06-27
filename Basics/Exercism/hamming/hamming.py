def distance(strand_a, strand_b):
    if (len(strand_a) != len(strand_b)):
        raise ValueError("The two strands are of a different legnth.")

    return sum([ch_a != ch_b for ch_a, ch_b in zip(strand_a, strand_b)])
