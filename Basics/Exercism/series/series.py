def slices(series, length):
    diff = len(series) - length
    strArray = []
    if diff >= 0 and diff < len(series):
        L = list(series)
        i = 0
        for i in range(diff+1):
            strArray.append(''.join(L[i:length+i]))
        return strArray
    else:
        raise ValueError("The length specified is out of bounds.")
