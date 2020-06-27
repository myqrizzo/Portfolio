def convert(number):
    hit = 0
    str1 = "Pling"
    str2 = "Plang"
    str3 = "Plong"
    strCat = ""
    if number % 3 == 0:
        hit = 1
        strCat = strCat + str1
    if number % 5 == 0:
        hit = 1
        strCat = strCat + str2
    if number % 7 == 0:
        hit = 1
        strCat = strCat + str3
    if hit == 0:
        strCat = str(number)
    return strCat
    
     
