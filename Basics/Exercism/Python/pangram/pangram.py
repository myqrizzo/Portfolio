def is_pangram(sentence):
    alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    sentence = ''.join(letter for letter in sentence.lower() if 'a' <= letter <= 'z')
    return 0 not in [character in sentence for character in alphabet]
