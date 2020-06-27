import random
import string 

class Robot(object):
    def __init__(self):
        self.name = ''.join(random.choice(string.ascii_uppercase) for i in range(2))+''.join(random.choice(string.digits) for j in range(3))

    def reset(self):
        seed = "Totally Random."
        random.seed(seed)
        self.name = ''.join(random.choice(string.ascii_uppercase) for i in range(2))+''.join(random.choice(string.digits) for j in range(3))
