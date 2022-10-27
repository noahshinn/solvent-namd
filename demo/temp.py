
class Thing:
    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
    def add(self, n3):
        self.__setattr__('n3', n3)

thing = Thing(1, 2)
thing.add(3)

print(thing.__dict__)
