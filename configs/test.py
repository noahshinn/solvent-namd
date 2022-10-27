# from solvent_namd.utils import Config


# config = Config.from_file('./example.yaml')

# print('success')

class Simple:
    def __init__(self, a, b) -> None:
        self.a = a
        self.b = b
    
    def as_dict(self):
        return self.__dict__

s = Simple(1, 2)
print(s.as_dict())
