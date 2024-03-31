from typing import Any, Dict
from copy import deepcopy

class DictObj:
    def __init__(self, value:Dict) -> None:
        self.value:Dict = value
        pass

    def __getattr__(self, __name: str) -> Any:
        # print(__name)
        return self.value[__name]

    def __str__(self) -> str:
        return self.value.__str__()
    
    def __repr__(self):
        return repr(self.value)      

    def __deepcopy__(self):
        return DictObj(deepcopy(self.value))

class ListObj:
    def __init__(self, list) -> None:
        self.list = list
        pass

    def __getattr__(self, attr):
        # print(self.list[0].__dict__)
        return list(map(lambda x: x.__getattr__(attr), self.list))

    def __deepcopy__(self):
        return DictObj(deepcopy(self.list))