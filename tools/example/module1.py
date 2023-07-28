from .subpackage1 import _submodule1
from .subpackage1 import _submodule2

class ClassInModule1:
    def __init__(self):
        pass

    def method1(self):
        print("Method 1 in ClassInModule1")

    def method2(self):
        # Using classes from the subpackage internally
        class_instance = _submodule1.ClassInSubmodule1()
        class_instance.method1()

class ClassInModule2:
    def __init__(self):
        pass

    def method1(self):
        print("Method 1 in ClassInModule2")
