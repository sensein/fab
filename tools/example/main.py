import sys
sys.path.append('../')


# Import the package
from tools import example


# Create instances of classes from the main package
class_instance1 = example.ClassInModule1()
class_instance2 = example.ClassInModule2()

# Call methods on the instances
class_instance1.method1()  # Method 1 in ClassInModule1
class_instance1.method2()  # Method 1 in ClassInSubmodule1
class_instance2.method1()  # Method 1 in ClassInModule2

class_instance = example.subpackage1._submodule1.ClassInSubmodule1()
class_instance.method1()
