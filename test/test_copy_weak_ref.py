import copy
import weakref

class NestedObj:
    def __init__(self, value):
        self.value = value

df = NestedObj(4)
class MyObj:
    def __init__(self):
        self.my_list_int = [1, 2, 3]
        self.my_list_obj = [NestedObj(1), NestedObj(2), NestedObj(3)]
        self.reference_attr = weakref.ref(df)  # Weak reference

    def __deepcopy__(self, memo):
        # Create a new object
        new_obj = MyObj()
        
        # Deep copy all attributes
        new_obj.my_list_int = copy.deepcopy(self.my_list_int, memo)
        new_obj.my_list_obj = copy.deepcopy(self.my_list_obj, memo)
        
        # Copy the weak reference directly
        new_obj.reference_attr = self.reference_attr
        
        return new_obj

def test_custom_deepcopy():
    # 创建对象 a
    a = MyObj()
    
    # 深拷贝对象 a 到对象 b
    b = copy.deepcopy(a)
    
    # 修改 b 的 list 和 weak reference
    b.my_list_int.append(4)
    b.my_list_obj[0].value = 100
    b.reference_attr().value = 200
    
    # 检查 a 的 list 和 reference_attr 是否发生改变
    print("Original int list:", a.my_list_int)
    print("Copied int list:", b.my_list_int)
    
    print("Original obj list:", [obj.value for obj in a.my_list_obj])
    print("Copied obj list:", [obj.value for obj in b.my_list_obj])
    
    print("Original reference_attr value:", a.reference_attr().value)
    print("Copied reference_attr value:", b.reference_attr().value)

    b.reference_attr().value = None  # Clear the weak reference in b
    print("After clearing, original reference_attr value:", a.reference_attr().value)
    print("After clearing, copied reference_attr value:", b.reference_attr().value)

if __name__ == "__main__":
    test_custom_deepcopy()
