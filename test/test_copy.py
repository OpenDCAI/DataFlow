import copy
import pandas as pd

class NestedObj:
    def __init__(self, value):
        self.value = value
    

class MyObj:
    def __init__(self):
        self.my_list_int = [1, 2, 3]
        self.my_list_obj = [NestedObj(1), NestedObj(2), NestedObj(3)]
        self.my_dataframe = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

def test_copy_methods():
    # 创建对象 a
    a = MyObj()
    
    # 浅拷贝对象 a 到对象 b
    b = copy.copy(a)
    
    # 深拷贝对象 a 到对象 c
    c = copy.deepcopy(a)
    
    # 修改 b 和 c 的 list 和 dataframe
    b.my_list_int.append(4)
    b.my_list_obj[0].value = 100
    b.my_dataframe.loc[0, 'A'] = 100
    
    c.my_list_int.append(5)
    c.my_list_obj[0].value = 200
    c.my_dataframe.loc[0, 'A'] = 200
    
    # 检查 a 的 list 和 dataframe 是否发生改变
    print("Original int list:", a.my_list_int)
    print("Copied int list (shallow):", b.my_list_int)
    print("Copied int list (deep):", c.my_list_int)
    
    print("Original obj list:", [obj.value for obj in a.my_list_obj])
    print("Copied obj list (shallow):", [obj.value for obj in b.my_list_obj])
    print("Copied obj list (deep):", [obj.value for obj in c.my_list_obj])
    
    print("Original DataFrame:\n", a.my_dataframe)
    print("Copied DataFrame (shallow):\n", b.my_dataframe)
    print("Copied DataFrame (deep):\n", c.my_dataframe)

if __name__ == "__main__":
    test_copy_methods()
