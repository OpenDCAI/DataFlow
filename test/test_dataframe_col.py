import pandas as pd

def test_dataframe_columns():
    # 创建一个示例 DataFrame
    data = {'col1': [1, 2], 'col2': [3, 4]}
    data = {}
    dataframe = pd.DataFrame(data)

    # 使用代码行
    result = dataframe.columns.tolist() if isinstance(dataframe, pd.DataFrame) else []

    print(result)
    # 验证结果是否为 list of str
    if isinstance(result, list) and all(isinstance(col, str) for col in result):
        print("返回的是 list of str")
    else:
        print("返回的不是 list of str")

# 运行测试函数
test_dataframe_columns()
