import os
import shutil
import pandas as pd


# 假设你的 FileStorage 已经在当前作用域可用
from dataflow.utils.storage import FileStorage

def main():
    cache_dir = "./cache_test_xlsx"
    os.makedirs(cache_dir, exist_ok=True)

    # 1) 构造一个 storage：first_entry_file_name 随便给个占位
    storage = FileStorage(
        first_entry_file_name="",          # 我们这里不读 step0
        cache_path=cache_dir,
        file_name_prefix="demo",
        cache_type="xlsx",
    ).reset()

    # 2) step0：这里仅用于把 operator_step 变成 0（符合你类的约束）
    storage.step()

    # 3) 准备数据并写入（写到 step1 文件）
    df_write = pd.DataFrame([
        {"id": 1, "text": "你好", "score": 0.95},
        {"id": 2, "text": "世界", "score": 0.88},
    ])

    out_path = storage.write(df_write)
    print("Wrote:", out_path)
    assert out_path.endswith(".xlsx")
    assert os.path.exists(out_path)

    # 4) step1：读取刚写入的文件
    storage.step()
    df_read = storage.read(output_type="dataframe")

    print("Read back:")
    print(df_read)

    # 5) 简单校验（xlsx 读回来的 dtype 可能略有变化，所以用值校验）
    assert df_read.shape == df_write.shape
    assert df_read["id"].tolist() == df_write["id"].tolist()
    assert df_read["text"].tolist() == df_write["text"].tolist()

    # 6) 清理（可选）
    # shutil.rmtree(cache_dir, ignore_errors=True)
    print("OK ✅")

if __name__ == "__main__":
    main()
