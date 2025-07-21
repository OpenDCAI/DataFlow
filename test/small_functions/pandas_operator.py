from dataflow.operators.pandas_operator import PandasOperator
from dataflow.utils.storage import FileStorage

class Dataframe_Filter():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../../dataflow/example/GeneralTextPipeline/pandas.json",
            cache_path="./cache",
            file_name_prefix="pandas_operator",
            cache_type="jsonl",
        )

        self.dataframe_filter = PandasOperator([
                                    # 1. 新增一列 normalized_score
                                    lambda df: df.assign(normalized_score=(df["score"] - df["score"].min()) / (df["score"].max() - df["score"].min())),

                                    # # 2. 替换 comment 中的感叹号
                                    # lambda df: df.assign(comment=df["comment"].str.replace("!", ".", regex=False)),

                                    # # 3. 根据 score 增加评级列 grade
                                    # lambda df: df.assign(grade=df["score"].apply(lambda x: "A" if x >= 90 else "B" if x >= 80 else "C")),

                                    # # 4. 只保留列 id, name, score, grade
                                    # lambda df: df[["id", "name", "score", "grade"]],

                                    # # 5. 按 score 降序排列
                                    # lambda df: df.sort_values(by="score", ascending=False).reset_index(drop=True)
                            ])             

    def forward(self):
        self.dataframe_filter.run(
            storage = self.storage.step(),
        )


if __name__ == "__main__":
    model = Dataframe_Filter()
    model.forward()
