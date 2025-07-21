from dataflow.operators.filter import GeneralFilter
from dataflow.utils.storage import FileStorage

class Dataframe_Filter():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../../dataflow/example/GeneralTextPipeline/filtering.json",
            cache_path="./cache",
            file_name_prefix="filtering",
            cache_type="jsonl",
        )

        self.dataframe_filter = GeneralFilter([
                                lambda df: df["score"] > 0.8,
                                ### Some lambda examples:
                                # lambda df: df["score"] == 0.9,
                                # lambda df: df["score"] != 1.0,
                                # lambda df: df["score"] > 0.7,
                                # lambda df: df["score"] >= 0.75,
                                # lambda df: df["score"] < 1.1,
                                # lambda df: df["score"] <= 1.0,
                                # lambda df: (df["score"] >= 0.8) & (df["score"] <= 1.0),

                                # lambda df: df["status"].isin(["valid", "reviewed"]),
                                # lambda df: ~df["status"].isin(["pending", "rejected"]),
                                # lambda df: df["comment"].str.contains("excellent", na=False),

                                # lambda df: df["comment"].str.startswith("Good", na=False),
                                # lambda df: df["comment"].str.endswith("done", na=False),

                                # lambda df: df["remark"].isna(),
                                # lambda df: df["remark"].notna(),
                            ])             

    def forward(self):
        # Initial filters
        self.dataframe_filter.run(
            storage = self.storage.step(),
        )


if __name__ == "__main__":
    # This is the entry point for the pipeline

    model = Dataframe_Filter()
    model.forward()
