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
                                {"op": "custom", "value": lambda df: (df["score"] > 0.8) & df["comment"].str.contains("excellent", na=False)},
                                # { "key": "score", "op": "==", "value": 0.9 },
                                # { "key": "score", "op": "!=", "value": 1.0 },
                                # { "key": "score", "op": ">", "value": 0.7 },
                                # { "key": "score", "op": ">=", "value": 0.75 },
                                # { "key": "score", "op": "<", "value": 1.1 },
                                # { "key": "score", "op": "<=", "value": 1.0 },
                                # { "key": "score", "op": "range", "value": [0.8, 1.0] },

                                # { "key": "status", "op": "in", "value": ["valid", "reviewed"] },
                                # { "key": "status", "op": "not in", "value": ["pending", "rejected"] },

                                # { "key": "comment", "op": "contains", "value": "excellent" },
                                # { "key": "comment", "op": "startswith", "value": "Good" },
                                # { "key": "comment", "op": "endswith", "value": "done" },
                                # { "key": "remark", "op": "isna"},
                                # { "key": "remark", "op": "notna"}
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
