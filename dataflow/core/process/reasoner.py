from dataflow.data import DataFlowDataset
from dataflow.core import ScoreRecord
from datasets import Dataset

class Reasoner():
    def __init__(self, args=None):
        pass
        
    def reason_func(self, dataset):
        pass
    
    def __call__(self, dataset: DataFlowDataset):
        pass

class ReasonerFilter(Reasoner):
    def __init__(self, args=None):
        super().__init__()
        self.data_type = "text"
        self.filter_name = "ReasonerFilter"
        self.args = args
        
        self.input_question_key = args.get("input_question_key","")
        self.max_worker = args.get("max_worker",1)
        
        # answer format filter
        self.keys = args.get("keys","")
        # self.output_question_key = args.get("output_question_key","")
        
        # answer gt verification
        self.test_answer_key = args.get("test_answer_key","")
        self.gt_answer_key = args.get("gt_answer_key","")
        
        # ngram filter
        self.question_key = args.get("question_key","")
        self.answer_key = args.get("answer_key","")
        
        # api args
        api_args = args.get('api_args', None)
        if api_args is not None:
            self.model_name = api_args['model_name']
            self.api_url = api_args['api_url']
            self.mode_test = api_args['mode_test']
            
    def filter_func(self, dataset):
        pass

    def __call__(self, dataset: DataFlowDataset):
        """Processes the dataset using the reasoner"""
        init_len = len(dataset)
        score_record = ScoreRecord()
        dataset.set_score_record(score_record)
        labels = self.filter_func(dataset)
        if isinstance(dataset.dataset, Dataset):
            def filter_by_labels(example, index):
                return labels[index] == 1
            dataset.dataset = dataset.dataset.filter(filter_by_labels, with_indices=True)
            filtered_dataset = dataset
        else:
            filtered_dataset = dataset.filter(labels)

        print(f'Implemented {self.filter_name}. Data Number: {init_len} -> {len(filtered_dataset)}', flush=True)
        return filtered_dataset