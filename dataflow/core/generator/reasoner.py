from dataflow.data import DataFlowDataset
from dataflow.core import ScoreRecord
from datasets import Dataset

class Reasoner():
    def __init__(self, args=None):
        pass
        
    def reason_func(self, dataset):
        pass
    
    # def __call__(self, dataset: DataFlowDataset):
    #     pass

class ReasonerGenerator(Reasoner):
    def __init__(self, args=None):
        super().__init__()
        self.data_type = "text"
        self.generator_name = "ReasonerGenerator"
        self.args = args
        
        # 新增参数
        self.input_file = args.get("input_file", "")
        self.output_file = args.get("output_file", "")
        self.input_key = args.get("input_key", "")
        self.output_key = args.get("output_key", "")
        self.generator_type = args.get("generator_type", "")
        self.num_prompts = args.get("num_prompts")
        self.api_url = args.get("api_url", "")
        self.api_key = args.get("api_key", "")
        self.db_path = args.get("db_path", "")
        self.max_workers = args.get("max_workers")
        self.model_name = args.get("model_name", "")
        self.system_prompt = args.get("system_prompt", "")
        
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
            
    def generate_func(self):
        pass

    def __call__(self, dataset: DataFlowDataset):  # generate func
        init_len = len(dataset)
        score_record = ScoreRecord()
        dataset.set_score_record(score_record)
        
        # 调用生成函数生成新的数据
        generated_data = self.generate_func(dataset)
        
        if isinstance(dataset.dataset, Dataset):
            # 将生成的数据添加到原数据集中
            new_dataset = Dataset.from_dict(generated_data)
            dataset.dataset = dataset.dataset.concatenate(new_dataset)
        else:
            # 假设 dataset 有一个方法可以添加新数据
            dataset.extend(generated_data)
        
        print(f'Implemented {self.generator_name}. Data Number: {init_len} -> {len(dataset)}', flush=True)
        return dataset
