from .task2vec.task2vec import Task2Vec
from .task2vec import task_similarity
import torch
import random
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from dataflow.core import TextScorer
from dataflow.utils.registry import MODEL_REGISTRY

# Task2Vec dataset diversity evaluation
# cited from: Beyond Scale: the Diversity Coefficient as a Data Quality Metric Demonstrates LLMs are Pre-trained on Formally Diverse Data
@MODEL_REGISTRY.register()
class Task2VecScorer(TextScorer):
    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.sample_nums = args_dict.get('sample_nums')
        self.sample_size = args_dict.get('sample_size')
        self.batch_size = -1
        self.device = args_dict.get('device')
        self.method = args_dict.get('method')
        self.score_type = float
        self.data_type = 'text'
        self.score_name = 'Task2VecScore'
        self.use_meta = True
        self.model_cache_dir = args_dict.get('model_cache_dir') 

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=self.model_cache_dir)
        self.probe_network = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=self.model_cache_dir)
        
        self.device = torch.device(self.device if self.device and torch.cuda.is_available() else "cpu")
        self.probe_network = self.probe_network.to(self.device)


    def preprocess(self, texts):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        tokenized_outputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return {key: value.to(self.device) for key, value in tokenized_outputs.items()}

    def evaluate_batch(self, batch):
        input_data = next(iter(batch.values()))
        embeddings = []
        data_length = len(input_data)
        for sample_num in range(self.sample_nums):
            print(f'--> {sample_num=}\n')

            indices = random.sample(range(data_length), self.sample_size)
            texts = [input_data[i] for i in indices]
            tokenized_batch = self.preprocess(texts)

            tokenized_dataset = CustomTensorDataset(tokenized_batch)

            embedding, _ = Task2Vec(self.probe_network, method=self.method).embed(tokenized_dataset)
            embeddings.append(embedding)

        distance_matrix = task_similarity.pdist(embeddings, distance='cosine')
        div_coeff, conf_interval = task_similarity.stats_of_distance_matrix(distance_matrix)
        
        return {
            "Task2VecDiversityScore": div_coeff,
            "ConfidenceInterval": conf_interval
        }


class CustomTensorDataset(Dataset):
    def __init__(self, tokenized_batch):
        self.tokenized_batch = tokenized_batch

    def __getitem__(self, index):
        return {key: self.tokenized_batch[key][index] for key in self.tokenized_batch}

    def __len__(self):
        return len(next(iter(self.tokenized_batch.values())))