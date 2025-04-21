import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


# define your value model class
class ChatGLM_VM(nn.Module):
    def __init__(self, base, vocab_size, num_classes=1):
        super(ChatGLM_VM, self).__init__()
        self.base_model = base
        self.LN = nn.Linear(vocab_size, num_classes, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1]
        value_outputs = self.LN(outputs)
        return value_outputs.squeeze(dim=1)


class Mistral_VM(nn.Module):
    def __init__(self, base, vocab_size=32000):
        super(Mistral_VM, self).__init__()
        self.base_model = base
        self.LN = nn.Linear(vocab_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        value_outputs = self.LN(outputs)
        return value_outputs.squeeze(dim=1)


class ChatGLM_PRM(nn.Module):
    def __init__(self, base):
        super(ChatGLM_PRM, self).__init__()
        self.base_model = base

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(outputs, dim=-1)
        output = probs[:, -1, 7081]  # n*1 tensor, 7081 is the index of token 'True'
        return output


class Mistral_PRM(nn.Module):
    def __init__(self, base):
        super(Mistral_PRM, self).__init__()
        self.base_model = base

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(outputs, dim=-1)
        output = probs[:, -1, 7081]  # n*1 tensor, 7081 is the index of token 'True'
        return output


# get value model
def get_value_model(base_model_dir, state_dict_file):
    value_tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    value_base_model = AutoModel.from_pretrained(base_model_dir, trust_remote_code=True).bfloat16().cuda()
    if state_dict_file is None:
        return value_tokenizer, value_base_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is set to: ", device, '\n')
    vocab_size = value_base_model.config.padded_vocab_size
    VM = ChatGLM_VM(value_base_model, vocab_size, 1)
    VM.load_state_dict(torch.load(state_dict_file))
    VM.to(device)
    VM.eval()
    return value_tokenizer, VM


def get_value_model_mistral(base_model_dir, state_dict_file):
    value_tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    value_tokenizer.pad_token = value_tokenizer.eos_token
    value_base_model = AutoModelForCausalLM.from_pretrained(base_model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
    if state_dict_file is None:
        return value_tokenizer, value_base_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is set to: ", device, '\n')
    vocab_size = value_base_model.config.vocab_size
    VM = Mistral_VM(value_base_model, vocab_size)
    VM.load_state_dict(torch.load(state_dict_file))
    VM.to(device)
    VM.eval()
    return value_tokenizer, VM


# get prm
def get_value_model_prm(base_model_dir, state_dict_file):
    prm_tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    prm_base_model = AutoModel.from_pretrained(base_model_dir, trust_remote_code=True).bfloat16().cuda()
    if state_dict_file is None:
        return prm_tokenizer, prm_base_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is set to: ", device, '\n')
    prm = ChatGLM_PRM(prm_base_model)
    prm.load_state_dict(torch.load(state_dict_file))
    prm.to(device)
    prm.eval()
    return prm_tokenizer, prm


def get_value_model_prm_mistral(base_model_dir, state_dict_file):
    prm_tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    prm_tokenizer.pad_token = prm_tokenizer.eos_token
    prm_base_model = AutoModelForCausalLM.from_pretrained(base_model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
    
    if state_dict_file is None:
        return prm_tokenizer, prm_base_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is set to: ", device, '\n')
    prm = Mistral_PRM(prm_base_model)
    prm.load_state_dict(torch.load(state_dict_file))
    print('** device: ', device)
    prm.to(device)
    prm.eval()
    return prm_tokenizer, prm



def get_local_value(prompt_answer, model, tokenizer, max_length=2048, low=0, high=1):
    encoded_pair = tokenizer.encode_plus(
        prompt_answer,
        # prompt_answer,
        padding='max_length',
        max_length=max_length,  
        truncation=True,
        return_tensors='pt', 
    )
    input_ids = encoded_pair['input_ids'].to('cuda')
    attention_mask = encoded_pair['attention_mask'].to('cuda')
    model = model.to('cuda')
    output = model(input_ids, attention_mask=attention_mask)
    logits = output.logits  
    first_token_logits = logits[0, 0, :]  
    probs = torch.softmax(first_token_logits, dim=-1)
    value = probs.max().item() 
    
    value = min(high, max(value, low))
    return value
