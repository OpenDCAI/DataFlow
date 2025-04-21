import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, AutoProcessor, LlavaOnevisionForConditionalGeneration, Qwen2VLForConditionalGeneration
import requests


# get model and tokenizer
def get_inference_model(model_dir):
    inference_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    inference_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
    inference_model.eval()
    return inference_tokenizer, inference_model


# get llama model and tokenizer
def get_inference_model_llama(model_dir):
    inference_model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
    inference_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    device = "cuda"
    inference_model.to(device)
    return inference_tokenizer, inference_model

def get_inference_model_llavaOneVision(model_dir):
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return model, processor, tokenizer

# get mistral model and tokenizer
def get_inference_model_mistral(model_dir):
    inference_model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
    inference_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # inference_tokenizer.pad_token = inference_tokenizer.eos_token
    device = "cuda"
    inference_model.to(device)
    return inference_tokenizer, inference_model


# get glm model response
def get_local_response(query, model, tokenizer, max_length=2048, truncation=True, do_sample=False, max_new_tokens=1024, temperature=0.7):
    cnt = 2
    all_response = ''
    while cnt:
        try:
            inputs = tokenizer([query], return_tensors="pt", truncation=truncation, max_length=max_length).to('cuda')
            output_ = model.generate(**inputs, do_sample=do_sample, max_new_tokens=max_new_tokens, temperature=temperature)
            output = output_.tolist()[0][len(inputs["input_ids"][0]):]
            response = tokenizer.decode(output)

            print(f'obtain response:{response}\n')
            all_response = response
            break
        except Exception as e:
            print(f'Error:{e}, obtain response again...\n')
            cnt -= 1
    if not cnt:
        return []
    split_response = all_response.strip().split('\n')
    return split_response


import base64
def get_local_response_api(query, tokenizer=None, max_length=2048, truncation=True, max_new_tokens=2048, temperature=0.7, do_sample=False, pipe=None, processor=None, client=None):
    cnt = 2
    
    all_response = ''
    while cnt:
        try:
            chat_response = client.chat.completions.create(
                model="qwen2.5:7b",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
            )
            all_response = chat_response.choices[0].message.content
            break
        except Exception as e:
            print(f'Error:{e}, obtain response again...\n')
            cnt -= 1
    if not cnt:
        return []
    
    split_response = all_response.split('\n')
    
    return split_response

    




# get mistral model response
def get_local_response_mistral(query, model, tokenizer, max_length=1024, truncation=True, max_new_tokens=1024, temperature=0.7, do_sample=False):
    cnt = 2
    all_response = ''
    # messages = [{"role": "user", "content": query}]
    # data = tokenizer.apply_chat_template(messages, max_length=max_length, truncation=truncation, return_tensors="pt").cuda()
    message = '[INST]' + query + '[/INST]'
    data = tokenizer.encode_plus(message, max_length=max_length, truncation=truncation, return_tensors='pt')
    input_ids = data['input_ids'].to('cuda')
    attention_mask = data['attention_mask'].to('cuda')
    while cnt:
        try:
            output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            ori_string = tokenizer.decode(output[0])
            processed_string = ori_string.split('[/INST]')[1].strip()
            response = processed_string.split('</s>')[0].strip()

            print(f'obtain response:{response}\n')
            all_response = response
            break
        except Exception as e:
            print(f'Error:{e}, obtain response again...\n')
            cnt -= 1
    if not cnt:
        return []
    all_response = all_response.split('The answer is:')[0].strip()  # intermediate steps should not always include a final answer
    ans_count = all_response.split('####')
    if len(ans_count) >= 2:
        all_response = ans_count[0] + 'Therefore, the answer is:' + ans_count[1]
    all_response = all_response.replace('[SOL]', '').replace('[ANS]', '').replace('[/ANS]', '').replace('[INST]', '').replace('[/INST]', '').replace('[ANSW]', '').replace('[/ANSW]', '')  # remove unique answer mark for mistral
    split_response = all_response.split('\n')
    return split_response
