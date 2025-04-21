from models.model import *


# given prompt, generate proposal under instruction, unwrap is required
def get_proposal(prompt, method='glm', temperature=0.7, max_tokens=2048, seed=170, max_length=2048, truncation=True,
                 do_sample=True, max_new_tokens=1024):
    response = []
    cnt = 2
    if method == 'glm':
        while not response and cnt:
            response = glm(prompt, BASE_MODEL_GLM, temperature=temperature, max_tokens=max_tokens, seed=seed)
            cnt -= 1
        if not response:
            print(f'obtain<{method}>response fail!\n')
            return []
        return response

    elif method == 'gpt':
        while not response and cnt:
            response = gpt(prompt, model=BASE_MODEL_GPT, temperature=temperature, max_tokens=max_tokens)
            cnt -= 1
        if not response:
            print(f'obtain<{method}>response fail!\n')
            return []
        return response

    elif method == 'llama' or method == 'mistral' or method == 'local':
        while not response and cnt:
            response = local_inference_model(prompt, max_length=max_length, truncation=truncation, do_sample=do_sample,
                                             max_new_tokens=max_new_tokens, temperature=temperature)
            cnt -= 1
        if not response:
            print(f'obtain<{method}>response fail!\n')
            return []
        return response

    else:
        print('This method of getting responses is not yet supported!\n')
        return []


# given prompt + answer, find its value
# if you use api, unwrap is required. if you use local value model, the value is directly obtained
def get_value(prompt_answer, method='glm', temperature=0.7, max_tokens=1000, seed=170, max_length=2048, low=0, high=1):
    response = []
    cnt = 2
    if method == 'glm':
        while not response and cnt:
            response = glm(prompt_answer, BASE_MODEL_GLM, temperature=temperature, max_tokens=max_tokens, seed=seed)
            cnt -= 1
        if not response:
            print(f'obtain<{method}>score fail!\n')
            return []
        return response

    elif method == 'gpt':
        while not response and cnt:
            response = gpt(prompt_answer, model=BASE_MODEL_GPT, temperature=temperature, max_tokens=max_tokens)
            cnt -= 1
        if not response:
            print(f'obtain<{method}>score fail!\n')
            return []
        return response

    elif method == 'local':
        value = low
        while cnt:
            try:
                value = local_value_model(prompt_answer, max_length=max_length, low=low, high=high)
                break
            except Exception as e:
                print(f'obtain<{method}>score fail!\nError:{e}\n')
                cnt -= 1
        return value

    else:
        print('This method of getting scores is not yet supported!\n')
        return []
