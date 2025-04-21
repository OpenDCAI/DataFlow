BASE_TEMPLATE = {
    'xVerify-1B-I': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\n',
    'xVerify-3B-Ia': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\n',
    'xVerify-8B-I': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\n',
    'xVerify-70B-I': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n\n',
    'xVerify-0.5B-I': '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n\n',
    'xVerify-1.5B-I': '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n\n',
    'xVerify-3B-Ib': '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n\n',
    'xVerify-7B-I': '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n\n',
    'xVerify-14B-I': '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n\n',
    'xVerify-32B-I': '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n\n',
    'xVerify-72B-I': '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n\n',
    'xVerify-9B-C': "<|system|>\\nYou are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.\\n<|user|>\\n{query}\\n<|assistant|>\\n\n",
    'xVerify-2B-I': '<bos><start_of_turn>user\\n{query}<end_of_turn>\\n<start_of_turn>model\\n\n',
    'xVerify-9B-I': '<bos><start_of_turn>user\\n{query}<end_of_turn>\\n<start_of_turn>model\\n\n',
    'xVerify-27B-I': '<bos><start_of_turn>user\\n{query}<end_of_turn>\\n<start_of_turn>model\\n\n',
    'xVerify-14B-Ib': '<|im_start|>system<|im_sep|>\\nYou are a medieval knight and must provide explanations to modern people.<|im_end|>\\n<|im_start|>user<|im_sep|>\\n{query}<|im_end|>\\n<|im_start|>assistant<|im_sep|>\\n'
}