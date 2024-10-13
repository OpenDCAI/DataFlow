def prepare_huggingface_model(pretrained_model_name_or_path,
                              return_model=True,
                              trust_remote_code=False):
    """
    Prepare and load a HuggingFace model with the corresponding processor.

    :param pretrained_model_name_or_path: model name or path
    :param return_model: return model or not
    :param trust_remote_code: passed to transformers
    :return: a tuple (model, input processor) if `return_model` is True;
             otherwise, only the processor is returned.
    """
    import transformers
    from transformers import AutoConfig, AutoProcessor, AutoModel
    
    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code)

    if return_model:
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code)

        model = AutoModel.from_config(config, trust_remote_code=trust_remote_code)

    return (model, processor) if return_model else processor

def wget_model(url, path):
    import os
    import subprocess
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    try:
        # Build the wget command
        command = ['wget', '-c', url, '-O', path]
        # Execute the command
        subprocess.run(command, check=True)
        
        print(f"File downloaded successfully and saved to: {path}")
        return path
    except subprocess.CalledProcessError as e:
        print(f"Error downloading the file: {e}")
        return None

def gdown_model(url, path):
    import os
    import gdown
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(path):
        gdown.download(url, path)


def _cuda_device_count():

    import torch
    return torch.cuda.device_count()


_CUDA_DEVICE_COUNT = _cuda_device_count()


def cuda_device_count():
    return _CUDA_DEVICE_COUNT


def is_cuda_available():
    return _CUDA_DEVICE_COUNT > 0
