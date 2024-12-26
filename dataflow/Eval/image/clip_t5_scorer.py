import torch
import os

from dataflow.core.scorer import ImageTextScorer
from dataflow.utils.image_utils import expand2square
from dataflow.utils.image_utils import load_pretrained_model, t5_tokenizer_image_token
from dataflow.Eval.image.clip_t5.model import CLIPT5ForConditionalGeneration, ModelArguments
from dataflow.utils.registry import MODEL_REGISTRY
from dataflow.utils.utils import download_model_from_hf


SYSTEM_MSG = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100


default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
default_answer_template = "Yes"

def format_question(question, conversation_style='plain'):
    if conversation_style == 't5_plain': # for 1st stage t5 model
        question = DEFAULT_IMAGE_TOKEN + question
    elif conversation_style == 't5_chat': # for 2nd stage t5 model
        question = SYSTEM_MSG + " USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: "
    elif conversation_style == 't5_chat_no_system': # for 2nd stage t5 model
        question = "USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: "
    elif conversation_style == 't5_chat_no_system_no_user': # for 2nd stage t5 model
        question = "" + DEFAULT_IMAGE_TOKEN + "\n" + question + " : "
    # elif conversation_style == 't5_chat_ood_system': # for 2nd stage t5 model
    #     question = SYSTEM_MSG + " HUMAN: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " GPT: "
    else:
        raise NotImplementedError()
    return question

def format_answer(answer, conversation_style='plain'):
    return answer

# def image_loader(image_path):
#     if image_path.split('.')[-1] == 'npy':
#         return Image.fromarray(np.load(image_path)[:, :, [2, 1, 0]], 'RGB')
#     else:
#         return Image.open(image_path).convert("RGB")

@MODEL_REGISTRY.register()
class ClipT5Scorer(ImageTextScorer):
    """A wrapper for the CLIP-FlanT5 or CLIP-T5 models"""
    def __init__(self,
                 args_dict: dict,
                 ):
        super().__init__(args_dict)
        self.model_size = args_dict['model_size']
        self.model_path = f"zhiqiulin/clip-flant5-{self.model_size}" # model path in huggingface
        self.device = args_dict["device"]
        self.context_len = args_dict['context_len']
        self.cache_dir = os.path.join(args_dict["model_cache_dir"], "clip_t5", f"clip-flant5-{self.model_size}")
        # self.cache_dir = "/mnt/petrelfs/chenjingzhou/cjz/ckpt/clip_t5/clip-flant5-xl"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        # self.image_loader = image_loader
        self.load_model()
        self.data_type = "image_caption"
        self.scorer_name = "ClipT5Scorer"


    def load_model(self):
        """Load the model, tokenizer, image transform
        """
        model_args = ModelArguments()
        model_max_length = self.context_len
        padding_side = None
        mmprojector_repo = None
        mmprojector_name = None
        self.image_aspect_ratio = 'pad'
        self.conversational_style = 't5_chat'
        
        def load_model():
            self.tokenizer, self.model, self.image_processor = load_pretrained_model(
                CLIPT5ForConditionalGeneration,
                model_args,
                # model_path=self.model_path,
                model_path=self.cache_dir, # To accelerate downloading, we download the model using "download_model_from_hf" function instead of using "load_pretrained_model"
                tokenizer_path=f'google/flan-t5-{self.model_size}',
                model_max_length=model_max_length,
                padding_side=padding_side,
                image_aspect_ratio=self.image_aspect_ratio,
                mmprojector_repo=mmprojector_repo,
                mmprojector_name=mmprojector_name,
                device=self.device,
                cache_dir=self.cache_dir
            ) 

        try:
            load_model()
        except:
            download_model_from_hf(self.model_path, self.cache_dir)
            load_model()


    # def load_images(self,
    #                 image: List[str]) -> torch.Tensor:
    #     """Load the image(s), and return a tensor (after preprocessing) put on self.device
    #     """
    #     # image = [self.image_loader(x) for x in image]
    #     image = [Image.open(x).convert("RGB") for x in image]
    #     if self.image_aspect_ratio == 'pad':
    #         image = [expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean)) for image in image]
    #     image = [self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0] for image in image]
    #     assert all(x.shape == image[0].shape for x in image)
    #     image = torch.stack(image, dim=0).to(self.device)
    #     return image

    def total_image_preprocessor(self, image) -> torch.Tensor:
        image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        return image

    # def get_image_preprocessor(self):
    #     return self.total_image_preprocessor
    
    # def get_text_preprocessor(self):
    #     return None

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def evaluate_batch(self,
                data,
                question_template: str=default_question_template,
                answer_template: str=default_answer_template) -> torch.Tensor:
        images = data[0]
        texts = data[1]
        assert len(images) == len(texts), "Number of images and texts must match"
        # Turn "a photo of a dog" into
        # Q: "Does this figure show "a photo of a dog"? Please answer yes or no."
        # A: "Yes"
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        
        # Formatting for CLIP-FlanT5 desired input including system message and image tokens
        questions = [format_question(question, conversation_style=self.conversational_style) for question in questions]
        answers = [format_answer(answer, conversation_style=self.conversational_style) for answer in answers]

        # images = self.load_images(images)
        
        input_ids = [t5_tokenizer_image_token(qs, self.tokenizer, return_tensors='pt') for qs in questions]
        labels = [t5_tokenizer_image_token(ans, self.tokenizer, return_tensors='pt') for ans in answers]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        decoder_attention_mask = labels.ne(IGNORE_INDEX)
        
        input_ids, attention_mask, decoder_attention_mask, labels = input_ids.to(self.device), \
            attention_mask.to(self.device), decoder_attention_mask.to(self.device), labels.to(self.device)
        model_input_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels,
            'images': images,
            'past_key_values': None,
            'inputs_embeds': None,
            'use_cache': None,
            'output_attentions': None,
            'output_hidden_states': None,
            'return_dict': True,
        }
        
        outputs = self.model(
            **model_input_kwargs
        )

        logits = outputs.logits
        lm_prob = torch.zeros(logits.shape[0])
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        for k in range(lm_prob.shape[0]):
            lm_prob[k] = (-loss_fct(logits[k], labels[k])).exp() # exp to cancel the log and get raw prob between 0 and 1
        return lm_prob
