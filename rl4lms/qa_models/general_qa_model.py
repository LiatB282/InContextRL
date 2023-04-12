from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

class GeneralQAModel:

    def __init__(self, model_id: str = 'EleutherAI/gpt-j-6B'):
        logger.info(f"Initialize QA model {model_id}")

        #self.model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='/home/gamir/liat/cache').eval()#.to('cuda')

        if model_id == "EleutherAI/gpt-j-6B":
            config = AutoConfig.from_pretrained(model_id)

            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config)

            model.tie_weights()

            #self.model = AutoModelForCausalLM.from_config(config)
            self.model = load_checkpoint_and_dispatch(
                model, "/home/gamir/liat/sharded-gpt-j-6B", device_map="auto", no_split_module_classes=["GPTJBlock"]
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id).eval().to('cuda')

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def generate_answer(self, prompt):
        model_inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
        attention_mask = model_inputs.attention_mask.to('cuda')
        gen_tokens = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            temperature=0,
            max_new_tokens=20,
            #pad_token_id=self.tokenizer.eos_token_id
        )
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0][len(prompt):].split('#')[0]
        return gen_text
