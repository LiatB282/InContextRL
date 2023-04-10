from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
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
        config = AutoConfig.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_config(config)
        self.model = self.model.eval()#.to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def generate_answer(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        gen_tokens = self.model.module.generate(
            input_ids,
            do_sample=False,
            temperature=0,
            max_length=20,
        )
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        return gen_text
