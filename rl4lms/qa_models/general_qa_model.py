from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
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
            device_map = infer_auto_device_map(model, max_memory={0: "0GIB", 1: "46GIB", 2: "46GIB", 3: "46GIB", 4: "46GIB", 5: "46GIB", 6: "46GIB", 7: "46GIB"})

            #self.model = AutoModelForCausalLM.from_config(config)
            self.model = load_checkpoint_and_dispatch(
                model, "/home/gamir/liat/sharded-gpt-j-6B", device_map="auto", no_split_module_classes=["GPTJBlock"]
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id).eval().to('cuda')

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_answer(self, prompts):
        prompts = [self.tokenizer.eos_token + prompt for prompt in prompts]
        model_inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = model_inputs.input_ids.to('cuda')
        attention_mask = model_inputs.attention_mask.to('cuda')
        gen_tokens = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            temperature=0,
            max_new_tokens=20,
            pad_token_id=self.tokenizer.eos_token_id
        )

        gen_texts = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        results = []
        for prompt, answer in zip(prompts, gen_texts):
            result = answer[len(prompt):].split('#')[0]
            results.append(result)

        return results
