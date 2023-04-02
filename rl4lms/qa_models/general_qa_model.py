from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


class GeneralQAModel:

    def __init__(self, model_id: str = 'EleutherAI/gpt-j-6B'):
        config = AutoConfig.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_config(config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def generate_answer(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        gen_tokens = self.model.generate(
            input_ids,
            do_sample=False,
            temperature=0,
            max_length=20,
        )
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        return gen_text
