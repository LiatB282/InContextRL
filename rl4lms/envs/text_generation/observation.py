from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from transformers import AutoTokenizer
from rl4lms.data_pools.text_generation_pool import Sample
from copy import deepcopy
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from index_utils.retriever import DenseRetriever

FULL_SIZE = 700

@dataclass
class Observation:
    # encoded input
    prompt_or_input_encoded_pt: torch.tensor
    # attention mask for the input
    prompt_or_input_attention_mask_pt: torch.tensor
    # input text
    prompt_or_input_text: str
    # encoded context
    context_encoded_pt: torch.tensor
    # attention mask for the context
    context_attention_mask_pt: torch.tensor
    # context text
    context_text: str
    # reference texts
    target_or_reference_texts: List[str]

    # concatenated input
    input_encoded_pt: torch.tensor
    input_attention_mask_pt: torch.tensor

    # list of actions
    action_history: List[str]

    # other meta info
    meta_info: Dict[str, Any]

    index: int

    def to_dict(self) -> Dict[str, torch.tensor]:
        """
        For stable baselines (only return tensor items)
        """
        dict_obs = {
            "prompt_or_input_encoded_pt": self.prompt_or_input_encoded_pt.numpy().flatten(),
            "prompt_or_input_attention_mask_pt": self.prompt_or_input_attention_mask_pt.numpy().flatten(),
            "context_encoded_pt": self.context_encoded_pt.numpy().flatten(),
            "context_attention_mask_pt": self.context_attention_mask_pt.numpy().flatten(),
            "input_encoded_pt": self.input_encoded_pt.numpy().flatten(),
            "input_attention_mask_pt": self.input_attention_mask_pt.numpy().flatten(),
            "index": self.index
        }
        return dict_obs

    @staticmethod
    def _concat(prompt: torch.tensor, prompt_mask: torch.tensor,
                context: torch.tensor, context_mask: torch.tensor,
                pad_token: int):

        prompt_ = prompt[:, prompt_mask.flatten().bool().tolist()]
        context_ = context[:, context_mask.flatten().bool().tolist()]
        actual_size = prompt_.shape[1] + context_.shape[1]

        full_size = prompt.shape[1] + context.shape[1]
        concatenated = torch.full(
            (full_size,), fill_value=pad_token).reshape(1, -1)
        concatenated_mask = torch.zeros((1, full_size)).int()

        concatenated[:, full_size -
                     actual_size:] = torch.cat((prompt_, context_), dim=1)
        concatenated_mask[:, full_size -
                          actual_size:] = 1
        return concatenated, concatenated_mask

    def update(self, action: int, tokenizer: AutoTokenizer, retriver: DenseRetriever) -> "Observation":
        """
        Updates the observation using the given action
        """

        # update the action history
        current_action_history = deepcopy(self.action_history)
        current_action_history.append(action)

        # get the current context
        current_prompt = deepcopy(self.prompt_or_input_encoded_pt)
        # current_context_attention_mask = deepcopy(
        #     self.context_attention_mask_pt)

        input_ids = retriver.get_input_ids_from_docs_id(action)

        input_ids = torch.masked_select(input_ids, input_ids != 0)
        input_ids[-1] = 1713
        current_prompt = torch.masked_select(current_prompt, current_prompt != 0)
        current_prompt = torch.cat([input_ids.to(current_prompt.device), current_prompt], dim=0)
        
        #current_prompt_attention_mask = current_prompt != tokenizer.pad_token_id

        # # just shift the context (also the attention mask) to left by 1
        # current_context[:, 0:-1] = current_context[:, 1:].clone()
        # current_context_attention_mask[:, 0:-
        #                                1] = current_context_attention_mask[:, 1:].clone()

        # # add the action always at the end (assumes left padding)
        # current_context[:, -1] = action
        # current_context_attention_mask[:, -1] = 1

        # decode the context
        prompt_text = tokenizer.decode(
            current_prompt.flatten(), skip_special_tokens=True)

        # concatenate and still keep the left padding
        # input_encoded_pt, input_attention_mask_pt = Observation._concat(
        #     current_prompt, current_prompt_attention_mask,
        #     self.context_encoded_pt, self.context_attention_mask_pt,
        #     tokenizer.pad_token_id)

        full_size_prompt = torch.zeros((1, FULL_SIZE), device=current_prompt.device).int()
        full_size_prompt[:, (FULL_SIZE - current_prompt.shape[0]):] = current_prompt #TODO: check the shape
        current_prompt = full_size_prompt
        current_prompt_attention_mask = current_prompt != tokenizer.pad_token_id

        # and create a new observation
        obs = Observation(current_prompt,
                          current_prompt_attention_mask,
                          prompt_text,
                          self.context_encoded_pt,
                          self.context_attention_mask_pt,
                          self.context_text,
                          self.target_or_reference_texts,
                          current_prompt,
                          current_prompt_attention_mask,
                          current_action_history,
                          self.meta_info,
                          self.index)

        return obs

    @ classmethod
    def init_from_sample(cls, sample: Sample,
                         tokenizer: AutoTokenizer,
                         max_input_length: int,
                         max_context_length: int,
                         prompt_truncation_side: str,
                         context_start_token: int = None,
                         meta_info: Dict[str, Any] = None,
                         sample_index: int = None):
        # encode the prompt text
        # override truncation side for prompt
        prev_truncation_side = tokenizer.truncation_side
        tokenizer.truncation_side = prompt_truncation_side
        prompt_outputs = tokenizer(sample.prompt_or_input_text,
                                   padding="max_length",
                                   max_length=max_input_length,
                                   return_tensors="pt",
                                   return_attention_mask=True,
                                   truncation=True)
        tokenizer.truncation_side = prev_truncation_side
        

        # for seq2seq models, context should be initialized to start token if provided
        if context_start_token is not None:
            context_outputs = tokenizer("",
                                    padding="max_length",
                                    max_length=max_context_length,
                                    return_tensors="pt",
                                    return_attention_mask=True)
            context_outputs.input_ids = torch.ones(1, max_context_length, dtype=torch.int32) * tokenizer.pad_token_id
            context_outputs.input_ids[:, -1] = context_start_token
            context_outputs.attention_mask = torch.zeros(1, max_context_length, dtype=torch.int32)
            context_outputs.attention_mask[:,-1] = 1
        else:
            context_outputs = tokenizer("",
                                    padding="max_length",
                                    max_length=max_context_length,
                                    return_tensors="pt",
                                    return_attention_mask=True)

        # concatenate
        # input_encoded_pt, input_attention_mask_pt = Observation._concat(
        #     prompt_outputs.input_ids, prompt_outputs.attention_mask,
        #     context_outputs.input_ids, context_outputs.attention_mask,
        #     tokenizer.pad_token_id)

        obs = Observation(prompt_or_input_encoded_pt=prompt_outputs.input_ids,
                          prompt_or_input_attention_mask_pt=prompt_outputs.attention_mask,
                          prompt_or_input_text=sample.prompt_or_input_text,
                          context_encoded_pt=context_outputs.input_ids,
                          context_attention_mask_pt=context_outputs.attention_mask,
                          input_encoded_pt=prompt_outputs.input_ids,
                          input_attention_mask_pt=prompt_outputs.attention_mask,
                          context_text="",
                          target_or_reference_texts=sample.references,
                          action_history=[],
                          meta_info=meta_info,
                          index=sample_index)

        return obs


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    sample = Sample("1", "Hello, this is cool", ["it is good", "going well"])

    obs = Observation.init_from_sample(
        sample=sample,
        tokenizer=tokenizer,
        max_input_length=24,
        max_context_length=24
    )
    updated_obs = obs.update(10, tokenizer)
    updated_obs = updated_obs.update(11, tokenizer)
