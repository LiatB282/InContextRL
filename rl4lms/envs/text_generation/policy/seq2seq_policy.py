from typing import Any, Dict, Optional, List, Union
import torch
from gym.spaces import Discrete
from gym.spaces.dict import Dict as DictSpace
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5EncoderModel
from stable_baselines3.common.distributions import CategoricalDistribution
from torch.distributions import Categorical
from copy import deepcopy
from rl4lms.algorithms.common.maskable.distributions import (
    MaskableCategoricalDistribution,
)
from rl4lms.envs.text_generation.hf_generation_utils import override_generation_routines
from stable_baselines3.common.type_aliases import TensorDict, Schedule
from rl4lms.algorithms.common.maskable.logits_processor import (
    MaskLogitsProcessorSeq2SeqLM,
)
from rl4lms.envs.text_generation.warm_start import (
    ActorCriticWarmStartMixin,
    MaskableActorCriticWarmStartMixin,
)
from transformers.modeling_utils import unwrap_model
from rl4lms.envs.text_generation.policy.base_policy import (
    GenerationInputs,
    LMActorCriticPolicy,
    PolicyOutput,
    RefPolicyOutput,
    ValueOutput,
    PolicyType,
    EvaluateActionsOutput,
    GenerationOutputs,
)
import sys
import os
import numpy

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from index_utils.retriever import DenseRetriever

class Seq2SeqLMActorCriticPolicy(LMActorCriticPolicy, ActorCriticWarmStartMixin):
    def __init__(
        self,
        observation_space: DictSpace,
        action_space: Discrete,
        lr_schedule: Schedule,
        model_name: str,
        optimizer_kwargs: Dict[str, Any] = {},
        weight_decay: float = 1e-6,
        use_sde: bool = None,
        apply_model_parallel: bool = True,
        optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
        generation_kwargs: Dict[str, Any] = {},
        prompt_truncation_side: str = "left",
        state_dict: Dict[str, Any] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            model_name,
            optimizer_kwargs,
            weight_decay,
            use_sde,
            apply_model_parallel,
            optimizer_class,
            generation_kwargs,
            prompt_truncation_side,
        )
        self.load_from_dict(state_dict)

    def _build_model_heads(self, model_name: str):
        self._policy_model = T5EncoderModel.from_pretrained(model_name).cuda()
        # self._policy_model.__class__ = override_generation_routines(
        #     type(self._policy_model)
        # )

        self._value_model = T5EncoderModel.from_pretrained(model_name).cuda()
        self._ref_model = deepcopy(self._policy_model).eval()

        self._value_head = nn.Linear(
            self._value_model.config.hidden_size, 1, bias=False
        )

        # apply model parallel
        if torch.cuda.is_available():
            if self._apply_model_parallel and self._policy_model.is_parallelizable:
                self._policy_model.parallelize()
                self._ref_model.parallelize()
                self._value_model.parallelize()
                self._value_head = self._value_head.to(self.device)
            else:  # else defaults to data parallel
                self._policy_model = torch.nn.DataParallel(self._policy_model)
                self._ref_model = torch.nn.DataParallel(self._ref_model)
                self._value_model = torch.nn.DataParallel(self._value_model)
                self._value_head = torch.nn.DataParallel(
                    self._value_head.to(self.device)
                )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def generate_by_embeds(self,
        tokenizer: AutoTokenizer,
        texts: List[str],
        query_ids: List[int],
        max_steps: int,
        input_ids: torch.tensor = None,
        attention_mask: torch.tensor = None,
        retriever: DenseRetriever = None):
        
        batch_size = input_ids.shape[0]
        used_ids = [[query_ids[i]] for i in range(batch_size)]
        actions = [[] for _ in range(batch_size)]
        step_wise_logprobs = []
        step_wise_actions = []
        step_wise_actions_idx = []
        all_doc_embeds = []
        all_doc_ids = []
        finished = [False for _ in range(batch_size)]
        finished_counter = 0
        max_top_queries = 100

        #decoder_input_ids = torch.tensor(tokenizer.pad_token_id).repeat(batch_size, 1).to(input_ids.device)
        for step in range(max_steps):
            if finished_counter == batch_size:
                break

            outputs = self._policy_model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True#, output_hidden_states=True, decoder_input_ids=decoder_input_ids
            )

            embeddings = self.mean_pooling(outputs, attention_mask)

            #last_hidden_state = outputs['last_hidden_states'][-1].squeeze(1)

            embeddings = torch.nn.functional.normalize(embeddings, dim=1)

            top_queries = retriever.get_top_docs(embeddings.cpu().detach().numpy(), max_top_queries + step + 1) #TODO: check that they are ordered
            rand_docs_vectors_list = []
            rand_docs_ids_list = []
            top_docs_ids_list = []
            top_scores_list = []
            top_docs_vectors_list = []
            input_ids_list = []

            for i in range(batch_size):
                # TODO: make this true
                if top_queries[i][0][0][0] == 0 and not finished[i]:
                    finished[i] = True
                    finished_counter += 1

                current_ids = used_ids[i]
                current_scores = []
                current_top_ids = []

                counter = 0
                query_input_ids = None

                # Choosing the real top-100 filtering used queries
                for j, q in enumerate(top_queries[i][0]):
                    id = q[0]
                    if id in actions[i] or id in used_ids[i]:
                        continue

                    if query_input_ids is None:
                        query_input_ids = q[1]
                    
                    score = top_queries[i][1][j]
                    current_scores.append(score)
                    current_top_ids.append(id)
                    counter += 1
                    if counter == max_top_queries:
                        break

                top_docs_ids_list.append(current_top_ids)
                top_scores_list.append(current_scores)

                query_id = current_top_ids[0]

                actions[i].append(current_top_ids[0])
                
                if not finished[i]:
                    current_ids.append(query_id)
                    query_input_ids = torch.masked_select(query_input_ids, query_input_ids != 0)
                    query_input_ids[-1] = 1713
                    current_input_ids = torch.masked_select(input_ids[i], input_ids[i] != 0)
                    new_input_ids = torch.cat([query_input_ids.cuda(), current_input_ids], dim=0)
                    input_ids_list.append(new_input_ids)  


                rand_docs_data_and_vectors = retriever.get_random_docs(1000, current_ids + current_top_ids)
                rand_docs_vectors_list.append([d[1] for d in rand_docs_data_and_vectors])
                rand_docs_ids_list.append([d[0][0] for d in rand_docs_data_and_vectors])
                top_docs_vectors_list.append(retriever.get_embeds_from_docs_ids(current_top_ids)) #check

            rand_docs_tensor = torch.tensor(numpy.array(rand_docs_vectors_list, dtype=numpy.float32), device=input_ids.device)
            top_docs_tensor = torch.tensor(numpy.array(top_docs_vectors_list, dtype=numpy.float32), device=input_ids.device)

            rand_docs_scores = torch.bmm(embeddings.unsqueeze(1), rand_docs_tensor.transpose(1,2)).squeeze(1)

            top_scores_tensor = torch.tensor(top_scores_list, device=input_ids.device)
            all_scores = torch.cat([rand_docs_scores, top_scores_tensor], dim=1) 
            all_actions = [l1 + l2 for l1, l2 in zip(rand_docs_ids_list, top_docs_ids_list)]
            all_actions = torch.tensor(all_actions, device=input_ids.device)
            all_vectors = torch.cat([rand_docs_tensor, top_docs_tensor], dim=1)

            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True)
            attention_mask = input_ids != tokenizer.pad_token_id    

            # all_logprobs = torch.log_softmax(all_scores, dim=1)
            # logprobs, current_actions_indices = torch.max(all_logprobs, dim=1)
            # current_actions = torch.gather(all_actions, current_actions_indices, dim=1)
            # step_wise_logprobs.append(logprobs)
            # step_wise_actions.append(current_actions)
            all_doc_ids.append(all_actions)
            all_doc_embeds.append(all_vectors)

            actions_idx_at_step = torch.argmax(all_scores, dim=1)
            distribution = Categorical(logits=all_scores)
            log_probs = distribution.log_prob(actions_idx_at_step)
            step_wise_logprobs.append(log_probs)
            actions_at_step = torch.gather(all_actions, 1, actions_idx_at_step.unsqueeze(1)).squeeze(1)
            step_wise_actions.append(actions_at_step)
            step_wise_actions_idx.append(actions_idx_at_step)

        texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        logger.info(f"Generated text = {texts[0]}")

        gen_output = GenerationOutputs(
            step_wise_logprobs, step_wise_actions, step_wise_actions_idx, texts, doc_ids=all_doc_ids, doc_embeds=all_doc_embeds
        )
        return gen_output

    def generate(
        self,
        tokenizer: AutoTokenizer,
        texts: List[str] = None,
        query_ids: List[int] = None,
        max_steps: int = None,
        input_ids: torch.tensor = None,
        attention_mask: torch.tensor = None,
        gen_kwargs: Dict[str, Any] = None,
        retriever: DenseRetriever = None
    ) -> GenerationOutputs:

        # if it different from rollout gen kwargs
        if gen_kwargs is None:
            gen_kwargs = self._generation_kwargs

        # switch to eval
        self._policy_model.eval()

        if (
            input_ids is None
            and attention_mask is None
            and texts is not None
            #and max_prompt_length is not None
        ):
            # override truncation side for prompt
            prev_truncation_side = tokenizer.truncation_side
            tokenizer.truncation_side = self._prompt_truncation_side
            encodings = tokenizer(
                texts,
                padding="max_length",
                max_length=700,
                return_tensors="pt",
                return_attention_mask=True,
                truncation=True,
            )
            input_ids = encodings.input_ids.cuda()
            attention_mask = encodings.attention_mask.cuda()
            tokenizer.truncation_side = prev_truncation_side

        # if min_length argument is set and if policy is not a seq2seq LM (ie. causal LM)
        # then it has to be adjusted to input_size + min_length
        if "min_length" in gen_kwargs.keys() and not self.is_encoder_decoder(
            self._policy_model
        ):
            generation_kwargs_ = deepcopy(gen_kwargs)
            generation_kwargs_["min_length"] = (
                input_ids.shape[1] + gen_kwargs["min_length"]
            )
        else:
            generation_kwargs_ = gen_kwargs
        
        return self.generate_by_embeds(tokenizer, texts, query_ids, max_steps, input_ids, attention_mask, retriever)

    def forward_policy(
        self,
        obs: TensorDict,
        actions: torch.tensor,
        actions_idx: torch.tensor,
        doc_embeds: torch.tensor,
        past_model_kwargs: Optional[Dict[str, torch.tensor]] = None
    ) -> PolicyOutput:

        # Temp workaround for Seq2seq policy
        # past_model_kwargs = None

        # if past_model_kwargs is None:
        #     # 1. prepare model inputs
        #     past_model_kwargs = {
        #         "attention_mask": obs["prompt_or_input_attention_mask_pt"],
        #     }
        #     inputs_tensor, model_input_name, past_model_kwargs = unwrap_model(
        #         self._policy_model
        #     )._prepare_model_inputs(
        #         obs["prompt_or_input_encoded_pt"].int(), None, past_model_kwargs
        #     )

        #     # 2. prepare encoder outputs
        #     past_model_kwargs = unwrap_model(
        #         self._policy_model
        #     )._prepare_encoder_decoder_kwargs_for_generation(
        #         inputs_tensor, past_model_kwargs, model_input_name
        #     )

        #     # 3. Prepare input_ids for auto-regressive generation
        #     input_ids = obs["context_encoded_pt"].int()
        #     decoder_attn_mask = obs["context_attention_mask_pt"]
        # else:
        #     input_ids = obs["context_encoded_pt"].int()
        #     decoder_attn_mask = past_model_kwargs.pop("decoder_attention_mask")

         
        # all set to get into auto-regressive mode
        # prepare all of the model inputs for the decoder
        # batch_size = input_ids.shape[0]
        # model_inputs = unwrap_model(self._policy_model).prepare_inputs_for_generation(
        #     input_ids, **past_model_kwargs
        # )

        # attention_mask = input_ids != 0
        input_ids = obs["prompt_or_input_encoded_pt"].int()
        batch_size = input_ids.shape[0]
        #decoder_input_ids = torch.tensor(0, device=input_ids.device).repeat(batch_size, 1)

        
        # and forward pass to get next token logits
        # outputs = self._policy_model(
        #     input_ids=input_ids, attention_mask=obs["prompt_or_input_attention_mask_pt"], decoder_input_ids=decoder_input_ids, return_dict=True, output_hidden_states=True
        # )

        # last_hidden_state = outputs['decoder_hidden_states'][-1].squeeze(1)
        # embeddings = torch.nn.functional.normalize(last_hidden_state, dim=1)
        attention_mask=obs["prompt_or_input_attention_mask_pt"]

        outputs = self._policy_model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True#, output_hidden_states=True, decoder_input_ids=decoder_input_ids
            )

        embeddings = self.mean_pooling(outputs, attention_mask)

        #last_hidden_state = outputs['last_hidden_states'][-1].squeeze(1)

        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        
        scores = torch.bmm(embeddings.unsqueeze(1), doc_embeds.transpose(1,2).float()).squeeze(1).float()

        # get log probs
        dist = self._action_dist.proba_distribution(action_logits=scores)
        log_prob = dist.log_prob(actions_idx)
        entropy = dist.entropy()

        # update the model kwargs for further generation
        # past_model_kwargs = unwrap_model(
        #     self._policy_model
        # )._update_model_kwargs_for_generation(
        #     outputs,
        #     past_model_kwargs,
        #     is_encoder_decoder=unwrap_model(
        #         self._policy_model
        #     ).config.is_encoder_decoder,
        # )
        # past_model_kwargs["decoder_attention_mask"] = torch.cat(
        #     (decoder_attn_mask, torch.ones(batch_size, 1).to(decoder_attn_mask.device)),
        #     dim=-1,
        # )

        policy_output = PolicyOutput(
            actions, actions_idx, log_prob, log_prob, entropy, None
        )

        return policy_output

    def forward_value(
        self,
        obs: TensorDict,
        past_model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ) -> ValueOutput:
        # Temp workaround for Seq2seq policy
        past_model_kwargs = None

        # if past_model_kwargs is None:
        #     # 1. prepare model inputs
        #     past_model_kwargs = {
        #         "attention_mask": obs["prompt_or_input_attention_mask_pt"],
        #     }
        #     inputs_tensor, model_input_name, past_model_kwargs = unwrap_model(
        #         self._value_model
        #     )._prepare_model_inputs(
        #         obs["prompt_or_input_encoded_pt"].int(), None, past_model_kwargs
        #     )

        #     # 2. prepare encoder outputs
        #     past_model_kwargs = unwrap_model(
        #         self._value_model
        #     )._prepare_encoder_decoder_kwargs_for_generation(
        #         inputs_tensor, past_model_kwargs, model_input_name
        #     )

        #     # 3. Prepare input_ids for auto-regressive generation
        #     input_ids = obs["context_encoded_pt"].int()
        #     #decoder_attn_mask = obs["context_attention_mask_pt"]
        # else:
        #     input_ids = obs["context_encoded_pt"].int()
        #     decoder_attn_mask = past_model_kwargs.pop("decoder_attention_mask")

        # # all set to get into auto-regressive mode
        # # prepare all of the model inputs for the decoder
        # batch_size = input_ids.shape[0]
        # model_inputs = unwrap_model(self._value_model).prepare_inputs_for_generation(
        #     input_ids, **past_model_kwargs
        # )

        input_ids = obs["prompt_or_input_encoded_pt"].int()
        attention_mask = obs["prompt_or_input_attention_mask_pt"]

        outputs = self._value_model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True#, output_hidden_states=True, decoder_input_ids=decoder_input_ids
            )

        embeddings = self.mean_pooling(outputs, attention_mask)

        #last_hidden_state = outputs['last_hidden_states'][-1].squeeze(1)

        #embeddings = torch.nn.functional.normalize(embeddings, dim=1)


        #model_inputs["decoder_input_ids"] = torch.tensor(0, device=input_ids.device).repeat(batch_size, 1)

        # and forrward pass to get hidden states
        # outputs = self._value_model(
        #     **model_inputs,
        #     output_hidden_states=True,
        #     #decoder_attention_mask=decoder_attn_mask,
        #     return_dict=True
        # )

        # # get decoder's last hidden state
        # last_tokens_hidden = outputs.decoder_hidden_states[-1][:, -1, :].to(self.device)
        values = self._value_head.forward(embeddings)

        # update the model kwargs for further generation
        # past_model_kwargs = unwrap_model(
        #     self._value_model
        # )._update_model_kwargs_for_generation(
        #     outputs,
        #     past_model_kwargs,
        #     is_encoder_decoder=unwrap_model(
        #         self._value_model
        #     ).config.is_encoder_decoder,
        # )
        # past_model_kwargs["decoder_attention_mask"] = torch.cat(
        #     (decoder_attn_mask, torch.ones(batch_size, 1).to(decoder_attn_mask.device)),
        #     dim=-1,
        # )

        value_output = ValueOutput(values, None)
        return value_output

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor, actions_idx: torch.Tensor, embeds: torch.Tensor
    ) -> EvaluateActionsOutput:

        policy_outputs = self.forward_policy(obs=obs, actions=actions, actions_idx=actions_idx, doc_embeds=embeds.reshape(-1, 1100, 768))
        value_outputs = self.forward_value(obs)

        eval_outputs = EvaluateActionsOutput(
            values=value_outputs.values,
            log_prob=policy_outputs.log_probs,
            entropy=policy_outputs.entropy,
        )
        return eval_outputs

    def to(self, device: str):
        if self._apply_model_parallel:
            self._value_head = self._value_head.to(device)
            return self
        else:
            return super().to(device)

    def get_log_probs_ref_model(
        self,
        obs: TensorDict,
        action: torch.tensor,
        action_idx: torch.tensor,
        doc_embeds: torch.tensor,
        model_kwarpast_model_kwargsgs: Dict[str, Any] = None
    ) -> RefPolicyOutput:
        # Temp workaround for Seq2seq policy
        # past_model_kwargs = None

        # if past_model_kwargs is None:
        #     # 1. prepare model inputs
        #     past_model_kwargs = {
        #         "attention_mask": obs["prompt_or_input_attention_mask_pt"],
        #     }
        #     inputs_tensor, model_input_name, past_model_kwargs = unwrap_model(
        #         self._ref_model
        #     )._prepare_model_inputs(
        #         obs["prompt_or_input_encoded_pt"].int(), None, past_model_kwargs
        #     )

        #     # 2. prepare encoder outputs
        #     past_model_kwargs = unwrap_model(
        #         self._ref_model
        #     )._prepare_encoder_decoder_kwargs_for_generation(
        #         inputs_tensor, past_model_kwargs, model_input_name
        #     )

        #     # 3. Prepare input_ids for auto-regressive generation
        #     input_ids = obs["context_encoded_pt"].int()
        #     #decoder_attn_mask = obs["context_attention_mask_pt"]
        # else:
        #     input_ids = obs["context_encoded_pt"].int()
        #     decoder_attn_mask = past_model_kwargs.pop("decoder_attention_mask")

        # # all set to get into auto-regressive mode
        # # prepare all of the model inputs for the decoder
        # batch_size = input_ids.shape[0]
        # model_inputs = unwrap_model(self._ref_model).prepare_inputs_for_generation(
        #     input_ids, **past_model_kwargs
        # )

        # model_inputs["decoder_input_ids"] = torch.tensor(0, device=input_ids.device).repeat(batch_size, 1)

        # # and forward pass to get next token logits
        # outputs = self._ref_model(
        #     **model_inputs, return_dict=True, output_hidden_states=True
        # )

        # last_hidden_state = outputs['decoder_hidden_states'][-1].squeeze(1)

        input_ids = obs["prompt_or_input_encoded_pt"].int()
        attention_mask = obs["prompt_or_input_attention_mask_pt"]

        outputs = self._ref_model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True#, output_hidden_states=True, decoder_input_ids=decoder_input_ids
            )

        embeddings = self.mean_pooling(outputs, attention_mask)

        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        
        scores = torch.bmm(embeddings.unsqueeze(1), doc_embeds.transpose(1,2)).squeeze(1)

        # get log probs
        dist = self._action_dist.proba_distribution(action_logits=scores)
        log_prob = dist.log_prob(action_idx)

        # update the model kwargs for further generation
        # past_model_kwargs = unwrap_model(
        #     self._ref_model
        # )._update_model_kwargs_for_generation(
        #     outputs,
        #     past_model_kwargs,
        #     is_encoder_decoder=unwrap_model(self._ref_model).config.is_encoder_decoder,
        # )
        # past_model_kwargs["decoder_attention_mask"] = torch.cat(
        #     (decoder_attn_mask, torch.ones(batch_size, 1).to(decoder_attn_mask.device)),
        #     dim=-1,
        # )

        ref_policy_output = RefPolicyOutput(log_prob, None)

        return ref_policy_output

    def get_policy_first_device(self):
        return (
            self._policy_model.get_encoder().first_device
            if self._apply_model_parallel
            else self.device
        )

    def get_inputs_for_generation(self, obs: TensorDict) -> GenerationInputs:

        generation_inputs = GenerationInputs(
            obs["prompt_or_input_encoded_pt"], obs["prompt_or_input_attention_mask_pt"]
        )
        return generation_inputs

    def get_policy_type(self):
        return PolicyType.SEQ2SEQ


class MaskedSeq2SeqLMActorCriticPolicy(
    Seq2SeqLMActorCriticPolicy, MaskableActorCriticWarmStartMixin
):
    def __init__(
        self,
        observation_space: DictSpace,
        action_space: Discrete,
        lr_schedule: Schedule,
        model_name: str,
        optimizer_kwargs: Dict[str, Any] = {},
        weight_decay: float = 1e-6,
        use_sde: bool = None,
        apply_model_parallel: bool = True,
        optimizer_class: torch.optim = torch.optim.AdamW,
        generation_kwargs: Dict[str, Any] = {},
        top_mask: Union[int, float] = None,
        mask_type: str = "learned_top_k",
        target_update_iterations: int = 1000,
        prompt_truncation_side: str = "left",
        state_dict: Dict[str, Any] = None,
        min_tokens_to_keep: int = 100,
    ):
        self.min_tokens_to_keep = min_tokens_to_keep
        self.mask_type = mask_type
        self.top_mask = top_mask if top_mask != -1 else self._action_space.n
        self.target_update_iterations = target_update_iterations
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            model_name,
            optimizer_kwargs,
            weight_decay,
            use_sde,
            apply_model_parallel,
            optimizer_class,
            generation_kwargs,
            prompt_truncation_side,
            state_dict,
        )

        self._action_dist = MaskableCategoricalDistribution(self._action_space.n)
        self._ref_action_dist = CategoricalDistribution(self._action_space.n)
        self._mask_action_dist = CategoricalDistribution(self._action_space.n)
        self.all_special_ids = None

    def _build_model_heads(self, model_name: str):
        super()._build_model_heads(model_name)
        if "learned" in self.mask_type:
            self._mask_model = deepcopy(self._policy_model).eval()
        else:
            self._mask_model = self._ref_model.eval()

        if torch.cuda.is_available():
            if (
                self._apply_model_parallel
                and unwrap_model(self._mask_model).is_parallelizable
            ):
                self._mask_model.parallelize()

        self.logits_processor = MaskLogitsProcessorSeq2SeqLM(
            self._mask_model,
            self.action_space,
            self.top_mask,
            self._apply_model_parallel,
            self.get_policy_first_device,
            self.mask_type,
            self.min_tokens_to_keep,
        )

    def forward_policy(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        action_masks: torch.Tensor = None,
        past_model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ) -> PolicyOutput:

        # Temp workaround for Seq2seq policy
        past_model_kwargs = None

        if past_model_kwargs is None:
            # 1. prepare model inputs
            past_model_kwargs = {
                "attention_mask": obs["prompt_or_input_attention_mask_pt"],
            }
            (inputs_tensor, model_input_name, past_model_kwargs,) = unwrap_model(
                self._policy_model
            )._prepare_model_inputs(
                obs["prompt_or_input_encoded_pt"].int(), None, past_model_kwargs
            )

            # 2. prepare encoder outputs
            past_model_kwargs = unwrap_model(
                self._policy_model
            )._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, past_model_kwargs, model_input_name
            )

            # 3. Prepare input_ids for auto-regressive generation
            input_ids = obs["context_encoded_pt"].int()
            decoder_attn_mask = obs["context_attention_mask_pt"]
        else:
            input_ids = obs["context_encoded_pt"].int()
            decoder_attn_mask = past_model_kwargs.pop("decoder_attention_mask")

        # all set to get into auto-regressive mode
        # prepare all of the model inputs for the decoder
        batch_size = input_ids.shape[0]
        model_inputs = unwrap_model(self._policy_model).prepare_inputs_for_generation(
            input_ids, **past_model_kwargs
        )
        # and forward pass to get next token logits
        outputs = self._policy_model(
            **model_inputs, decoder_attention_mask=decoder_attn_mask, return_dict=True
        )
        next_token_logits = outputs.logits[:, -1, :]

        if action_masks is None:
            action_masks = self._get_action_masks(model_inputs, decoder_attn_mask)

        # get log probs
        dist = self._action_dist.proba_distribution(action_logits=next_token_logits)
        raw_log_probs = dist.log_prob(actions)
        if action_masks is not None:
            dist.apply_masking(action_masks)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # update the model kwargs for further generation
        past_model_kwargs = unwrap_model(
            self._policy_model
        )._update_model_kwargs_for_generation(
            outputs,
            past_model_kwargs,
            is_encoder_decoder=unwrap_model(
                self._policy_model
            ).config.is_encoder_decoder,
        )
        past_model_kwargs["decoder_attention_mask"] = torch.cat(
            (decoder_attn_mask, torch.ones(batch_size, 1).to(decoder_attn_mask.device)),
            dim=-1,
        )

        policy_output = PolicyOutput(
            actions, raw_log_probs, log_probs, entropy, past_model_kwargs
        )

        return policy_output

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor, actions_idx: torch.Tensor, action_masks: torch.Tensor
    ) -> EvaluateActionsOutput:

        policy_outputs = self.forward_policy(
            obs=obs, actions=actions, action_masks=action_masks
        )
        value_outputs = self.forward_value(obs)

        eval_outputs = EvaluateActionsOutput(
            values=value_outputs.values,
            log_prob=policy_outputs.log_probs,
            entropy=policy_outputs.entropy,
        )
        return eval_outputs

    def _get_action_masks(self, model_inputs, decoder_attn_mask) -> torch.tensor:
        action_masks = torch.zeros((decoder_attn_mask.size(0), self.action_space.n)).to(
            self.device
        )
        outputs = self._mask_model(
            **model_inputs, decoder_attention_mask=decoder_attn_mask, return_dict=True
        )
        next_token_logits = outputs.logits[:, -1, :]
        ref_distr = self._action_dist.proba_distribution(
            action_logits=next_token_logits
        )
        next_token_probs = ref_distr.distribution.probs
        _, topk_indices = torch.topk(
            next_token_probs, k=self.top_mask, dim=1, sorted=True
        )
        action_masks = action_masks.scatter(index=topk_indices.long(), dim=1, value=1)

        if self.all_special_ids is not None:
            action_masks = action_masks.scatter(
                index=self.all_special_ids, dim=1, value=1
            )
        action_masks = action_masks.bool()

        return action_masks

    def generate(
        self,
        tokenizer: AutoTokenizer,
        texts: List[str] = None,
        max_prompt_length: int = None,
        input_ids: torch.tensor = None,
        attention_mask: torch.tensor = None,
        gen_kwargs: Dict[str, Any] = None,
    ):

        # if it different from rollout gen kwargs
        if gen_kwargs is None:
            gen_kwargs = self._generation_kwargs

        # switch to eval
        self._policy_model.eval()
        self.logits_processor.reset()

        if (
            input_ids is None
            and attention_mask is None
            and texts is not None
            and max_prompt_length is not None
        ):
            prev_truncation_side = tokenizer.truncation_side
            tokenizer.truncation_side = self._prompt_truncation_side
            encodings = tokenizer(
                texts,
                padding="max_length",
                max_length=max_prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
                truncation=True,
            )
            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask
            tokenizer.truncation_side = prev_truncation_side

        self.logits_processor.attention_mask = attention_mask.to(
            self.get_policy_first_device()
        )
        self.logits_processor.all_special_ids = self.all_special_ids = (
            torch.tensor(
                tokenizer.all_special_ids,
                dtype=input_ids.dtype,
                device=self.get_policy_first_device(),
            )
            .unsqueeze(0)
            .expand((input_ids.size(0), -1))
        )

        # if min_length argument is set and if policy is not a seq2seq LM (ie. causal LM)
        # then it has to be adjusted to input_size + min_length
        if (
            "min_length" in gen_kwargs.keys()
            and not unwrap_model(self._policy_model).config.is_encoder_decoder
        ):
            generation_kwargs_ = deepcopy(gen_kwargs)
            generation_kwargs_["min_length"] = (
                input_ids.shape[1] + gen_kwargs["min_length"]
            )
        else:
            generation_kwargs_ = gen_kwargs

        # generate
        gen_output = unwrap_model(self._policy_model).generate(
            inputs=input_ids.to(self.get_policy_first_device()),
            attention_mask=attention_mask.to(self.get_policy_first_device()),
            return_dict_in_generate=True,
            output_scores=True,
            logits_processor=[self.logits_processor],
            **generation_kwargs_
        )

        # number of tokens generated
        seq_length = len(gen_output["scores"])

        # get only the generated text (excluding prompt)
        gen_tokens = gen_output["sequences"][:, -seq_length:]

        # to texts
        gen_texts = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in gen_tokens.tolist()
        ]

        # extract scores (logits)
        step_wise_logprobs = []
        step_wise_actions = []
        action_masks = []
        for step, logits in enumerate(gen_output["scores"]):
            raw_logits, processed_logits = logits
            actions_at_step = gen_tokens[:, step]
            distribution = Categorical(logits=raw_logits)
            log_probs = distribution.log_prob(actions_at_step)
            step_wise_logprobs.append(log_probs)
            step_wise_actions.append(actions_at_step)

            # TBD: workaround due to beam search not returning processed logits yet
            if processed_logits is not None:
                # recalculating action masks
                action_mask = ~torch.isneginf(processed_logits)
                # assert torch.sum(~action_mask.long()).item() != 0
                # assert torch.all(torch.isfinite(Categorical(logits=processed_logits).log_prob(actions_at_step)))
                action_masks.append(action_mask)

        gen_output = GenerationOutputs(
            step_wise_logprobs, step_wise_actions, gen_tokens, gen_texts, action_masks
        )
        return gen_output

    def update_mask_model(self):
        self._mask_model = deepcopy(self._policy_model).eval()
