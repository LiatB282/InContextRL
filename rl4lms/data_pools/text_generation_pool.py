import random
from abc import abstractclassmethod
from dataclasses import dataclass
from typing import Any, List, Dict
from datasets import load_dataset


@dataclass(init=True)
class Sample:
    id: str
    prompt_or_input_text: str
    references: List[str]
    meta_data: Dict[str, Any] = None


class TextGenPool:
    def __init__(self, samples: List[Sample]):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, ix: int) -> Sample:
        if ix >= len(self):
            raise StopIteration
        sample = self._samples[ix]
        return sample, 1.0

    def sample(self) -> Sample:
        random_sample = random.choice(self._samples)
        return random_sample

    @abstractclassmethod
    def prepare(cls, **args) -> 'TextGenPool':
        """
        A factory method to instantiate data pool
        """
        raise NotImplementedError

    def split(self, split_ratios: List[float]) -> List['TextGenPool']:
        start_ix = 0
        pools = []
        for ratio in split_ratios:
            count = int(len(self) * ratio)
            end_ix = start_ix + count
            pools.append(type(self)(self._samples[start_ix: end_ix]))
            start_ix = end_ix
        return pools


class TriviaQAPool(TextGenPool):
    @classmethod
    def prepare(cls, split: str, encoded_dataset_path:str=None, is_debug:bool=False, cache_path:str=None):
        hf_split = split if split in ['train', 'test'] else 'validation'
        dataset = load_dataset('trivia_qa', 'rc', split=hf_split, cache_dir=cache_path)
        if is_debug and split == "train":
            dataset = [dataset[i] for i in range(1000)]
        samples = []
        for ix, item in enumerate(dataset):
            sample = Sample(id=f"{split}_{ix+1}",
                            prompt_or_input_text=f"Question: {item['question']} Answer:",
                            references=list(set([item['answer']['value']] + item['answer']['aliases']))
                            )
            samples.append(sample)
        pool_instance = cls(samples)
        return pool_instance
