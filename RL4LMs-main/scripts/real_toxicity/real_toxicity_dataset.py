#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 11:01:27 2023

@author: deoj
"""

from datasets import load_dataset
from rl4lms.data_pools.text_generation_pool import Sample, TextGenPool
from rl4lms.envs.text_generation.registry import DataPoolRegistry

class RealToxicityPromptsDataPool(TextGenPool):
    @classmethod
    def prepare(cls, split: str, prefix: str = "complete sentence: "):
        
        dataset = load_dataset('allenai/real-toxicity-prompts')
        
        ds = dataset['train'].train_test_split(train_size = 0.70, shuffle = False, seed = 42)
        dataset_split = None
        if split == "train":
            dataset_split = ds['train']
        else:
            ds = ds['test'].train_test_split(test_size = 0.5, shuffle = False, seed = 42)
            if split == "val":
                dataset_split = ds['train']
            elif split == "test":
                dataset_split = ds['test']
            else:
                raise NotImplementedError
        
        samples = []
        for ix, item in enumerate(dataset_split):
            prompt = prefix + item["prompt"]['text']
            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=prompt,
                            references=[item["continuation"]['text']],
                            meta_data={
                                 "raw_table": item
                             }
                     )
            samples.append(sample)
        
        pool_instance = cls(samples)
        return pool_instance

def register():
    DataPoolRegistry.add('real-toxicity-prompts', RealToxicityPromptsDataPool)    
