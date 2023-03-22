
#!/usr/bin/env python
# coding: utf-8
# Author: DJoshi

import json
import pandas as  pd
import re
from datasets import load_dataset
from numba import jit, cuda

import warnings
warnings.filterwarnings("ignore")

def match_category(text, category, category_variants):
    matches = []
    text = text.lower()
    for variant in category_variants:
        match = re.search(f"\\b{variant.lower()}\\b", text)
        if match:
            matches.append(variant)
            break
        
    return matches

def get_matches_rec(text, parent_categories, category, descriptors, matches):
    if len(parent_categories) > 0:
        parent_categories += ' -> '
    parent_categories = parent_categories + category
    for key, value in descriptors.items():
        if type(value) is dict:
            get_matches_rec(text, parent_categories, key, value, matches)
        else:
            match = match_category(text, key, value)
            if len(match) > 0:
                matches.append({'parent_categories': parent_categories, 'category': key, 'matches': match})
        
def get_matches(text, descriptors):
    matches = []
    get_matches_rec(text, "", "", descriptors, matches)
    return matches

def compute_bias_bucket(batch, descriptors):
    choseen_assistant_texts = batch.get('chosen_assistant')
    batch['chosen_bias_matches'] = []
    for choseen_assistant_text in choseen_assistant_texts:
        batch['chosen_bias_matches'].append(get_matches(choseen_assistant_text, descriptors))
        
    rejected_assistant_texts = batch.get('rejected_assistant')
    batch['rejected_bias_matches'] = []
    for rejected_assistant_text in rejected_assistant_texts:
        batch['rejected_bias_matches'].append(get_matches(rejected_assistant_text, descriptors))
        
    return batch

@jit(target_backend='cuda')
def process(descriptors, dataset, split):
    ds_split = dataset[split]
    ds = ds_split.map(compute_bias_bucket, fn_kwargs={'descriptors': descriptors}, batched=True, batch_size = 50)
    
    ds.to_csv(f'./ddetoxify_unbiased_hhrlhf_last_assistant_biasmatch_{split}.csv')

with open('./../bias_category_descriptors.json') as json_file:
    descriptors = json.load(json_file)
    
dataset = load_dataset("Deojoandco/detoxify_unbiased_hhrlhf_last_assistant")
process(descriptors, dataset, 'train')
process(descriptors, dataset, 'test')
        