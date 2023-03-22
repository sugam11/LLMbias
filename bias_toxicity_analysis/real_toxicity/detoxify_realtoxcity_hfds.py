#!/usr/bin/env python
# coding: utf-8
# Author: DJoshi

from detoxify import Detoxify
from datasets import load_dataset
import pandas as pd
import numpy as np

#MODEL_TYPE = "original"
MODEL_TYPE = 'unbiased'

def get_texts(continuations):
    texts = []
    for cont in continuations:
        texts.append(cont['text'])
    return texts

def detoxify(batch, detoxify_model):
    texts = get_texts(batch.get('continuation'))
   
    chosen_toxicity = detoxify_model.predict(texts)
    for key, value in chosen_toxicity.items():
        batch[f'cont_{key}'] = value
        
    values = np.array(list(chosen_toxicity.values())).T.tolist()
    keys = list(chosen_toxicity.keys())
    batch['cont_max_toxity_key'] = []
    batch['cont_max_toxity_value'] = []
    batch['cont_toxic'] = []
    for i in range(len(values)):
        max_values = max(zip(keys, values[i]))
        batch['cont_max_toxity_key'].append(max_values[0])
        batch['cont_max_toxity_value'].append(max_values[1])
        batch['cont_toxic'].append(True if float(max_values[1]) >= 0.01 else False)
    return batch

detoxify_model = Detoxify(MODEL_TYPE, device='cuda')
dataset = load_dataset("allenai/real-toxicity-prompts")
ds_train = dataset['train']
ds = ds_train.map(detoxify, fn_kwargs={'detoxify_model': detoxify_model}, batched=True, batch_size = 50)

ds.to_csv(f'./detoxify_{MODEL_TYPE}_real_toxicity.csv')

print("Uploading to Huggingface")
ds.push_to_hub(f'detoxify_{MODEL_TYPE}_real_toxicity', token ='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')