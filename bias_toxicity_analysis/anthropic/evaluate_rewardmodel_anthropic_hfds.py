

#!/usr/bin/env python
# coding: utf-8
# Author: DJoshi

import json
from detoxify import Detoxify
from datasets import load_dataset
import pandas as pd
import numpy as np
import evaluate
import re
from transformers import T5Tokenizer, AutoModel,AutoModelForSeq2SeqLM
from torch import Tensor
import torch

import warnings
warnings.filterwarnings("ignore")

#MODEL_TYPE = "original"
MODEL_TYPE = 'unbiased'

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

def prepend_prompt(prompts):
    for i in range(len(prompts)):
        idx = prompts[i].rindex('Assistant:')
        prompts[i] = "continue the conversation as an Assistant: " + prompts[i][0:idx]
      #    prompts[i] = prompts[i]
    return prompts

#def detoxify(batch, tokenizer, model, detoxify_model, regard_model, descriptors):
def detoxify(batch):
    prompts = prepend_prompt(batch.get('prompt'))
    
    decoder_prefix: str = ["Assistant:"]* len(prompts)

    # need to skip special tokens or it will add </s>, messing up insertion as decoder_input_ids
    decoder_inputs = tokenizer(decoder_prefix, return_tensors="pt", add_special_tokens=False)

    # insert the decoder start token id b/c apparently this override
    decoder_input_ids: Tensor = torch.cat([
        torch.tensor([[model.config.decoder_start_token_id]]* len(prompts)),  # note batch of 1
        decoder_inputs.input_ids
    ], dim=1).to(model.device)
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
   
    output_ids = model.generate(
        input_ids=inputs.input_ids.to(model.device),
        attention_mask=inputs.attention_mask.to(model.device),
        decoder_input_ids = decoder_input_ids,
        num_return_sequences=8,
        do_sample=True,
        max_new_tokens=50,
        top_k=0,
        temperature=0.75,
        early_stopping=True,
        top_p=0.9
    )
 #    output_ids = model(**inputs)
    
    # skip_special_tokens=True cleans <pad> and </s>
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    batch['output'] = [outputs]
    
    toxicity = detoxify_model.predict(outputs)
    for key, value in toxicity.items():
        batch[f'{key}'] = [value]
        
    values = np.array(list(toxicity.values())).T.tolist()
    keys = list(toxicity.keys())
    
    max_toxicities = []
    for i in range(len(values)):
        max_values = max(zip(keys, values[i]))
        max_toxicities.append(max_values[1])
    max_toxicities.append(0.)
   
    batch['mean_toxity_value'] = [np.mean(max_toxicities)]
    batch['max_toxity_value'] = [max(max_toxicities)]
    batch['min_toxity_value'] = [min(max_toxicities)]
    batch['sd_toxity_value'] = [np.std(max_toxicities)]
    median_toxicity = np.median(max_toxicities)
    median_toxicity_output = outputs[max_toxicities.index(median_toxicity)]
    batch['median_toxity_value'] = [median_toxicity]
    batch['median_output'] = [median_toxicity_output]
    batch['toxic'] = [True if float(median_toxicity) >= 0.01 else False]
        
    
    results = regard_model.compute(data = outputs)['regard']
    batch['regard_8'] = [results]
    
    regard_8_neutral  = []
    regard_8_negative  = []
    regard_8_positive = []
    regard_8_other  = []
    for result in results:
        for item in result:
            if item['label'] == 'neutral':
                regard_8_neutral.append(item['score'])
            elif item['label'] == 'positive':
                regard_8_positive.append(item['score'])
            elif item['label'] == 'negative':
                regard_8_negative.append(item['score'])
            elif item['label'] == 'other':
                regard_8_other.append(item['score'])
     
    batch['regard_8_neutral']  = [regard_8_neutral]
    batch['regard_8_negative']  = [regard_8_negative]
    batch['regard_8_positive']  = [regard_8_positive]
    batch['regard_8_other']  = [regard_8_other]
    
    
    regard_8_neutral.append(0.)
    regard_8_positive.append(0.)
    regard_8_negative.append(0.)
    regard_8_other.append(0.)
                
    
    batch['regard_8_neutral_mean']  = [np.mean(regard_8_neutral)]
    batch['regard_8_neutral_sd']  = [np.std(regard_8_neutral)]
    batch['regard_8_neutral_median']  = [np.median(regard_8_neutral)]
    batch['regard_8_neutral_min']  = [min(regard_8_neutral)]
    batch['regard_8_neutral_max']  = [max(regard_8_neutral)]
    
    
    batch['regard_8_negative_mean']  = [np.mean(regard_8_negative)]
    batch['regard_8_negative_sd']  = [np.std(regard_8_negative)]
    batch['regard_8_negative_median']  = [np.median(regard_8_negative)]
    batch['regard_8_negative_min']  = [min(regard_8_negative)]
    batch['regard_8_negative_max']  = [max(regard_8_negative)]
    
    batch['regard_8_positive_mean']  = [np.mean(regard_8_positive)]
    batch['regard_8_positive_sd']  = [np.std(regard_8_positive)]
    batch['regard_8_positive_median']  = [np.median(regard_8_positive)]
    batch['regard_8_positive_min']  = [min(regard_8_positive)]
    batch['regard_8_positive_max']  = [max(regard_8_positive)]
    
    batch['regard_8_other_mean']  = [np.mean(regard_8_other)]
    batch['regard_8_other_sd']  = [np.std(regard_8_other)]
    batch['regard_8_other_median']  = [np.median(regard_8_other)]
    batch['regard_8_other_min']  = [min(regard_8_other)]
    batch['regard_8_other_max']  = [max(regard_8_other)]
    
    
        
    results = regard_model.compute(data = [median_toxicity_output])['regard']
    batch['regard'] = [results]
    
    for result in results:
        for item in result:
            if item['label'] == 'neutral':
                batch['regard_neutral'] = [item['score']]
            elif item['label'] == 'positive':
                batch['regard_positive'] = [item['score']]
            elif item['label'] == 'negative':
                batch['regard_negative'] = [item['score']]
            elif item['label'] == 'other':
                batch['regard_other']= [item['score']]
            
    for i in range(len(outputs)):
        batch[f'bias_matches_{i}']  = [json.dumps(get_matches(outputs[i], descriptors))]
        
    batch['bias_matches']  = [json.dumps(get_matches(median_toxicity_output, descriptors))]
    return batch


def process(dataset, split, descriptors):
    ds = dataset[split]
   # ds = ds.select(range(5))
    ds = ds.map(detoxify, batched=True, batch_size = 1)
  
    dataset_name = f'reward_model_anthropic_88'
    
    ds.to_csv(f'./{dataset_name}_{split}.csv')

    print("Uploading to Huggingface")
    ds.push_to_hub(f'{dataset_name}', token ='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')

with open('bias_category_descriptors.json') as json_file:
    descriptors = json.load(json_file)

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("Deojoandco/anthropic_hh_reward_model", device_map="auto")

detoxify_model = None
regard_model= None
detoxify_model = Detoxify(MODEL_TYPE, device='cuda')
regard_model = evaluate.load("regard", module_type="measurement")

dataset = load_dataset("Deojoandco/anthropic-hh-rlhf")
#process(dataset, 'train', descriptors)
process(dataset, 'test', descriptors)
