
#!/usr/bin/env python
# coding: utf-8
# Author: DJoshi
# In[1]:


from detoxify import Detoxify
from datasets import load_dataset
import pandas as pd
import numpy as np

LAST_ASSISTANT = True
#MODEL_TYPE = "original"
MODEL_TYPE = 'unbiased

def get_assistant_text(text):
    splits = text.split('\n')
    
    assistants = []
    assistant_text = ''
    assistant = False
    for split in splits:
        split = split.strip()
        if len(split) > 0:
            if split.startswith('Assistant:'):
                assistant = True
            elif split.startswith('Human:'):
                
                assistant = False
                if len(assistant_text) > 0:
                    assistants.append(assistant_text)
                assistant_text = ''
            
            if assistant == True:
                if len(assistant_text) > 0:
                    assistant_text += '\n'
                
                assistant_text += split
    
    if len(assistant_text) > 0:
        assistants.append(assistant_text)
        
    return assistants

def get_assistant_texts(texts):
    assistant = []
    for txt in texts:
        assistant_texts = get_assistant_text(txt)
        if LAST_ASSISTANT == True:
            assistant_texts = [assistant_texts[-1]]
        assistant.append("\n".join(assistant_texts))
    return assistant

def detoxify(batch, detoxify_model):
    items = batch.items()
    chosen = get_assistant_texts(batch.get('chosen'))
    rejected = get_assistant_texts(batch.get('rejected'))
    
    batch['chosen_assistant'] = chosen
    batch['rejected_assistant'] = rejected
    
    chosen_toxicity = detoxify_model.predict(chosen)
    for key, value in chosen_toxicity.items():
        batch[f'chosen_{key}'] = value
        
    values = np.array(list(chosen_toxicity.values())).T.tolist()
    keys = list(chosen_toxicity.keys())
    batch['chosen_max_toxity_key'] = []
    batch['chosen_max_toxity_value'] = []
    batch['chosen_toxic'] = []
    for i in range(len(values)):
        max_values = max(zip(keys, values[i]))
        batch['chosen_max_toxity_key'].append(max_values[0])
        batch['chosen_max_toxity_value'].append(max_values[1])
        batch['chosen_toxic'].append(True if float(max_values[1]) >= 0.01 else False)
        
    rejected_toxicity = detoxify_model.predict(rejected)
    for key, value in rejected_toxicity.items():
        batch[f'rejected_{key}'] = value
        
    values = np.array(list(rejected_toxicity.values())).T.tolist()
    keys = list(rejected_toxicity.keys())
    batch['rejected_max_toxity_key'] = []
    batch['rejected_max_toxity_value'] = []
    batch['rejected_toxic'] = []
    for i in range(len(values)):
        max_values = max(zip(keys, values[i]))
        batch['rejected_max_toxity_key'].append(max_values[0])
        batch['rejected_max_toxity_value'].append(max_values[1])
        batch['rejected_toxic'].append(True if float(max_values[1]) >= 0.01 else False)
        
    return batch


def process(dataset, split):
    ds = dataset[split]
    ds = dataset.map(detoxify, fn_kwargs={'detoxify_model': detoxify_model}, batched=True, batch_size = 50)

    dataset_name = f'detoxify_{model_type}_hhrlhf_'
    if LAST_ASSISTANT == True:
        dataset_name += f'last_assistant'
    else:
        dataset_name += f'assistant'


    ds.to_csv(f'./{dataset_name}_{split}.csv')


    print("Uploading to Huggingface")
    ds.push_to_hub(f'dataset_name', token ='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')


detoxify_model = Detoxify(model_type, device='cuda')
dataset = load_dataset("Anthropic/hh-rlhf")
process(dataset, 'train')
process(dataset, 'test')





# In[ ]:




