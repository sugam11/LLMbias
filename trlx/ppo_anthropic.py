import os
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from detoxify import Detoxify
import evaluate
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPT2ForSequenceClassification
import torch
import random

#REWARD_MODE = "REWARD_TOXICITY"
REWARD_MODE = "REWARD_MODEL"

#REWARD_TOKENIZER_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"
#REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"

REWARD_TOKENIZER_NAME = "microsoft/DialogRPT-updown"
REWARD_MODEL_NAME = "sugam11/gpt2-rlhf-reward"

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
MAX_OUTPUT_LEN = 50

BATCH_SIZE = 8


lm_model_name = "google/flan-t5-base"

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device

device = get_device()


tokenizer = T5Tokenizer.from_pretrained(lm_model_name)
model = T5ForConditionalGeneration.from_pretrained(lm_model_name).to(device)
    
decoder_prefix = ["Assistant:"]*BATCH_SIZE

# need to skip special tokens or it will add </s>, messing up insertion as decoder_input_ids
decoder_inputs = tokenizer(decoder_prefix, return_tensors="pt", add_special_tokens=False)

# insert the decoder start token id b/c apparently this override
decoder_input_ids = torch.cat([
    torch.tensor([[model.config.decoder_start_token_id]] * BATCH_SIZE), 
    decoder_inputs.input_ids
], dim=1).to(device)

del tokenizer
del model
torch.cuda.empty_cache()

import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig

detoxify_model = Detoxify('unbiased', device='cuda')

config = TRLConfig(
    train=TrainConfig(
        seq_length=612,
        epochs=100,
        total_steps=100000,
        batch_size=BATCH_SIZE,
        checkpoint_interval=10000,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
    ),
    model=ModelConfig(
        model_path=lm_model_name,
        model_arch_type="seq2seq",
        num_layers_unfrozen=2,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path=lm_model_name,
        truncation_side="right",
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 1.0e-5,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 1.0e-6,
        },
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={
            "T_max": 10000,
            "eta_min": 1.0e-6,
        },
    ),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=512,
        chunk_size=BATCH_SIZE,
        ppo_epochs=4,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=0.99,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1.0,
        scale_reward=None,
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs={
            "max_new_tokens": MAX_OUTPUT_LEN,
            'do_sample': True,
            'top_k': 0,
            'temperature': 0.75,
            'early_stopping': True,
            "top_p": 0.95,
            "num_return_sequences": 1,
            "decoder_input_ids": decoder_input_ids,
        },
    ),
)

def rewards_toxicity_fn(samples: List[str], **kwargs) -> Dict[str, List[float]]:
    
    
    results = detoxify_model.predict(kwargs['outputs'])
    results = np.transpose(list(results.values())).tolist()
    rewards  = []
    for i in range(len(results)):
        max_toxicity = max(results[i])
        #toxicity_reward = 100 - (100 * max_toxicity)
        rewards.append(-1 * max_toxicity)
        
    return rewards


def reward_model_fn(samples: List[str], prompts: List[str], outputs: List[str]):
    rewards = []
   
    # disabled batch processing as we are getting outof memory errors
    # inputs = reward_tokenizer(prompts, outputs, return_tensors="pt", padding=True).to(device)
    #rewards = reward_model(**inputs).logits.cpu().detach().tolist()
    
    for i in range(len(prompts)):
        inputs = reward_tokenizer(prompts[i], outputs[i], return_tensors="pt").to(device)
        logits = reward_model(**inputs).logits
        rewards.append(logits.item())
    
    
    return rewards

def prepend_prompt(prompts):
    for i in range(len(prompts)):
        idx = prompts[i].rindex('Assistant:')
        prompts[i] = "continue the conversation as an Assistant: " + prompts[i][0:idx]
      #    prompts[i] = prompts[i]
    return prompts



dataset = load_dataset("Deojoandco/anthropic-hh-rlhf")
print(dataset)

train_size = int(len(dataset['train']) / BATCH_SIZE) * BATCH_SIZE
prompts = prepend_prompt(dataset['train']['prompt'][0:train_size])

val_size  = int(len(dataset['test']) / BATCH_SIZE) * BATCH_SIZE
eval_prompts = prepend_prompt(dataset['test']['prompt'][0:val_size])

if REWARD_MODE == "REWARD_MODEL":
    reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME).to(device)
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_TOKENIZER_NAME)

trainer = trlx.train(
    prompts = prompts,
    eval_prompts = eval_prompts,
    reward_fn = reward_model_fn if REWARD_MODE == "REWARD_MODEL" else rewards_toxicity_fn,
    config = config,
    #stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    #stop_sequences=[ "Assistant:", "assistant:"],
)

trainer.model.push_to_hub('anthropic_hh', use_auth_token='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')


