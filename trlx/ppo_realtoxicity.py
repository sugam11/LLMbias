import os
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from detoxify import Detoxify
import evaluate
from transformers import AutoTokenizer

MAX_OUTPUT_LEN = 50

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
        batch_size=10,
        checkpoint_interval=10000,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
    ),
    model=ModelConfig(
        model_path="google/flan-t5-base",
        model_arch_type="seq2seq",
        num_layers_unfrozen=2,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path="google/flan-t5-base",
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
        chunk_size=1,
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
            "num_return_sequences": 1
        }
    ),
)


def main(hparams={}):
    
    def reward_fn_meteor(samples: List[str], prompts: List[str], outputs: List[str]):
        original_summaries = [prompt_label[prompt.strip()] for prompt in prompts]
        scores = [
            meteor.compute(predictions=[output.strip()], references=[original])["meteor"]
            for (original, output) in zip(original_summaries, outputs)
        ]
        return scores
        

    def rewards_toxicity_fn(samples: List[str], **kwargs) -> Dict[str, List[float]]:
        results = detoxify_model.predict(kwargs['outputs'])
        results = np.transpose(list(results.values())).tolist()
        rewards  = []
        for i in range(len(results)):
            max_toxicity = max(results[i])
            #toxicity_reward = 100 - (100 * max_toxicity)
            rewards.append(-1 * max_toxicity)
            '''
            prompt = samples[i]
            output = kwargs['outputs'][i].split()
            len_percent = (len(output)/MAX_OUTPUT_LEN) * 100
            if len_percent >= 10:
                max_toxicity = max(results[i])
                toxicity_reward = 100 - (100 * max_toxicity)
                len_reward = 100 if len_percent <=50 else 70
                reward = (toxicity_reward * 0.7) + (len_reward * 0.3)
                rewards.append(reward)
            else:
                rewards.append(-50)
            '''
        return rewards

    def build_dataset():
        ds = load_dataset("allenai/real-toxicity-prompts")

        def curate_dataset(batch):
            prompts_dict = batch.get('prompt')
            prompts = []
            for record in prompts_dict:
                prompts.append("complete the sentence: " + record['text'])
                
            continuations_dict = batch.get('continuation')
            continuations = []
            for record in continuations_dict:
                continuations.append(record['text'])
                
            batch['prompt'] = prompts
            batch['continuation'] = continuations
            return batch
            
        
        train_ds = ds['train'].map(curate_dataset, batched = True, batch_size=50)
        
        dataset = train_ds.train_test_split(test_size=0.2, seed=42)
        dataset = dataset.remove_columns(['filename', 'begin', 'end', 'challenging'])
        return dataset

    dataset = build_dataset()
    meteor = evaluate.load('meteor')
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    tokenizer.sep_token = "<sep>"
    
    prompts = dataset['train']['prompt'][0:80]
    continuations = dataset['train']['continuation'][0:2500]
    eval_prompts = dataset['test']['prompt'][0:20]
    eval_continuations = dataset['test']['continuation'][0:500]
    
    prompt_label = {}
    max_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
    
    for i in range(len(prompts)):
        key = tokenizer.decode(
            tokenizer(prompts[i], truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"],
            skip_special_tokens=True,
        )  # get prompt like trlx's prompt
        prompt_label[key.strip()] = continuations[i]
    
    for i in range(len(eval_prompts)):
        try:
            key = tokenizer.decode(
                tokenizer(eval_prompts[i], truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"],
                skip_special_tokens=True,
            )  # get prompt like trlx's prompt
            prompt_label[key.strip()] = eval_continuations[i]
        except:
            print("error")
    
    trlx.train(
        prompts=prompts,
        eval_prompts=eval_prompts,
        reward_fn=rewards_toxicity_fn,
        config=config,
    )

if __name__ == "__main__":
    main()
