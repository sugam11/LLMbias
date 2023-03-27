# LLMbias

This repo contains for our experiments to evaluate toxicity in LLM and Bias Datasets.

What the notebooks contain and mean:

1. Reward.ipynb - Training a reward model on anthropic HH data. 

2. Finetuning Flan-t5.ipynb - Finetuning Flan-t5 base model on Antrhopic HH data, evaluate toxicity on real toxicity prompts data

3. OpenAI Bias Eval.ipynb - Notebook for evaluating bias in open AI models

For dataset bias and Toxicity on Anthropic and REALTOXICITYPROMPTS, please refer to the README inside the bias_toxicity_analysis folder. 

For the final RLHF PPO trained model with a reward model, please refer to the README in the trlx folder.
