### Reinforced Learning with Human Feedback using CarperAI/trlx library

### Dependencies
1. Python 3.9.16
2. CarperAI/trlx library
3. Huggingface transformers and datasets

### Insallation
1. Install CarperAI/trlx libary by following installation instructions mentioned herein: https://github.com/CarperAI/trlx
2. Execute ppo_anthropic.py for RLHF training of Anthropic/hh-rlhf using google/flan-t5-base model. 
3. Execute ppo_realtoxicity.py for RLHF training of allenai/real-toxicity-prompts using google/flan-t5-base model. 

### Configuration
During first execution, the code will prompt to enter wandb token for logging to Weights & Biases. Following configuration applies only to ppo_anthropic.py file:
1. REWARD_MODE = "REWARD_MODEL" or "REWARD_TOXICITY". The code can be executed using either a reward model or unitary toxic function to evaluate flan-t5 model generated outputs for toxicity
2. REWARD_MODEL_NAME and REWARD_TOKENIZER_NAME. This is applicable to only ppo_anthropic.py file. Either use OpenAssistant/reward-model-deberta-v3-large-v2 or sugam11/gpt2-rlhf-reward reward model for RLHF training

### Reports
Reports and evaluation output are reported to Weights & biasess. Some example runs:
1. https://wandb.ai/devavratj/trlx_anthropic_2500 - RLHF training of Anthropic/hh-rlhf using google/flan-t5-base model. TrainSize = 2500, TestSize=500
2. https://wandb.ai/devavratj/trlx_realtoxicity_2500 - RLHF training of allenai/real-toxicity-prompts using google/flan-t5-base model. TrainSize = 2500, TestSize=500
