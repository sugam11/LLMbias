### Bias and Toxicity Analysis of Anthropic/hh-rlhf Dataset

Goal is to analyze bias and toxicity of Anthropic/hh-rlhf dataset. We will be using [Unitary Detoxify](https://github.com/unitaryai/detoxify) library for evaluating toxicity of the assistant text field from the Anthropic dataset. We have defined a dictionary (bias_category_descriptors.json) for classifying bias categories of the assistant text field from the Anthropic dataset

#### Dependencies
1. Python 3.9.16
2. Huggingface transformers and datasets
3. Unitary Detoxify - pip install detoxify

#### Execution
Execute python files in following order:
1. Execute 'detoxify_anthropic_hfds.py' in terminal console (python detoxify_anthropic_hfds.py). This will use Unitary Detoxify to evaluate assistant text's toxicity, create train and test csv files and upload the resultant dataset on to [Huggingface repository (Deojoandco/detoxify_unbiased_hhrlhf_last_assistant)](https://huggingface.co/datasets/Deojoandco/detoxify_unbiased_hhrlhf_last_assistant). The code will evaluate either the full assistant text or the last assistant's text by configuring the flag LAST_ASSISTANT = True/False in the code file (default is True). By default, we will use the unbiased Unitary Detoxify model for evaluating toxicity and can be controlled by setting the flag MODEL_TYPE to 'original' or 'unbiased'.
2. Execute 'bias_classification_anthropic.py' in terminal console (python bias_classification_anthropic.py). This will download 'Deojoandco/detoxify_unbiased_hhrlhf_last_assistant' dataset (created in step 1) from Huggingface and classify last assistant text with bias categories defined in the 'bias_category_descriptors.json' dictionary file. It will generate ddetoxify_unbiased_hhrlhf_last_assistant_biasmatch_train.csv and ddetoxify_unbiased_hhrlhf_last_assistant_biasmatch_test.csv output files having bias classification details.
