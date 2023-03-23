### Bias and Toxicity Analysis of allenai/real-toxicity-prompts Dataset
Goal is to analyze bias and toxicity of allenai/real-toxicity-prompts dataset. We will be using [Unitary Detoxify](https://github.com/unitaryai/detoxify) library for evaluating toxicity of the continuation text field from the dataset. We have defined a dictionary (bias_category_descriptors.json) for classifying bias categories of the continuation text field from the dataset



### Dependencies
1. Python 3.9.16
2. Huggingface transformers and datasets
3. Unitary Detoxify - pip install detoxify



### Execution
Execute python files in following order:
1. Execute 'detoxify_realtoxicity_hfds.py' in terminal console (python detoxify_realtoxicity_hfds.py). This will use Unitary Detoxify to evaluate continuation text's toxicity, create train and test csv files and upload the resultant dataset on to [Huggingface repository (Deojoandco/detoxify_unbiased_real_toxicity)]([https://huggingface.co/datasets/Deojoandco/detoxify_unbiased_hhrlhf_last_assistant](https://huggingface.co/datasets/Deojoandco/detoxify_unbiased_real_toxicity)). By default, we will use the 'unbiased' Unitary Detoxify model for evaluating toxicity and can be controlled by setting the flag MODEL_TYPE to 'original' or 'unbiased'.
2. Execute 'bias_classification_realtoxicity.py' in terminal console (python bias_classification_realtoxicity.py). This will download 'Deojoandco/detoxify_unbiased_real_toxicity' dataset (created in step 1) from Huggingface and classify continuation text with bias categories defined in the 'bias_category_descriptors.json' dictionary file. It will generate real_toxcity_biasmatch.csv output file having bias classification details.
3. Execute 'bias_toxicity_realtoxicity_race_plot.py' in terminal console (bias_toxicity_realtoxicity_race_plot.py). This will load the csv file created in step 2, calculate regard toxicity for race categories, plot sub-category race distribution for continuation Text and plot race-regard toxicity plot.
4. Execute 'bias_toxicity_realtoxicity_all_plot.py' in terminal console (python bias_toxicity_realtoxicity_all_plot.py). This will load the csv file created in step 2, plot sub-category and category distribution for continuation Text.



### Race Plots
![real_toxic_bias_distribution_subcategory](https://user-images.githubusercontent.com/50883840/227076247-b5e5b3f4-8267-48f0-b716-39d97aa72d02.jpg)
![image](https://user-images.githubusercontent.com/50883840/226848483-fefa51cf-7032-48ef-9bfe-5da3f080452d.png)

### All Categories and Subcategories plots
![real_toxic_bias_distribution_categories](https://user-images.githubusercontent.com/50883840/227076310-2526c883-d492-4af3-9873-b8c704ed78e6.jpg)
![real_toxic_bias_distribution_subcategories](https://user-images.githubusercontent.com/50883840/227076349-a6241a98-614a-4a72-b775-8d8e26297d7a.jpg)

