### Bias and Toxicity Analysis of Anthropic/hh-rlhf Dataset
Goal is to analyze bias and toxicity of Anthropic/hh-rlhf dataset. We will be using [Unitary Detoxify](https://github.com/unitaryai/detoxify) library for evaluating toxicity of the assistant text field from the Anthropic dataset. We have defined a dictionary (bias_category_descriptors.json) for classifying bias categories of the assistant text field from the Anthropic dataset



### Dependencies
1. Python 3.9.16
2. Huggingface transformers and datasets
3. Unitary Detoxify - pip install detoxify



### Execution
Execute python files in following order:
1. Execute 'detoxify_anthropic_hfds.py' in terminal console (python detoxify_anthropic_hfds.py). This will use Unitary Detoxify to evaluate assistant text's toxicity, create train and test csv files and upload the resultant dataset on to [Huggingface repository (Deojoandco/detoxify_unbiased_hhrlhf_last_assistant)](https://huggingface.co/datasets/Deojoandco/detoxify_unbiased_hhrlhf_last_assistant). The code will evaluate either the full assistant text or the last assistant's text by configuring the flag LAST_ASSISTANT = True/False in the code file (default is True). By default, we will use the 'unbiased' Unitary Detoxify model for evaluating toxicity and can be controlled by setting the flag MODEL_TYPE to 'original' or 'unbiased'.
2. Execute 'bias_classification_anthropic.py' in terminal console (python bias_classification_anthropic.py). This will download 'Deojoandco/detoxify_unbiased_hhrlhf_last_assistant' dataset (created in step 1) from Huggingface and classify last assistant text with bias categories defined in the 'bias_category_descriptors.json' dictionary file. It will generate ddetoxify_unbiased_hhrlhf_last_assistant_biasmatch_train.csv and ddetoxify_unbiased_hhrlhf_last_assistant_biasmatch_test.csv output files having bias classification details.
3. Execute 'bias_toxicity_anthropic_race_plot.py' in terminal console (bias_toxicity_anthropic_race_plot.py). This will load the train and test csv files created in step 2, calculate regard toxicity for race categories, plot sub-category race distribution for chosen and rejected Assistant Text and plot race-regard toxicity plot.
4. Execute 'bias_toxicity_anthropic_all_plot.py' in terminal console (python bias_toxicity_anthropic_all_plot.py). This will load the train and test csv files created in step 2, plot category and sub-category distribution for chosen and rejected Assistant Text and plot race-regard toxicity plot.
5. Execute 'evaluate_rewardmodel_anthropic_hfds.py' in terminal console (python evaluate_rewardmodel_anthropic_hfds.py). This will use Unitary Detoxify to evaluate PPO trained model's output text's toxicity, bias classification, evaluate regard and create train and test csv files and upload the resultant dataset on to [Huggingface repository (Deojoandco/reward_model_anthropic)](https://huggingface.co/datasets/Deojoandco/reward_model_anthropic). By default, we will use the 'unbiased' Unitary Detoxify model for evaluating toxicity and can be controlled by setting the flag MODEL_TYPE to 'original' or 'unbiased'.
6. Execute 'bias_toxicity_rewardmodel_anthropic_race_plot.py' in terminal console (bias_toxicity_rewardmodel_anthropic_race_plot.py). This will load the train and test csv files created in step 5, plot sub-category race distribution for chosen and rejected Assistant Text and plot race-regard toxicity plot.

### Race Plots
![chosen_response_toxic_bias_distribution_subcategory_train](https://user-images.githubusercontent.com/50883840/226833167-a65692ff-e493-4937-8441-7d5fd6555368.jpg)
![rejected_response_toxic_bias_distribution_subcategory_train](https://user-images.githubusercontent.com/50883840/226833176-f5e7ce3d-0185-4285-94af-024f7ad04fc9.jpg)
![image](https://user-images.githubusercontent.com/50883840/226841525-37e6cd2d-6cea-487a-b3d3-8ac52c8329c2.png)
![image](https://user-images.githubusercontent.com/50883840/226841575-46c4ed48-ce16-43cd-a17c-5337626d957d.png)
![chosen_response_toxic_bias_distribution_subcategory_test](https://user-images.githubusercontent.com/50883840/226833230-6bb6163c-0397-4327-b4e0-0506b0328351.jpg)
![rejected_response_toxic_bias_distribution_subcategory_test](https://user-images.githubusercontent.com/50883840/226833239-26adde9c-3933-4dd5-8399-25dd276c5390.jpg)
![image](https://user-images.githubusercontent.com/50883840/226841627-97a1f4df-33da-4fe0-8cd6-34b7a5eb8789.png)
![image](https://user-images.githubusercontent.com/50883840/226841657-b94abe6e-cea3-4da1-a974-2414d3216198.png)


### All Categories and Subcategories Plots
![chosen_response_toxic_bias_distribution_categories_train](https://user-images.githubusercontent.com/50883840/227074626-b79523e9-04e3-46d8-924b-d9fb9101775d.jpg)
![chosen_response_toxic_bias_distribution_subcategories_train](https://user-images.githubusercontent.com/50883840/227074631-5a6e809b-e73e-4d2a-b064-c8e40ab9c2a9.jpg)
![rejected_response_toxic_bias_distribution_categories_train](https://user-images.githubusercontent.com/50883840/227074669-63ade3a2-f587-4a75-968d-b473f98c6b6f.jpg)
![rejected_response_toxic_bias_distribution_subcategories_train](https://user-images.githubusercontent.com/50883840/227074705-b5a1cf64-dd20-491f-b2b9-e5758e62351e.jpg)
![chosen_response_toxic_bias_distribution_categories_test](https://user-images.githubusercontent.com/50883840/227075195-4456dc63-4a2e-47e2-b24c-a5aab8d70977.jpg)
![chosen_response_toxic_bias_distribution_subcategories_test](https://user-images.githubusercontent.com/50883840/227075226-d9c409f3-5844-406c-8aec-7a58a7b71dad.jpg)
![rejected_response_toxic_bias_distribution_categories_test](https://user-images.githubusercontent.com/50883840/227075283-bc6eac06-9333-4744-b709-61d049cf2bf3.jpg)
![rejected_response_toxic_bias_distribution_subcategories_test](https://user-images.githubusercontent.com/50883840/227075293-5f439e73-720c-4853-befa-f5b45d15e451.jpg)


### Race Plots for outputs generated by PPO Trained model (Deojoandco/anthropic_hh_reward_model)
![chosen_response_toxic_bias_distribution_subcategory_train](https://user-images.githubusercontent.com/50883840/227583888-76c25dfb-cb96-4c5d-9392-2bd3c913cfd8.jpg)
![hhrlhf_chosen_train_regard_race](https://user-images.githubusercontent.com/50883840/227583922-c4d81bb0-56f3-4825-b906-8fcd91775a0b.jpeg)

![chosen_response_toxic_bias_distribution_subcategory_test](https://user-images.githubusercontent.com/50883840/227584032-72a9b19e-937a-4b4c-a94f-7c5d42512e39.jpg)
![hhrlhf_chosen_test_regard_race](https://user-images.githubusercontent.com/50883840/227584063-5b6774b4-f4ed-43c9-976f-106bb0ae317e.jpeg)

