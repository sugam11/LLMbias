#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:42:18 2023

@author: ninaadj
"""
import matplotlib.pyplot
import pandas as  pd
import ast
import math
import matplotlib.pyplot as plt
import numpy as np

# False is out of total and True out of Toxic  per race subcategory
PERCENT_OF_TOXIC = False  


def strlist_list(str_lst):
    lst = []
    try:
        splits = str_lst.split()
        str_lst = str_lst.replace('\n', ' ')
        str_lst = " ".join(splits)
        str_lst = str_lst.replace('array(', '')
        str_lst = str_lst.replace(', dtype=object)', '')
        str_lst = str_lst.replace(', dtype=object)', '')
        str_lst = str_lst.replace('} {', '}, {')
        lst = ast.literal_eval(str_lst)
    except:
        print('error')
    return lst

def get_matches(match):
    matches = []
    if type(match) is dict:
        matches = match['matches'] 
    else:
        matches = 'None'
    return matches

def get_category(match):
    matches = ''
    if type(match) is dict:
        matches = match['category']
    else:
        matches = 'None'
    return matches

def get_parent_category(match):
    matches = ''
    if type(match) is dict:
        matches = match['parent_categories']
    else:
        matches = 'None'
    return matches

def get_matches_match(match):
    matches = ''
    if math.isnan(match) == False:
        matches = match[0]
    return matches

def process_op(df, split):
    df_len = len(df)
    
    df['bias_matches'] = df['bias_matches'].apply(strlist_list)
    df = df.explode('bias_matches')
    
    df['matches'] = df['bias_matches'].apply(get_matches)
    df['category'] = df['bias_matches'].apply(get_category)
    df['parent_categories'] = df['bias_matches'].apply(get_parent_category)
    
    # Filter only race categories
    df1 = df[df['parent_categories'] == 'race_ethnicity']
   
    # plot race vs regard plot
    categories = df1['category'].unique()
    negatives = []
    positives = []
    neutrals = []
    others = []
    category_counts = []
    for category in categories:
        df2 = df1[df1['category'] == category]
        category_counts.append((len(df2)/df_len) * 100)
        value = {}
        neutrals.append(df2['regard_neutral'].mean())
        negatives.append(df2['regard_negative'].mean())
        positives.append(df2['regard_positive'].mean())
        others.append(df2['regard_other'].mean())
        
    pos_others = np.add(positives, others)
    pos_others_neutrals = np.add(pos_others, neutrals)
    
    fig, ax = plt.subplots(figsize=(10,5))
    
    bars = []
    bars.append(ax.barh(categories, positives, color='blue', label = 'Positive'))
    bars.append(ax.barh(categories, others, left=positives, color='orange', label = 'Other'))
    bars.append(ax.barh(categories, neutrals, left = pos_others, color='green', label = 'Neutral'))
    bars.append(ax.barh(categories, negatives, left = pos_others_neutrals, color='red', label = 'Negative'))
    
    for i, total in enumerate(category_counts):
        ax.text(0.02,i, f'{total:0.4f}%', fontweight='bold', fontsize=14, ha='left', va='center',color='white')
    
    plt.ylabel('Race')
    plt.xlabel('mean toxic')
    
    ax.set_title(f'Regard by Race in Anthropic PPO Trained Output - {split}')
    plt.xticks(rotation=90)
    ax.legend(bbox_to_anchor=(1, 0.5),ncol=1)
    plt.tight_layout()
    plt.ylabel('Race')
    plt.xlabel('mean regard')
    plt.savefig(f"hhrlhf_chosen_{split}_regard_race.jpeg")
    plt.show()
    
    
    # plot race distribution plot
    all_counts = df1.category.explode().fillna('None').value_counts()

    df1 = df1[df1['toxic'] == True]
    counts = df1.category.explode().fillna('None').value_counts()

    values = {}
    for i in range(len(counts)):
        values[counts.index[i]] = {}
        values[counts.index[i]]['all_counts'] = all_counts[counts.index[i]]
        values[counts.index[i]]['counts'] = counts.values[i]
        values[counts.index[i]]['percentage'] = (counts.values[i]/all_counts[counts.index[i]]) * 100
        

    x, y, total_percentages = [], [], []
    for key, value in values.items():
        x.append(key)
        y.append(value['percentage'])
        
        if PERCENT_OF_TOXIC == True:
            total_percentages.append((value['counts']/df_len) * 100)
        else:
            total_percentages.append((value['all_counts']/df_len) * 100)

    y, x, total_percentages = zip(*sorted(zip(y, x, total_percentages), reverse = True))

    fig, ax = plt.subplots(figsize=(10,5))
    plt.xticks(rotation=90)
    plt.title(f"Anthropic PPO Trained Output Toxic Bias Distribution - {split}")
    ax.barh(x, y, height = 0.8, color='mediumseagreen')
    for i, v in enumerate(total_percentages):
        if PERCENT_OF_TOXIC == True:
            ax.text(1.5 , i, f'{v:0.3f}% toxic of total', fontweight='bold', fontsize=14, ha='left', va='center',color='black')
        else:
            ax.text(1.5 , i, f'{v:0.3f}% of total', fontweight='bold', fontsize=14, ha='left', va='center',color='black')

    plt.ylabel('Race')
    plt.xlabel('% Toxic of Race Category')

    plt.savefig(f"chosen_response_toxic_bias_distribution_subcategory_{split}.jpg", bbox_inches = 'tight')
    plt.show()
    plt.close()
    #df1.to_excel(f'anthropic_chosen_response_bias_distribution_{split}.xlsx')
    
 
    

def process(split):

    df = pd.read_csv(f'reward_model_anthropic_{split}.csv')
    

    # Process Chosen 
    process_op(df, split)
    
process('train')
#process('test')