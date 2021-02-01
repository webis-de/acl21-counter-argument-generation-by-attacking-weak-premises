import sys
import os
import pandas as pd
import random
import numpy as np
from collections import defaultdict

sys.path.append('/workspace/computationally-undermining-arguments/scripts/')
sys.path.append('/workspace/computationally-undermining-arguments/src-py')
sys.path.append('/workspace/computationally-undermining-arguments/thirdparty/transfer-learning-conv-ai/')

import interact


def clean_df(df, title_clm='title', post_clm='post', comment_clm='comment_sents'):
    df[title_clm] = df[title_clm].apply(lambda x : x.strip().lower())
    df[post_clm] = df[post_clm].apply(lambda x : [sent.strip().lower() for sent in x])
    df[comment_clm] = df[comment_clm].apply(lambda x : [sent.strip().lower() for sent in x])
    
    return df

def prepare_data_for_training(dataset, full_counter=True, context='title', post_clm='post', comment_clm='comment_sents', attacks_clm='premise_counter_premise_pairs', max_sens=20, baseline=False):
    json_output = {}
    
    for split_name, split in dataset.items():
        split_data = []
        for row_idx , row in split.iterrows():
            post_counters = []
            premise_counter_premise_pair = row[attacks_clm]
            attacked_sents = [x[0] for x in premise_counter_premise_pair]
            for i, premise_counter_premise_pair in enumerate(premise_counter_premise_pair):

                full_comment = [sent.strip().lower() for sent in row[comment_clm]]

                
                for attack in attacked_sents:
                    for sent in attack:
                        if  sent.strip().lower() in full_comment:
                            full_comment.remove(sent.strip().lower())# removed quoted premises from the comments
                
                candidate = []
                cntr_prem = " ".join(full_comment[0:5]) if full_counter else premise_counter_premise_pair[1]
                random_post_sent = random.sample(row[post_clm], 1)[0]

                candidate.append(random_post_sent)
                candidate.append(cntr_prem) # select the correct candidate

                post_counters.append({
                    'candidates' : candidate,
                    'premise' : [] if baseline else premise_counter_premise_pair[0]
                })
            
            if context== 'title':
                context_str = [row['title']]
            elif context== 'full_post':
                context_str = row[post_clm][0:max_sens]
            elif context== 'title+post':
                context_str = [row['title']] + row[post_clm][0:max_sens]
                
            split_data.append({
                'context': context_str,
                'attacks': post_counters
            })

        json_output[split_name] = split_data
        

    return json_output


def trim_argument(argument, max_length, args):
    while sum([len(s) for s in argument]) >= (max_length - args.max_length) and len(argument) > 1:
        argument = argument[0: -1]

    return argument

def pred_attacks(df, model_path, args, output_clm, context='title', 
                 attacks_clm='premise_counter_premise_pairs', post_clm='post', rand_premise_idx=-1, rand_premises_clm='', baseline=False, random=False):
    model, tokenizer = interact.load_model(model_path)
    
    predictions = []
    for row_idx, row in df.iterrows():
        post_attacks = []
        for premise_counter in row[attacks_clm]:
            
            if context=='title+full_post':
                argument = [tokenizer.encode(row['title'])] + [tokenizer.encode(sent) for sent in row[post_clm]]
            else:
                argument = [tokenizer.encode(row['title'])] if context == 'title' else [tokenizer.encode(sent) for sent in row[post_clm]]
            
            #print([row['title']]+row[post_clm])
            #print(premise_counter[0])
            if baseline:
                argument = trim_argument(argument, 510, args)
                pred_counter = interact.sample_sequence(argument, [], tokenizer, model, args, baseline=True)
            elif random:
                if rand_premise_idx != None:
                    weak_premise_encoded = [tokenizer.encode(row[rand_premises_clm][rand_premise_idx])]
                else:
                    weak_premise_encoded = [tokenizer.encode(row[rand_premises_clm])]
                    
                argument = trim_argument(argument, 510, args)
                pred_counter = interact.sample_sequence(argument, weak_premise_encoded, tokenizer, model, args)
            else:
                weak_premise_encoded = [tokenizer.encode(premise) for premise in premise_counter[0]]
                argument = trim_argument(argument, 510, args)
                pred_counter = interact.sample_sequence(argument, weak_premise_encoded, tokenizer, model, args)
            
            #remove quoted text
            #pred_counter = tokenizer.decode(pred_counter)
            #for p in premise_counter[0]:
            #    pred_counter = pred_counter.replace(p.lower(), '')
            
            post_attacks.append([premise_counter[0], pred_counter])

        predictions.append(post_attacks)
    
    df[output_clm] = predictions
    
    return df

def perform_attacks_hua_df(df, model_path, args, output_clm, context, weak_premise_clm, weak_premise_idx=None, baseline=False):
    model, tokenizer = interact.load_model(model_path)

    pred_attacks = []
    for idx, row in df.iterrows():
        if baseline:
            argument = [row['claim']] + row['post']
            argument = [tokenizer.encode(sentence) for sentence in argument]
            argument = trim_argument(argument, 510, args)
            
            pred_counter = interact.sample_sequence(argument, [], tokenizer, model, args, baseline=True)
            
            pred_counter = pred_counter.replace(row['claim'].lower(), '')
            for sent in row['post']:
                pred_counter = pred_counter.replace(sent.lower(), '')
        else:
            
            if context=='title+full_post':
                argument = [tokenizer.encode(row['claim'])] + [tokenizer.encode(sent) for sent in row['post']]
            else:
                argument = [tokenizer.encode(row['claim'])] if context == 'title' else [tokenizer.encode(sent) for sent in row['post']]
            
            if weak_premise_idx != None:
                weak_premise_encoded = [tokenizer.encode(premise) for premise in row[weak_premise_clm][weak_premise_idx]]
            else:
                weak_premise_encoded = [tokenizer.encode(premise) for premise in row[weak_premise_clm]]
                
            if args.premise_extra:
                argument = trim_argument(argument, 510-len([token for p in weak_premise_encoded for token in p]), args)
            else:
                argument = trim_argument(argument, 510, args)

            pred_counter = interact.sample_sequence(argument, 
                                                     weak_premise_encoded,
                                                     tokenizer, model, args)
            
            #remove quoted text
            #pred_counter = tokenizer.decode(pred_counter)
            for p in row[weak_premise_clm]:
                pred_counter = pred_counter.replace(p.lower(), '')
                        
        pred_attacks.append(pred_counter)

    df[output_clm] = pred_attacks
    
    return df

def overlap_between_attack_and_attacked_premises(attacked_premises, attack):
    import nltk
    from nltk.corpus import stopwords
    en_stopwords = stopwords.words('english')

    attack_tokens =[token for token in nltk.word_tokenize(attack) if token not in en_stopwords]
    
    attacked_premise_tokens = []
    for sentence in attacked_premises:
        attacked_premise_tokens += [token for token in nltk.word_tokenize(sentence) if token not in en_stopwords]

    return len(set(attack_tokens).intersection(attacked_premise_tokens))/len(set(attacked_premise_tokens).union(set(attack_tokens)))