import collections
import nltk
import numpy as np
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score

def check_sig(v1s, v2s, alpha=0.05):
    from scipy import stats

    diff = list(map(lambda x1 , x2: x1 - x2, v1s, v2s))
    is_normal = stats.shapiro(diff)[1] > alpha
    
    ttest = stats.ttest_rel(v1s, v2s) if is_normal else stats.wilcoxon(v1s, v2s)
    if ttest.statistic >=0:
        if (ttest.pvalue/2) <= alpha:
            return True
        else:
            return False
    else:
        return False

def eval_meteor(references, preds, best_match=False):
    if best_match:
        meteor_scores = []
        for refs, pred in zip(references, preds):
            instance_scores = [meteor_score([ref], pred) for ref in refs]
            meteor_scores.append(max(instance_scores))
    else:
        meteor_scores = [meteor_score(inst[0], inst[1]) for inst in zip(references, preds)]
        
    return round(sum(meteor_scores)/len(meteor_scores),3), meteor_scores

def eval_bleu(references, preds, weights=None, best_match=False):
    references  = list(map(lambda refs: [refs.split()] if type(refs) == str else [ref.split() for ref in refs], references))
    preds = list(map(lambda x: x.split(), preds))

    chencherry = SmoothingFunction()

    if best_match:
        bleu_scores = []
        for refs, pred in zip(references, preds):
            instance_bleu_scores = [bleu_score.sentence_bleu([single_ref], pred, weights, smoothing_function=chencherry.method1) for single_ref in refs]
            bleu_scores.append(max(instance_bleu_scores))
            #print('max: ', max(instance_bleu_scores))
            #print('avg: ',bleu_score.sentence_bleu(refs, pred, weights, smoothing_function=chencherry.method1))
            #print('----------')
            
    else: 
        bleu_scores = [bleu_score.sentence_bleu(ref, pred, weights, smoothing_function=chencherry.method1) 
                    for ref, pred in zip(references, preds)]

    
    score = sum(bleu_scores)/len(bleu_scores)


    return round(score * 100, 3), bleu_scores



def eval_model(df_gt, df_preds, best_match=False):
    
    bleu_1, bleu1_scores = eval_bleu(df_gt, df_preds, weights=(1,0,0,0), best_match=best_match)
    bleu, bleu_scores  = eval_bleu(df_gt, df_preds, weights=(0.5, 0.5,0,0), best_match=best_match)
    meteor, meteor_scores = eval_meteor(df_gt, df_preds, best_match=best_match)

    return [meteor, bleu_1, bleu], [bleu1_scores, bleu_scores, meteor_scores]

def prec_at(ranking_problems, baseline=None, k=1):
    correct_cases = 0
    total_num_cases = 0
    for problem in ranking_problems:      
        if baseline != None:
            if baseline == 'sen_length':
                selected_premises = sorted(problem['documents'], key=lambda x: -len(nltk.word_tokenize(x['docText'])))[0:k]
            else:
                selected_premises = np.random.choice(problem['documents'], k)
        else:
            selected_premises = sorted(problem['documents'], key=lambda x: -x['score'])[0:k]

        if 1 in [x['relevance'] for x in selected_premises]:
            correct_cases+=1
        
        total_num_cases+=1
        
    return round(correct_cases/total_num_cases, 3)


def perform_significance_tests(app1_scores, app2_cores):
    all_data = np.array(list(zip(app1_scores, app2_cores)))
    chunks = np.array_split(all_data, 5, axis=2)
    
    scores_1 = [x[:,0,:].mean(axis=1) for x in chunks]
    scores_2 = [x[:,1,:].mean(axis=1) for x in chunks]
        
    sig_report = {}
    for idx, measure in enumerate(['bleu-1', 'bleu', 'meteor']):
        s1 = [round(s[idx], 3) for s in scores_1]
        s2 = [round(s[idx], 3) for s in scores_2]

        sig_report[measure] = {'@5%':check_sig(s2, s1, alpha=0.05), 
                               '%10': check_sig(s2, s1, alpha=0.1)
        }
        
    return sig_report