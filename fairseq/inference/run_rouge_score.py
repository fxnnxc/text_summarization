import os 
import numpy as np 
import sys 
from rouge_score import rouge_scorer


def main():
    os.chdir("/home/bumjin/fairseq")
    if len(sys.argv)==3:
        data_path = sys.argv[2]
    else:
        data_path = "data/cnn_dm-large"
    target = open(data_path+'/test.target', "r", encoding="utf8")
    
    hypoth = open(data_path+'/'+sys.argv[1],   "r", encoding="utf8").readlines()


    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = []
    for i in range(len(hypoth)):
        scores.append(scorer.score(target.readline(), hypoth[i]))


    R1 = [s['rouge1'][2] for s in scores] #pre, recall, f1
    R2 = [s['rouge2'][2] for s in scores]
    RL = [s['rougeL'][2] for s in scores]

    print("Rouge1-F1 :", np.mean(R1).round(3)) 
    print("Rouge2-F1 :", np.mean(R2).round(3))
    print("RougeL-F1 :", np.mean(RL).round(3))

main()

