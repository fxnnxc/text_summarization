import os 
import numpy as np
from tqdm import tqdm 
import torch
from fairseq.models.bart import BARTModel
import shutil
import threading
from fairseq import hub_utils

def seperate_file(file_path, save_dir, n=5):
    """
    Seperate file equally
    """
    with open(file_path, "r") as f :
        texts = f.readlines()
    length = len(texts)
    parallel_length = [length//n for i in range(n)] # 17->[3,3,3,3,3]
    residual = length%n
    for i in range(residual):                       # [3,3,3,3,3] -> [4,4,3,3,3]
        parallel_length[i] +=1 
    for i in range(n-1):
        parallel_length[i+1] += parallel_length[i] # [4,4,3,3,3]-> [4,8,11,14,17]
    print("Parallel Distributed Size : ",parallel_length)
    parallel_length = [0] + parallel_length        # [4,8,11,14,17]-> [0, 4,8,11,14,17]
    save_paths = []
    for i in range(n):
        with open(save_dir+f"parallel_{i}", "w") as g:
            text = "".join(texts[parallel_length[i]:parallel_length[i+1]]) 
            g.write(text)
            save_paths.append(save_dir+f"parallel_{i}")

    return save_paths
    

import argparse 
def inference_bart(bart, source_path, save_path, gpu=1):
    # bart.cuda(gpu)
    bart.eval()
    bart.half()
    count = 1
    bsz = 128
    TOTAL = 1e17
    print("BART", source_path, " begin...")
    with open(source_path) as source, open(save_path, 'w') as fout:
        sline = source.readline().strip()
        slines = [sline]
        for sline in tqdm(source):
            if count % bsz == 0:
                with torch.no_grad():
                    hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
                slines = []
    
            slines.append(sline.strip())
            count += 1
        if slines != []:
            hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default='data/cnn_dm-base')
    parser.add_argument("--model", default="checkpoint_bestexperiment3_1.pt")
    parser.add_argument("--parallel", type=int, default=5,  action='store')
    parser.add_argument("--test-file", default="test.source", action="store")
    parser.add_argument("--hypo", default="test.hypo_vae",  action='store')
    parser.add_argument("--gpu", type=int, default=1,  action='store')
    config = parser.parse_args()
    print(config)

    os.chdir("/home/bumjin/fairseq")
    data_path = config.data_path
    #-- Make distributed files
    if not os.path.isdir(data_path+"/source_parallel"):
        os.mkdir(data_path+"/source_parallel")
    else:
        raise ValueError("Remove folder first")

    source_parallel = seperate_file(data_path+"/"+config.test_file, data_path+"/source_parallel/", n=config.parallel)
    
    print("*--------Making Bart List----------*")
    barts = [BARTModel.from_pretrained(
                    'checkpoints/',
                    checkpoint_file=config.model,
                    data_name_or_path=data_path+'-bin').cuda(config.gpu) for i in range(config.parallel)]
    
    print("*--------Making Threads----------*")
    threads = [threading.Thread(target=inference_bart, args=(barts[i], source_parallel[i], source_parallel[i]+"h")) for i in range(config.parallel)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()    

    with open(data_path+"/"+config.hypo, "w") as h:
        for source in source_parallel:
            hypo = source+"h"
            with open(hypo, "r") as hp:
                h.write(hp.read())


    #-- Remove distributed files
    if os.path.isdir(data_path+"/source_parallel"):
        shutil.rmtree(data_path+"/source_parallel")


if __name__=="__main__":
    main()
