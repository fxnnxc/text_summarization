def novel_1_gram(source, summ):
    source = source.split()
    summ = summ.split()
    count = 0 
    for c in summ:
        if c not in source:
            count +=1
    return count/len(summ)

def novel_2_gram(source, summ):
    source = source.split()
    summ = summ.split()
    n_gram_source = list(nltk.ngrams(source, 2))
    count = 0 
    for i in range(len(summ)-1):
        c = summ[i]
        if c not in source:
            if (c, summ[i+1]) not in n_gram_source:
                count +=1
    return count/len(n_gram_source)

def novel_3_gram(source, summ):
    source = source.split()
    summ = summ.split()
    n_gram_source = list(nltk.ngrams(source, 3))
    count = 0 
    for i in range(len(summ)-2):
        c = summ[i]
        if c not in source:
            if (c, summ[i+1], summ[i+2]) not in n_gram_source:
                    count +=1
    return count/len(n_gram_source)

from tqdm import tqdm

def novel_gram_full(source, target, n ):
    cumm = 0 
    for i in tqdm(range(len(target))):
        if n==1:
            cumm += novel_1_gram(source[i], target[i])
        elif n==2:
            cumm += novel_2_gram(source[i], target[i])
        elif n==3:
            cumm += novel_3_gram(source[i], target[i])
    return cumm/len(target)