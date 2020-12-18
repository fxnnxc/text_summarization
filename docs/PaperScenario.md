# Bumjin's Second Paper

# Goal

BART 모델이 Extractive 한 성질을 가지고 있다. 
내가 제안한 모델은 Extractive한 성질을 줄이면서 RougeScore는 올리는 모델이다.  (Best)


# How?

BART 모델에서 Extractive한 성질을 없애게 강조한다. 
마치 혹독한 환경에 놓인 개처럼 굶주림을 통해서 학습을 촉진시킨다. 
초반에는 모델이 제대로 학습하지 못하면서 역경을 겪지만, 시간이 지나면서 더욱 강인한 모델이 된다. 
이것은 마치 인간이 생각을 하면 할수록 더 좋은 아이디어가 나오는 것과 같다. 
단순히 시간을 들이는 것이 아니라, 방법을 찾기 위해서 시간을 들인다. 


# Architecture





# Experiment 
BART_BASAE + Additional Rule



|Model|R1|R2|R3|
|---|---|---|---|
|BART|-|-|-|
|BART_VAE|-|-|-|
|BART_VAE+|-|-|-|
|BART_VAE++|-|-|-|


