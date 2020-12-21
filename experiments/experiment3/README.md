
<p align="center">
  <a href="https://github.com/pytorch/fairseq"><img src=https://img.shields.io/badge/fairseq-v0.10.1-blue?style=flat height=30px></a> 
  <img src=https://img.shields.io/badge/Python-v3.6.8-green?style=flat&logo=python height=30px></a> 
</p>


# Text Summarization Using Latent Space in Transformer (experiment3)

A similar experiment with [experiment2](https://github.com/fxnnxc/text_summarization/tree/main/experiments/experiment2) but more stable code.

Now the experimental codes extend fairseq and I found a better way to build a class which is suitable **to use pretrained model**(Nice!).

> Start Date : 2020.12.19

> Finish Date : 👨‍💻

# Goal
Get higher ROUGE score than the rouge-score of original bart using latent space in the transformer.

# Models
|index|Structure|info|
|---|---|---|
|1|<img src="docs/model1.png" width=500px>|x = source <br/> x =encoder(x) <br/> z1 = GRU(x) <br/> x = decoder(x)  <br/>z2 = GRU(x) |


# Experiment

This experiment used bart_base model, not bart_large. 
because I thought model architecture is more important than model performance itself.  

|Model|Rouge Score(R1,R2,RL)|epoch|
|:-:|:-:|:-:|
|bart-base|0.296 0.073 0.186|8|
|model1|-|


# Source Code

> You can use it **without modification** in the original fairseq code.  


|cat|script|info|
|:-:|:--|:-:|
|criterions|kld_loss.py|KL-Divergence loss is defined|
|models|vae_bart.py|model class|
|models|hub_interface.py|for sampling. it is sample with BART hub-interface|

# 🛩️ Mini Experiments 🛩️

Additional Mini Experiments

## [Mini 1] KL annealing for additive latent variable z

|Loss with/without annealing|
|:-:|
|<img src="docs/mini11.png" width=700px>|

* The KL-D doesn't diverge even with beta 0. It looks stable.  
* The model architecture is additive xz=a\*x +(1-a)\*z

## [Mini 2] KL annealing with different cycle
|beta|KLD|
|---|---|
|![image](https://user-images.githubusercontent.com/51252792/102706323-371bc980-42d4-11eb-8a02-3ae327dc3b97.png)|Orange ![image](https://user-images.githubusercontent.com/51252792/102706632-06895f00-42d7-11eb-8e43-201d91d08a63.png)|
|![image](https://user-images.githubusercontent.com/51252792/102706340-561a5b80-42d4-11eb-9c6e-2701abeab587.png)|BLUE ![image](https://user-images.githubusercontent.com/51252792/102706407-12742180-42d5-11eb-8dcb-01ea01486671.png)|
|![image](https://user-images.githubusercontent.com/51252792/102706455-5cf59e00-42d5-11eb-9625-e1ad08bcca75.png)|RED ![image](https://user-images.githubusercontent.com/51252792/102706468-7eef2080-42d5-11eb-9996-7cb6ce96c6b7.png)|
|![image](https://user-images.githubusercontent.com/51252792/102706490-b2ca4600-42d5-11eb-8ada-674589825e51.png)|**SkyBlue** ![image](https://user-images.githubusercontent.com/51252792/102706537-1fdddb80-42d6-11eb-8903-3e322bc89832.png)|
|![image](https://user-images.githubusercontent.com/51252792/102706587-82cf7280-42d6-11eb-8f7a-da46da0baae6.png)|![image](https://user-images.githubusercontent.com/51252792/102706622-e194ec00-42d6-11eb-981c-04e77cdbdf0d.png)|
|![image](https://user-images.githubusercontent.com/51252792/102706834-a8f61200-42d8-11eb-8505-7a178f1ad1f8.png)|![image](https://user-images.githubusercontent.com/51252792/102706922-8284a680-42d9-11eb-916a-163266ac5a93.png)|



# Train / Inference shell

### Train
```bash
TOTAL_NUM_UPDATES=20000 
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=4
BART_PATH=checkpoints/bart.base/model.pt
CHECKPOINT_SUFFIX=experiment3_V4
ARCH=vae_bart_base
SAVE_INTERVAL=3
MAX_EPOCH=3

CUDA_VISIBLE_DEVICES=0 python train.py data/cnn_dm-base-bin \
    --user-dir examples/vae_bart/vae_bart_src2 \
    --arch $ARCH \
    --pretrained \
    --pretrained-checkpoint $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task text_summarization_annealing \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --criterion kld_loss \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --fp16 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --save-interval $SAVE_INTERVAL \
    --checkpoint-suffix $CHECKPOINT_SUFFIX \
    --max-epoch $MAX_EPOCH ;

```
