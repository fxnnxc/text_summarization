
# Experiment 2

> Start Date : 2020.12.16
> Finish Data : 

This experiment is finding best KL-annealing strategy to outperform BART base model. 


|number|Model|Structure|KL-annealing|ROUGE1|ROUGE2|ROUGEL|
|---|---|---|---|---|---|---|
|1|BART|-|-| ❌|❌ | ❌|
|2|BART_VAE|img.png|img.png|❌|❌|❌|




## Learning Curves
|number|ELBO|Reconstruction Error |KL term|
|---|---|---|---|
|0|<img src="docs/1_elbo.png" width=200px>|<img src="docs/1_recon.png" width=200px>|<img src="docs/1_kl.png" width=200px>|



##  Data

CNN-Daily Mail

    Train Size: 287227
    Test  Size: 11490
