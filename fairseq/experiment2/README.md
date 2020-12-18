
# Experiment 2 (Text Summarization T-CVAE Annealing Schedule)


> Start Date : 2020.12.16

> Finish Data : 👨‍💻

This experiment is finding best KL-annealing strategy to outperform BART base model. 


# Source code

## Scripts
|script|info|
|:--|:-:|
|bart_vae_annealing.py|model1 concat|
|bart_vae_annealing2.py|model2 add|
|annealing_loss.py| annealing strategy KLD loss|
|train.py|Don't need it in here actually|
|train_bart.py|baseline bart + vae part|


#Results


|number|Model|Structure|Total_Updates|Annealing_M|Annealing_R|ROUGE1|ROUGE2|ROUGEL|
|:-:  |:-:  |---      |:-:|:-:|:-:|:-:|:-:|:-:|
|V0|BART|-|29400|-|-| ❌|❌ | ❌|
|V1|BART_VAE|<img src="docs/model1.png" width=200px>|29400| 2| 0.5|❌|❌|❌|
|V2|BART_VAE|<img src="docs/model1.png" width=200px>|29400| 4| 0.5|❌|❌|❌|
|V3|BART_VAE|<img src="docs/model1.png" width=200px>|29400| 8| 0.5|❌|❌|❌|
|V4|BART_VAE|<img src="docs/model1.png" width=200px>|29400| Const.| Const.|❌|❌|❌|
|V5|BART_VAE|<img src="docs/model2.png" width=200px>|29400|2|0.5|❌|❌|❌|
|V6|BART_VAE|<img src="docs/model2.png" width=200px>|29400|4|0.5|❌|❌|❌|
|V7|BART_VAE|<img src="docs/model2.png" width=200px>|29400| 8|0.5|❌|❌|❌|




## Learning Curves

|Overlap|
|---|
|<img src="docs/overlap.png" width=600px>|

|Model|ELBO|Reconstruction Error |KL term|
|---|---|---|---|
|0|<img src="docs/0_elbo.png" width=200px>|<img src="docs/0_recon.png" width=200px>|<img src="docs/0_kl.png" width=200px>|
|BART|<img src="docs/0_bart.png" width=200px>|-|-|
|V_all|<img src="docs/v_elbo.png" width=200px>|<img src="docs/v_ce.png" width=200px>|<img src="docs/v_kl.png" width=200px>|
|V1|<img src="docs/1_elbo.png" width=200px>|<img src="docs/1_ce.png" width=200px>|<img src="docs/1_kl.png" width=200px>|
|V2|<img src="docs/2_elbo.png" width=200px>|<img src="docs/2_ce.png" width=200px>|<img src="docs/2_kl.png" width=200px>|
|V3|<img src="docs/3_elbo.png" width=200px>|<img src="docs/3_ce.png" width=200px>|<img src="docs/3_kl.png" width=200px>|
|V5|<img src="docs/5_elbo.png" width=200px>|<img src="docs/5_ce.png" width=200px>|<img src="docs/5_kl.png" width=200px>|
|V6|<img src="docs/6_elbo.png" width=200px>|<img src="docs/6_ce.png" width=200px>|<img src="docs/6_kl.png" width=200px>|




##  Data

CNN-Daily Mail

    Train Size: 287227
    Test  Size: 11490
