# Bart Base Inference

|script|info|
|:-:|:--|
|bart_base_inference_parallel.py|inference of bart using parallel threads|
|bart_base_inference.ipynb|notebook file for bartbase inference|
|bart_vae_inference.ipynb|notebook file for vae_bart inference|
|ROUGE_SCORE.ipynb|Calculate just rouge score and no inference|


```bash
python bart_base_inference_parallel.py \
  --data_path data/cnn_dm-base \
  --model checkpoint_bestBART_BASE.pt \
  --num_parallel 5 \
  --hypo_name test.hypo1 \
  --gpu 1 ;
```


# VAE_Bart Inference
```bash
Not yet implemented
```
