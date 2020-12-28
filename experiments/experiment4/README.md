
<p align="center">
  <a href="https://github.com/pytorch/fairseq"><img src=https://img.shields.io/badge/fairseq-v0.10.1-blue?style=flat height=30px></a> 
  <img src=https://img.shields.io/badge/Python-v3.6.8-green?style=flat&logo=python height=30px></a> 
  <img src=https://img.shields.io/badge/Experiment-v4-orange?style=flat height=30px></a> 
</p>


# Text Summarization Using Latent Space in Transformer

A similar experiment with [experiment3](https://github.com/fxnnxc/text_summarization/tree/main/experiments/experiment3) but more stable code.


* Start Date : 2020.12.23
* Finish Date : 👨‍💻

## Goal 🎯
Now using pretrained model is stable.

Find **a novel structure** to encode features in the latent space!



## Experiments 🧾

|code|Model|Beta scheduling|Loss||
|:-:|:-:|:-:|:-:|:-:|
|check_pretrained_model|BART|-|-|ROUGE|
|[simple_vae2](simple_vae2)|-|300||


## Source Codes 👨‍💻

|name|info|
|:-:|:--|
|inference|script for inference|
|check_pretrained_model|script to check whether using pretrained model is same with defining new model and upload the parameters|
|[simple_vae2](simple_vae2)|-|multihead attetion and annealing 300|



# 🛩️ Mini Experiments 🛩️

brief test and results

|Name|info|
|:-:|:-:|
|[print_gradient](print_gradient)|print gradient for specific layer|