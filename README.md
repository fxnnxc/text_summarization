<p align="left">
  <a href="https://github.com/pytorch/fairseq"><img src=https://img.shields.io/badge/fairseq-v0.10.1-blue?style=flat height=30px></a> 
  <img src=https://img.shields.io/badge/Python-v3.6.8-green?style=flat&logo=python height=30px></a> 
</p>

# text_summarization


## New

* 12/19 Implementated [Inference bart parallel](https://github.com/fxnnxc/text_summarization/blob/main/experiments/experiment3/inference/bart_base_inference_parallel.py)(spped up!)
* 12/20 [Experiment3-Mini2](https://github.com/fxnnxc/text_summarization/tree/main/experiments/experiment3#%EF%B8%8F-mini-experiments-%EF%B8%8F) finished(cyclic annealing) 
* 12/21 Implementated [Inference bart vae parallel](https://github.com/fxnnxc/text_summarization/blob/main/experiments/experiment3/inference/bart_vae_inference_parallel.py)(spped up!)


## Experiments 🥼

| Index | Info | Stability  <br/> of Code|
|:-:|:-:|:--|
|[Experiment1](https://github.com/fxnnxc/text_summarization/tree/main/experiments/experiment1)|BART + beta-VAE using LSTM|[![All Contributors](https://img.shields.io/badge/build-Unstable-red)](#contributors-)|
|[Experiment2](https://github.com/fxnnxc/text_summarization/tree/main/experiments/experiment2)|BART + beta-VAE LSTM with anealing schedule|[![All Contributors](https://img.shields.io/badge/build-Unstable-red)](#contributors-)|
|[Experiment3](https://github.com/fxnnxc/text_summarization/tree/main/experiments/experiment3)|BART + beta-VAE with annealing schedule| [![All Contributors](https://img.shields.io/badge/build-Unstable-red)](#contributors-) |
|[Experiment4](https://github.com/fxnnxc/text_summarization/tree/main/experiments/experiment4)|BART + beta-VAE with annealing schedule| [![All Contributors](https://img.shields.io/badge/build-Stable-green)](#contributors-) |

* **make_datafiles.py** : seperate files to speed up inference



## Literature Riview
* [Text Summarization](https://github.com/fxnnxc/text_summarization/tree/main/study)

## Ideas
Check [Novel Ideas](https://github.com/fxnnxc/text_summarization/tree/main/study/novel_idea) 


# Todo

- [ ] Find a good model
- [ ] Train bart base
