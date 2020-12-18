# Experiment 3 (Text Summarization T-CVAE Annealing Schedule) 

A same experiment with [experiment2](https://github.com/fxnnxc/text_summarization/tree/main/experiments/experiment2) but more stable code. now the code extends fairseq, and i found a better way to build a class which is suitable to use pretrained model(Nice!).


> Start Date : 2020.12.19

> Finish Date : 👨‍💻


# Source Code

> You can use it **without modification** in the original fairseq code.  


|cat|script|info|
|:-:|:--|:-:|
|criterions|kld_loss.py|KL-Divergence loss is defined|
|models|vae_bart.py|model class|
|models|hub_interface.py|for sampling. it is sample with BART hub-interface|

