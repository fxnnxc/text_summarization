# Text Summarization




---

[![Generic badge](https://img.shields.io/badge/Year-2020-<COLOR>.svg)](https://shields.io/)


---

[![Generic badge](https://img.shields.io/badge/Year-2019-<COLOR>.svg)](https://shields.io/)



[1][VAE][WNGT] [On the Importance of the Kullback-Leibler Divergence Term in Variational Autoencoders for Text Generation](https://github.com/fxnnxc/text_summarization/blob/main/study/2019/On-the-Importance-of-the-Kullback-Leibler-Divergence-Term-in-Variational-Autoencoders-for-Text-Generation.md)
  
[2][VAE] [VAE-PGN based Abstractive model in Multi-stage Archietecture for Text Summarization](https://github.com/fxnnxc/text_summarization/blob/main/study/2019/VAE-PGN%20based%20Abstractive%20model%20in%20Multi-stage%20Archietecture%20for%20Text%20Summarization.md)

[3][VAE] [Cyclic annealing schedule](https://github.com/fxnnxc/text_summarization/blob/main/study/2019/Cyclical%20Annealing%20Schedule.md)


[4] **Nucleus Sampling(NS)**
* *Holtzman et al*
* Threashold p (p=1 means sampling from the full distribution)

[5] **BERTSUM: bert-based extractive model**
* *Liu*
* eliminate less important sentences by scoring each sentence in the source text.


---

[![Generic badge](https://img.shields.io/badge/Year-2018-<COLOR>.svg)](https://shields.io/)


[1][VAE] [Semi-Amortized-Variational-Autoencoders](#..)

**Top-k**
* *Fan et al.,*

**word-level content selectino model to focuse on only critical information**
*Gehrmann et tal.*

---

[![Generic badge](https://img.shields.io/badge/Year-2017-<COLOR>.svg)](https://shields.io/)


[1][PGN] Get To The Point: Summarization with Pointer-Generator Networks

[2] Toward Controlled Generation of Text


---

[![Generic badge](https://img.shields.io/badge/Year-2016-<COLOR>.svg)](https://shields.io/)


**Vanilla VAE applied to text** 
* *Bowmman et al.*

**Beta VAE with annealing** 
* *Bowman et al.*
* gradually increase the beta term while training.

**Copy mechanism**
* *Gu et al. 2016*

---

[![Generic badge](https://img.shields.io/badge/Year-2015-<COLOR>.svg)](https://shields.io/)


---
[![Generic badge](https://img.shields.io/badge/Year-2014-<COLOR>.svg)](https://shields.io/)


**VAEs**
* *Kingma and Welling*

**Generative Adversarial Networks(GANs)**
* *Goodfellow et al.*


---

# Problems 

**Posterior collapse**
* the inference network produces uniformative latent variables

**Sol1.** Modifiying the architecture of the model by weakening decoders
* *Bowman et al., 2016; Miao et al., 2015; Yang et al., 2017; Semeniuta et al., 2017;*

**Sol2.** Introducing additional connections between the encder and decoder to enforce the dependence between x and z
* *Zhao et al., 2017; Goyal et al., 2017; Dieng et al., 2018*


**Sol3.** Using more flexible or multimodal priors
* *Tomczak and Welling, 2017; Xu and Durrett, 2018*

**Sol4.** Alternating the training b  on the inference network in the earlier stages
* *He et al., 2019*

**Sol5.** Augmenting amortized optimization of VAEs with instance based optimization of stochastic variational inference
* *Kim et al., 2018; Marino et al., 2018*

**Sol6.** delta-VAE
* *Razavi et al., 2019*

**Sol7.** beta-VAE
* *Hig- gins et al., 2017*
