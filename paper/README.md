# Paper

## Name


## Introduction branch

1. **Text Summarization branch**

* Get To The Point: Summarization with Pointer-Generator Networks
* BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
	
* ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training
* PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization
        
        
2. **Variational branch**

* Auto-Encoding Variational Bayes
        VAE
        beta-VAE : 
        KL vanish problem. 
        annealing schedules  : at least 2 
* beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
* Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing


3. **ELBO analysis**

* InfoVAE: Information Maximizing Variational Autoencoders
* ELBO surgery: yet another way to carve up the variational evidence lower bound
* Enhancing Variational Autoencoders with Mutual Information Neural Estimation for Text Generation

4. **CVAE**

* CVAE : Learning structured output representation using deep conditional generative models
* A Transformer-Based Variational Autoencoder for Sentence Generation
* T-CVAE: Transformer-Based Conditioned Variational Autoencoder for Story Completion
*  Improved Variational Neural Machine Translation by Promoting Mutual Information

## Story

  1. Text Summarization
  Text Summarization is ~~~~ Task. 
  there were many approaches  ex ~. by the adherent of transformer ~~
  Even though we can get a very powerful rouge score, we want to directly encode the feature of document. 
  --> I want Encode the global information!!!!
  
  2. Encode Global information using vae
  To encode the global information vae is one of approaches
  Potesterior collapse is one of the problem. 
  even though we do annealing schedule, decoder of transformer is so powerful 
  --> KL Vanishing!!!     
 
 3. modified elbo to keep information 
  There were many studies about ELBO itself and for text generation  

 4. Recent Study related and drawbacks
  CVAE is an good approach to encode informations but there were not enough study 
  for the text summarization task. 
  We propose a model for text summarization which have a high score for ROUGE and Novel N-gram both.
  
  Ours~!!


