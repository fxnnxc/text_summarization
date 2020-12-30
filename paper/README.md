# Paper

## Name


## Introduction branch

1. Text Summarization branch

        one paper before transformer : Get To The Point: Summarization with Pointer-Generator Networks
        using transformer : BART
        + variations  :  ProphetNet, PEGASUS

        #=
        Text Summarization is ~~~~ Task. 
        there were many approaches  ex ~. by the adherent of transformer ~~
        Even though we can get a very powerful rouge score, we want to directly encode the feature of document. 
        --> I want Encode the global information!!!!
        #=
        
2. Variational branch

        VAE
        beta-VAE : 
        KL vanish problem. 
        annealing schedules  : at least 2 
        
        #=
        To encode the global information vae is one of approaches
        Potesterior collapse is one of the problem. 
        even though we do annealing schedule, decoder of transformer is so powerful 
        --> KL Vanishing!!!     
        #=

3. ELBO analysis 

        InfoVAE, ELBO surgery, Enhancing Variational Autoencoders with Mutual Information Neural Estimation for Text Generation
        --> What is current?

        #=
        There were many studies about ELBO itself and for text generation  

        #=




4. CVAE(two papers : Improved Variational Neural Machine Translation by Promoting Mutual Information )

        Two papers without transformer 
        using transformer 

        --> What is good approach for the Text Summarization
        
        #=
        CVAE is an good approach to encode informations but there were not enough study 
        for the text summarization task. 
        We propose a model for text summarization which have a high score for ROUGE and Novel N-gram both.
        #=

        Ours~!!

---
