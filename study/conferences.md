# Conferences

1. AAAI
2. ACL
3. EMNLP
4. COLING

--- 
### Types

* **VAE**(just variational autoencoder), **BVA**(Beta VAE, Batch VAE), **KLD**(Non VAE but used KLD)
* **TRF**(Transformer), **RNN**(RNN, LSTM, GRU), **PGN**(Pointer-Generator Network), **GAN** , **BRT**(BERT)
* **ABS**(Abstractive), **EXT**(Extractive)
* **TRU**(Truthfulness)
* **SUM**(summarization), **SVY**(Survey) 
* **CNM**(CNN-Daily Mail)

* *Extra types* are stated at the last section


### Relevance(**R**) with the VAE-Transformer text summarization using CNN-DM data
Low 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ High
* 1️⃣ : irrelevant 
* 2️⃣ : simliar task 
* 3️⃣ : same task but different way
* 4️⃣ : same task and similar way
* 5️⃣ : Almost Same 

### Uesfulness(**U**)
* 🤍 : useless
* 💛 : hot potato
* ❤️ : treasure
---

## ACL(Association for Computational Linguistics)


<img src="docs/ACL_logo.png" width=400px>


<p>
<img src=https://img.shields.io/static/v1?label=Year&message=2020&color=blue&style=flat height=28px>
 </p>
 
|Title|Type|**R**|**U**|
|---|---|:-:|:-:|
|[Examining the State-of-the-Art in News Timeline Summarization](https://www.aclweb.org/anthology/2020.acl-main.122/)|SVY|1|🤍|
|[Improving Truthfulness of Headline Generation](https://www.aclweb.org/anthology/2020.acl-main.123/)|TRU|1|🤍|
|[Attend, Translate and Summarize: An Efficient Method for Neural Cross-Lingual Summarization](https://www.aclweb.org/anthology/2020.acl-main.121/)|PGN, TRF, EX1|3|💛|
|[Self-Attention Guided Copy Mechanism for Abstractive Summarization](https://www.aclweb.org/anthology/2020.acl-main.125/)|PGN, TRF, CNM|4|❤️|
|[Attend to Medical Ontologies: Content Selection for Clinical Abstractive Summarization](https://www.aclweb.org/anthology/2020.acl-main.172/)|RNN, SUM|2|🤍|
|[On Faithfulness and Factuality in Abstractive Summarization](https://www.aclweb.org/anthology/2020.acl-main.173/)|TRU, CNM|2|❤️
|[Screenplay Summarization Using Latent Narrative Structure](https://www.aclweb.org/anthology/2020.acl-main.174/)|KLD, EX2|2|💛
|[Unsupervised Opinion Summarization with Noising and Denoising](https://www.aclweb.org/anthology/2020.acl-main.175/)|RNN, EX3, EX4|2|❤️|
|[Improving Adversarial Text Generation by Modeling the Distant Future](https://www.aclweb.org/anthology/2020.acl-main.227/)|GAN,RNN|2|🤍|
|[A Batch Normalized Inference Network Keeps the KL Vanishing Away](https://www.aclweb.org/anthology/2020.acl-main.235/)|RNN, BVA|3|❤️|
|[Topological Sort for Sentence Ordering](https://www.aclweb.org/anthology/2020.acl-main.248/)|-|1|💛|
|[From Arguments to Key Points: Towards Automatic Argument Summarization](https://www.aclweb.org/anthology/2020.acl-main.371/)|-|1|🤍|
|[A Transformer-based Approach for Source Code Summarization](https://www.aclweb.org/anthology/2020.acl-main.449/)|TRF|2|💛|
|[Discourse-Aware Neural Extractive Text Summarization](https://www.aclweb.org/anthology/2020.acl-main.451/)|-|1|🤍|
|[Discrete Optimization for Unsupervised Sentence Summarization with Word-Level Extraction](https://www.aclweb.org/anthology/2020.acl-main.452/)|BERT|2|💛|
|[Understanding Points of Correspondence between Sentences for Abstractive Summarization](https://www.aclweb.org/anthology/2020.acl-srw.26/)|TRU,TRF|1|💛|
|[On the Encoder-Decoder Incompatibility in Variational Text Modeling and Beyond](https://www.aclweb.org/anthology/2020.acl-main.316/)|VAE|2|❤️
|[Autoencoding Pixies: Amortised Variational Inference with Graph Convolutions for Functional Distributional Semantics](https://www.aclweb.org/anthology/2020.acl-main.367/)|BRT, VAE|2|❤️
|[Crossing Variational Autoencoders for Answer Retrieval](https://www.aclweb.org/anthology/2020.acl-main.498/)|VAE in QA|2|❤️
|[Evidence-Aware Inferential Text Generation with Vector Quantised Variational AutoEncoder](https://www.aclweb.org/anthology/2020.acl-main.544/)|VAE, informatino inject|1|🤍
|[Interpretable Operational Risk Classification with Semi-Supervised Variational Autoencoder](https://www.aclweb.org/anthology/2020.acl-main.78/)|RNN, VAE|1|🤍
|[Pre-train and Plug-in: Flexible Conditional Text Generation with Variational Auto-Encoders](https://www.aclweb.org/anthology/2020.acl-main.23/)|RNN, plug-in VAE|3|❤️
|[Guiding Variational Response Generator to Exploit Persona](https://www.aclweb.org/anthology/2020.acl-main.7/)|LSTM, VAE|3|❤️
|[Variational Neural Machine Translation with Normalizing Flows](https://www.aclweb.org/anthology/2020.acl-main.694/)|VAE, TRF|4|❤️
|[Addressing Posterior Collapse with Mutual Information for Improved Variational Neural Machine Translation](https://www.aclweb.org/anthology/2020.acl-main.753/)|VAE, TRF|5|❤️
|[Semi-supervised Parsing with a Variational Autoencoding Parser](https://www.aclweb.org/anthology/2020.iwpt-1.5/)|형태소레벨|2|💛|
|[Generating Diverse and Consistent QA pairs from Contexts with Information-Maximizing Hierarchical Conditional VAEs](https://www.aclweb.org/anthology/2020.acl-main.20/)|Hierarchical VAE, BERT, QA|2|💛

<p>
<img src=https://img.shields.io/static/v1?label=Year&message=2019&color=blue&style=flat height=28px>
 </p>
 
 
|Title|Type|**R**|**U**|
|---|---|:-:|:-:|
|[Semi-supervised Stochastic Multi-Domain Learning using Variational Inference](https://www.aclweb.org/anthology/P19-1186/)|Semi-Sup. VAE|2|💛
|[Syntax-Infused Variational Autoencoder for Text Generation](https://www.aclweb.org/anthology/P19-1199/)|형태소레벨|1|🤍
|[Variational Pretraining for Semi-supervised Text Classification](https://www.aclweb.org/anthology/P19-1590/)|Bag of Words, Good Experience Setting, VAE|2|🤍|
|[Auto-Encoding Variational Neural Machine Translation](https://www.aclweb.org/anthology/W19-4315/)|VAE, NMT|2|🤍
|[Improving Abstractive Document Summarization with Salient Information Modeling](https://www.aclweb.org/anthology/P19-1205/)|sailent network, CNN, CNN-DM|3|💛
|[Abstractive Text Summarization Based on Deep Learning and Semantic Content Generalization](https://www.aclweb.org/anthology/P19-1501/)|Liguistic, more information|2|🤍

<p>
<img src=https://img.shields.io/static/v1?label=Year&message=2018&color=blue&style=flat height=28px>
 </p>

|Title|Type|**R**|**U**|
|---|---|:-:|:-:|
[A Unified Model for Extractive and Abstractive Summarization using Inconsistency Loss](https://www.aclweb.org/anthology/P18-1013/)|extract, PGN,RNN, diff loss|2|💛
[Retrieve, Rerank and Rewrite: Soft Template Based Neural Summarization](https://www.aclweb.org/anthology/P18-1015/)|RNN, template based | 2 |💛
[Extractive Summarization with SWAP-NET: Sentences and Words from Alternating Pointer Networks](https://www.aclweb.org/anthology/P18-1014/)|SWAP-NET, PGN |2 |💛 
[Neural Document Summarization by Jointly Learning to Score and Select Sentences](https://www.aclweb.org/anthology/P18-1061/)|graph|2 |💛
[Unsupervised Abstractive Meeting Summarization with Multi-Sentence Compression and Budgeted Submodular Maximization](https://www.aclweb.org/anthology/P18-1062/)|graph |2|❤️
[Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](https://www.aclweb.org/anthology/P18-1063/)|RNN, RL guided PGN|2|💛
[Soft Layer-Specific Multi-Task Summarization with Entailment and Question Generation](https://www.aclweb.org/anthology/P18-1064/)|RNN, attention, multi-task|1|🤍|
[Global Encoding for Abstractive Summarization](https://www.aclweb.org/anthology/P18-2027/)|RNN, remove repetition|2|🤍
[Unsupervised Semantic Abstractive Summarization](https://www.aclweb.org/anthology/P18-3011/)|Bag of words, Graph|1|🤍
[Reinforced Extractive Summarization with Question-Focused Rewards](https://www.aclweb.org/anthology/P18-3015/)|RL|-|-
[Controllable Abstractive Summarization](https://www.aclweb.org/anthology/W18-2706/)|SVY|-|-
[A Structured Variational Autoencoder for Contextual Morphological Inflection](https://www.aclweb.org/anthology/P18-1245/)|-|-|-
[Variational Inference and Deep Generative Models](https://www.aclweb.org/anthology/P18-5003/)|-|-|-|
[StructVAE: Tree-structured Latent Variable Models for Semi-supervised Semantic Parsing](https://www.aclweb.org/anthology/P18-1070/)|graph, choose action|3|❤️|
[Accelerating Neural Transformer via an Average Attention Network](https://www.aclweb.org/anthology/P18-1166/)|transformer, average attention|2|❤️




<p>
<img src=https://img.shields.io/static/v1?label=Year&message=2017&color=blue&style=flat height=28px>
 </p>

|Title|Type|**R**|**U**|
|---|---|:-:|:-:|
[Diversity driven attention model for query-based abstractive summarization](https://www.aclweb.org/anthology/P17-1098/)
[Get To The Point: Summarization with Pointer-Generator Networks](https://www.aclweb.org/anthology/P17-1099/)
[Supervised Learning of Automatic Pyramid for Optimization-Based Multi-Document Summarization](https://www.aclweb.org/anthology/P17-1100/)
[Selective Encoding for Abstractive Sentence Summarization](https://www.aclweb.org/anthology/P17-1101/)
[Abstractive Document Summarization with a Graph-Based Attentional Neural Model](https://www.aclweb.org/anthology/P17-1108/)
[Joint Optimization of User-desired Content in Multi-document Summaries by Learning from User Feedback](https://www.aclweb.org/anthology/P17-1124/)
[A Principled Framework for Evaluating Summarizers: Comparing Models of Summary Quality against Human Judgments](https://www.aclweb.org/anthology/P17-2005/)
[Oracle Summaries of Compressive Summarization](https://www.aclweb.org/anthology/P17-2043/)
[Multi-space Variational Encoder-Decoders for Semi-supervised Labeled Sequence Transduction](https://www.aclweb.org/anthology/P17-1029/)
[Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders](https://www.aclweb.org/anthology/P17-1061/)
[A Conditional Variational Framework for Dialog Generation](https://www.aclweb.org/anthology/P17-2080/)
[Variation Autoencoder Based Network Representation Learning for Classification](https://www.aclweb.org/anthology/P17-3010/)


---

## COLING(Computational Linguistics)

<img src="docs/COLING_logo.png" width=300px>

<p>
<img src=https://img.shields.io/static/v1?label=Year&message=2020&color=blue&style=flat height=28px>
 </p>
 
|Title|Type|**R**|**U**|
|---|---|:-:|:-:|
|[LAVA: Latent Action Spaces via Variational Auto-encoding for Dialogue Policy Optimization](https://www.aclweb.org/anthology/2020.coling-main.41/)|||
|[Variational Autoencoder with Embedded Student-t Mixture Model for Authorship Attribution](https://www.aclweb.org/anthology/2020.coling-main.45/)|||
|[R-VGAE: Relational-variational Graph Autoencoder for Unsupervised Prerequisite Chain Learning](https://www.aclweb.org/anthology/2020.coling-main.99/)|||
|[A Semantically Consistent and Syntactically Variational Encoder-Decoder Framework for Paraphrase Generation](https://www.aclweb.org/anthology/2020.coling-main.102/)|||
|[Semi-Supervised Dependency Parsing with Arc-Factored Variational Autoencoding](https://www.aclweb.org/anthology/2020.coling-main.224/)|||
|[Improving Variational Autoencoder for Text Modelling with Timestep-Wise Regularisation](https://www.aclweb.org/anthology/2020.coling-main.216/)|||
|[Auto-Encoding Variational Bayes for Inferring Topics and Visualization](https://www.aclweb.org/anthology/2020.coling-main.458/)|||

---

## EMNLP(Empirical Methods in Natural Language Processing)

<p>
<img src=https://img.shields.io/static/v1?label=Year&message=2020&color=blue&style=flat height=28px>
 </p>


|Title|Type|**R**|**U**|
|---|---|:-:|:-:|
[A Spectral Method for Unsupervised Multi-Document Summarization](https://www.aclweb.org/anthology/2020.emnlp-main.32/)
[What Have We Achieved on Text Summarization?](https://www.aclweb.org/anthology/2020.emnlp-main.33/)
[Q-learning with Language Model for Edit-based Unsupervised Summarization](https://www.aclweb.org/anthology/2020.emnlp-main.34/)
[Friendly Topic Assistant for Transformer Based Abstractive Summarization](https://www.aclweb.org/anthology/2020.emnlp-main.35/)
[Multi-document Summarization with Maximal Marginal Relevance-guided Reinforcement Learning](https://www.aclweb.org/anthology/2020.emnlp-main.136/)
[Multistage Fusion with Forget Gate for Multimodal Summarization in Open-Domain Videos](https://www.aclweb.org/anthology/2020.emnlp-main.144/)
[Modeling Content Importance for Summarization with Pre-trained Language Models](https://www.aclweb.org/anthology/2020.emnlp-main.293/)
[Neural Extractive Summarization with Hierarchical Attentive Heterogeneous Graph Network](https://www.aclweb.org/anthology/2020.emnlp-main.295/)
[Coarse-to-Fine Query Focused Multi-Document Summarization](https://www.aclweb.org/anthology/2020.emnlp-main.296/)
[Pre-training for Abstractive Document Summarization by Reinstating Source Text](https://www.aclweb.org/anthology/2020.emnlp-main.297/)
[Few-Shot Learning for Opinion Summarization](https://www.aclweb.org/anthology/2020.emnlp-main.337/)
[Learning to Fuse Sentences with Transformers for Summarization](https://www.aclweb.org/anthology/2020.emnlp-main.338/)
[Stepwise Extractive Summarization and Planning with Structured Transformers](https://www.aclweb.org/anthology/2020.emnlp-main.339/)
[Factual Error Correction for Abstractive Summarization Models](https://www.aclweb.org/anthology/2020.emnlp-main.506/)
[Compressive Summarization with Plausibility and Salience Modeling](https://www.aclweb.org/anthology/2020.emnlp-main.507/)
[Understanding Neural Abstractive Summarization Models via Uncertainty](https://www.aclweb.org/anthology/2020.emnlp-main.508/)
[Multi-hop Inference for Question-driven Summarization](https://www.aclweb.org/anthology/2020.emnlp-main.547/)
[TESA: A Task in Entity Semantic Aggregation for Abstractive Summarization](https://www.aclweb.org/anthology/2020.emnlp-main.646/)
[MLSUM: The Multilingual Summarization Corpus](https://www.aclweb.org/anthology/2020.emnlp-main.647/)
[Multi-XScience: A Large-scale Dataset for Extreme Multi-document Summarization of Scientific Articles](https://www.aclweb.org/anthology/2020.emnlp-main.648/)
[Intrinsic Evaluation of Summarization Datasets](https://www.aclweb.org/anthology/2020.emnlp-main.649/)
[On Extractive and Abstractive Neural Document Summarization with Transformer Language Models](https://www.aclweb.org/anthology/2020.emnlp-main.748/)
[Multi-Fact Correction in Abstractive Text Summarization](https://www.aclweb.org/anthology/2020.emnlp-main.749/)
[Evaluating the Factual Consistency of Abstractive Text Summarization](https://www.aclweb.org/anthology/2020.emnlp-main.750/)
[Re-evaluating Evaluation in Text Summarization](https://www.aclweb.org/anthology/2020.emnlp-main.751/)
[Do We Really Need That Many Parameters In Transformer For Extractive Summarization? Discourse Can Help!](https://www.aclweb.org/anthology/2020.codi-1.13/)
[Artemis: A Novel Annotation Methodology for Indicative Single Document Summarization](https://www.aclweb.org/anthology/2020.eval4nlp-1.8/)
[Best Practices for Crowd-based Evaluation of German Summarization: Comparing Crowd, Expert and Automatic Evaluation](https://www.aclweb.org/anthology/2020.eval4nlp-1.16/)
[A Hierarchical Network for Abstractive Meeting Summarization with Cross-Domain Pretraining](https://www.aclweb.org/anthology/2020.findings-emnlp.19/)
[ZEST: Zero-shot Learning from Text Descriptions using Textual Similarity and Visual Summarization](https://www.aclweb.org/anthology/2020.findings-emnlp.50/)
[Conditional Neural Generation using Sub-Aspect Functions for Extractive News Summarization](https://www.aclweb.org/anthology/2020.findings-emnlp.131/)
[Unsupervised Extractive Summarization by Pre-training Hierarchical Transformers](https://www.aclweb.org/anthology/2020.findings-emnlp.161/)
[TED: A Pretrained Unsupervised Summarization Model with Theme Modeling and Denoising](https://www.aclweb.org/anthology/2020.findings-emnlp.168/)
[KLearn: Background Knowledge Inference from Summarization Data](https://www.aclweb.org/anthology/2020.findings-emnlp.188/)
[Reducing Quantity Hallucinations in Abstractive Summarization](https://www.aclweb.org/anthology/2020.findings-emnlp.203/)
[Abstractive Multi-Document Summarization via Joint Learning with Single-Document Summarization](https://www.aclweb.org/anthology/2020.findings-emnlp.231/)
[Towards Zero-Shot Conditional Summarization with Adaptive Multi-Task Fine-Tuning](https://www.aclweb.org/anthology/2020.findings-emnlp.289/)
[SupMMD: A Sentence Importance Model for Extractive Summarization using Maximum Mean Discrepancy](https://www.aclweb.org/anthology/2020.findings-emnlp.367/)
[NMF Ensembles? Not for Text Summarization!](https://www.aclweb.org/anthology/2020.insights-1.14/)
[MAST: Multimodal Abstractive Summarization with Trimodal Hierarchical Attention](https://www.aclweb.org/anthology/2020.nlpbt-1.7/)
[Sparse Optimization for Unsupervised Extractive Summarization of Long Documents with the Frank-Wolfe Algorithm](https://www.aclweb.org/anthology/2020.sustainlp-1.8/)
[Variational Hierarchical Dialog Autoencoder for Dialog State Tracking Data Augmentation](https://www.aclweb.org/anthology/2020.emnlp-main.274/)
[Public Sentiment Drift Analysis Based on Hierarchical Variational Auto-encoder](https://www.aclweb.org/anthology/2020.emnlp-main.307/)
[Learning Variational Word Masks to Improve the Interpretability of Neural Text Classifiers](https://www.aclweb.org/anthology/2020.emnlp-main.347/)
[VCDM: Leveraging Variational Bi-encoding and Deep Contextualized Word Representations for Improved Definition Modeling](https://www.aclweb.org/anthology/2020.emnlp-main.513/)
[Composed Variational Natural Language Generation for Few-shot Intents](https://www.aclweb.org/anthology/2020.findings-emnlp.303/)
[Controllable Text Generation with Focused Variation](https://www.aclweb.org/anthology/2020.findings-emnlp.339/)


---


## NAACL(North American Chapter of the Association for Computational Linguistics)

<p>
<img src=https://img.shields.io/static/v1?label=Year&message=2019&color=blue&style=flat height=28px>
 </p>

|Title|Type|**R**|**U**|
|---|---|:-:|:-:|
[Topic-Guided Variational Auto-Encoder for Text Generation](https://www.aclweb.org/anthology/N19-1015/)
[Riemannian Normalizing Flow on Variational Wasserstein Autoencoder for Text Modeling](https://www.aclweb.org/anthology/N19-1025/)
[A Variational Approach to Weakly Supervised Document-Level Multi-Aspect Sentiment Classification](https://www.aclweb.org/anthology/N19-1036/)
[Combining Sentiment Lexica with a Multi-View Variational Autoencoder](https://www.aclweb.org/anthology/N19-1065/)
[Keyphrase Generation: A Text Summarization Struggle](https://www.aclweb.org/anthology/N19-1070/)
[Crowdsourcing Lightweight Pyramids for Manual Summary Evaluation](https://www.aclweb.org/anthology/N19-1072/)
[Fast Concept Mention Grouping for Concept Map-based Multi-Document Summarization](https://www.aclweb.org/anthology/N19-1074/)
[Modeling Recurrence for Transformer](https://www.aclweb.org/anthology/N19-1122/)
[Star-Transformer](https://www.aclweb.org/anthology/N19-1133/)
[Single Document Summarization as Tree Induction](https://www.aclweb.org/anthology/N19-1173/)
[A Robust Abstractive System for Cross-Lingual Summarization](https://www.aclweb.org/anthology/N19-1204/)
[Text Generation from Knowledge Graphs with Graph Transformers](https://www.aclweb.org/anthology/N19-1238/)
[Abstractive Summarization of Reddit Posts with Multi-level Memory Networks](https://www.aclweb.org/anthology/N19-1260/)
[Automatic learner summary assessment for reading comprehension](https://www.aclweb.org/anthology/N19-1261/)
[Guiding Extractive Summarization with Question-Answering Rewards](https://www.aclweb.org/anthology/N19-1264/)
[How Large a Vocabulary Does Text Classification Need? A Variational Approach to Vocabulary Selection](https://www.aclweb.org/anthology/N19-1352/)
[Recommendations for Datasets for Source Code Summarization](https://www.aclweb.org/anthology/N19-1394/)
[Question Answering as an Automatic Evaluation Metric for News Article Summarization](https://www.aclweb.org/anthology/N19-1395/)
[Understanding the Behaviour of Neural Abstractive Summarizers using Contrastive Examples](https://www.aclweb.org/anthology/N19-1396/)
[Jointly Extracting and Compressing Documents with Summary State Representations](https://www.aclweb.org/anthology/N19-1397/)
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://www.aclweb.org/anthology/N19-1423/)


<p>
<img src=https://img.shields.io/static/v1?label=Year&message=2018&color=blue&style=flat height=28px>
 </p>
 
|Title|Type|**R**|**U**|
|---|---|:-:|:-:|
[Entity Commonsense Representation for Neural Abstractive Summarization](https://www.aclweb.org/anthology/N18-1064/)
[Newsroom: A Dataset of 1.3 Million Summaries with Diverse Extractive Strategies](https://www.aclweb.org/anthology/N18-1065/)
[Deep Communicating Agents for Abstractive Summarization](https://www.aclweb.org/anthology/N18-1150/)
[Estimating Summary Quality with Pairwise Preferences](https://www.aclweb.org/anthology/N18-1152/)
[Generating Topic-Oriented Summaries Using Neural Attention](https://www.aclweb.org/anthology/N18-1153/)
[Provable Fast Greedy Compressive Summarization with Any Monotone Submodular Function](https://www.aclweb.org/anthology/N18-1157/)
[Ranking Sentences for Extractive Summarization with Reinforcement Learning](https://www.aclweb.org/anthology/N18-1158/)
[Relational Summarization for Corpus Analysis](https://www.aclweb.org/anthology/N18-1159/)
[Which Scores to Predict in Sentence Regression for Text Summarization?](https://www.aclweb.org/anthology/N18-1161/)
[A Hierarchical Latent Structure for Variational Conversation Modeling](https://www.aclweb.org/anthology/N18-1162/)
[Variational Knowledge Graph Reasoning](https://www.aclweb.org/anthology/N18-1165/)



---

## AAAI



---


### Extra Types

|Extra|Info|
|---|---|
|EX1|Cross Lingual Summaization|
|EX2|Screen to Summarization|
|EX3|Unsupervised Summarization|
|EX4|Noise Related|
