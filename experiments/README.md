- [Experiments](#experiments)
  - [🧚‍♂️ : Train freezed BART with addtional moudule](#️--train-freezed-bart-with-addtional-moudule)
    - [1. Is learning rate important?](#1-is-learning-rate-important)


# Experiments

## 🧚‍♂️ : Train freezed BART with addtional moudule
* Goal : Get higher ROUGE score with freezed bart with addtional module. 
* Rule
  * Do not test with more than 1 day. 

### 1. Is learning rate important?

* Token : 20210129

* Setup
  1. Model : BART + BGN(Bag of words Generator Network) from the encoder
  2. Criterion : ROUGE score
  3. Data : XSum

*  Conditions
   *  learning rate(3e-5, )
   *  training  w/wo combining the linear layer of the transformer
* Result
    * https://docs.google.com/spreadsheets/d/12pEqyhzrY7bBsbuEe191Vs0DQyCQLA7EJ_YZQN82Ozo/edit#gid=0


<img src="archived/20210119/model.png" width=500px>

<img src="archived/20210119/loss1.png" width=250px>

						
[Materials](archiveWd/20210119)
* Conclusion


