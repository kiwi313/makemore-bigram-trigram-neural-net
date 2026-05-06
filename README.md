# Makemore – Bigram vs Trigram (Neural Network)

This project is based on the first part of the *makemore* series by Andrej Karpathy.

## Overview

The goal was to implement and compare two simple character-level language models:

* **Bigram model** – predicts the next character based on one previous character
* **Trigram model** – predicts the next character based on two previous characters

Both models were implemented using a simple neural network approach:

* one-hot encoding (later replaced with direct weight indexing)
* a single linear layer (`W`)
* softmax implemented via exponentiation and normalization

## Dataset

The dataset consists of names (`names.txt`).

It was split into:

* **80% training set**
* **10% validation (dev) set**
* **10% test set**

The models were trained only on the training set.

## Evaluation

Model performance was evaluated using **negative log likelihood (NLL)**
(cross-entropy without regularization).

### Bigram Results

| Model  | Train NLL | Dev NLL | Test NLL |
| ------ | --------: | ------: | -------: |
| Bigram |    2.4592 |  2.4570 |   2.4446 |

### Trigram Results

| Model   | Train NLL | Dev NLL | Test NLL |
| ------- | --------: | ------: | -------: |
| Trigram |    2.2489 |  2.2548 |   2.2563 |

## Key Observations

* The trigram model has access to a larger context than the bigram model.
* With only a small number of training steps, the trigram model initially appeared undertrained and did not outperform the bigram model.
* After increasing the number of gradient descent steps, the trigram model achieved significantly lower NLL.
* The trigram model also generalized better than the bigram model on the dev/test sets.

The trigram model has significantly more parameters:

* Bigram: `27 × 27`
* Trigram: `729 × 27`

As a result, trigram contexts are much sparser, meaning many contexts are seen far less frequently during training. Because of this, the trigram model requires substantially more optimization steps before the larger context becomes beneficial.

## Implementation Notes

* Training uses L2 regularization (`weight decay`) to stabilize learning.
* Evaluation (dev/test) is performed without regularization using pure NLL.
* One-hot encoding was later replaced with direct indexing (`W[xs]`) for simplicity and efficiency.
* Sampling is performed character-by-character using the learned probability distribution over the vocabulary.

## Future Improvements

* Tune regularization strength using the validation set
* Add hidden layers or embeddings
* Experiment with larger context sizes (4-gram, 5-gram)
* Compare neural n-grams with count-based n-gram models

## Conclusion

This project demonstrates that:

* simple neural network language models can replicate the behavior of count-based n-gram models,
* larger-context models are more expressive,
* but larger models may require substantially more optimization steps due to sparse contexts and increased parameter count.
