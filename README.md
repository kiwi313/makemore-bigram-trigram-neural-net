# Makemore – Bigram vs Trigram (Neural Network)

This project is based on the first part of the *makemore* series by Andrej Karpathy.

## Overview

The goal was to implement and compare two simple character-level language models:

* **Bigram model** – predicts the next character based on one previous character
* **Trigram model** – predicts the next character based on two previous characters

Both models were implemented using a simple neural network approach:

* one-hot encoding (later replaced with direct indexing)
* a single linear layer (`W`)
* softmax (via exponentiation and normalization)

## Dataset

The dataset consists of names (`names.txt`).

It was split into:

* **80% training set**
* **10% validation (dev) set**
* **10% test set**

The models were trained only on the training set.

## Evaluation

Model performance was evaluated using **negative log likelihood (NLL)**
(i.e. cross-entropy without regularization).

| Model   | Dev NLL | Test NLL |
| ------- | ------: | -------: |
| Bigram  |     ... |      ... |
| Trigram |     ... |      ... |

## Key Observations

* The trigram model has access to more context than the bigram model.
* However, in this simple neural network setup, it did **not significantly improve** performance on the dev/test sets.
* This suggests that increasing context size alone does not guarantee better generalization.

In some cases, the trigram model may better fit the training data, but this does not necessarily translate to better performance on unseen data.

## Implementation Notes

* Training uses L2 regularization to stabilize learning.
* Evaluation (dev/test) is done **without regularization**, using pure NLL.
* One-hot encoding was replaced with direct weight indexing (`W[xs]`) for simplicity and efficiency.
* Sampling is done character-by-character using the learned probability distribution.

## Future Improvements

* Tune regularization strength using the validation set
* Increase training steps
* Add hidden layers or embeddings
* Experiment with larger context sizes (4-gram, 5-gram)

## Conclusion

This project demonstrates that:

* simple neural network language models can replicate n-gram behavior,
* larger context models are more expressive,
* but better generalization requires more than just increasing context size.
