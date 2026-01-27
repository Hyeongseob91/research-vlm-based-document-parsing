# Attention Is All You Need

Ashish Vaswani
Noam Shazeer
Niki Parmar
Jakob Uszkoreit
Llion Jones
Aidan N. Gomez
Łukasz Kaiser
Illia Polosukhin

Proceedings of the 31st Conference on Neural Information Processing Systems (NIPS 2017)

---

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.
The best performing models also connect the encoder and decoder through an attention mechanism.

We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.

Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results by over 2 BLEU.
On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8.

We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing.

---

## 1. Introduction

Recurrent neural networks, long short-term memory and gated recurrent neural networks have been firmly established as state-of-the-art approaches in sequence modeling and transduction problems such as language modeling and machine translation.

Recurrent models typically factor computation along the symbol positions of the input and output sequences.
This inherently sequential nature precludes parallelization within training examples and becomes critical at longer sequence lengths.

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models, allowing modeling of dependencies without regard to their distance in the input or output sequences.

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.

The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight GPUs.

---

## 2. Background

The goal of reducing sequential computation also forms the foundation of convolutional sequence models such as ByteNet and ConvS2S.

In these models, the number of operations required to relate signals from two arbitrary positions grows with the distance between positions.

Self-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.

To the best of our knowledge, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned recurrent networks or convolutions.

---

## 3. Model Architecture

Most competitive neural sequence transduction models have an encoder-decoder structure.

The encoder maps an input sequence of symbol representations to a sequence of continuous representations.
The decoder generates an output sequence of symbols one element at a time in an auto-regressive manner.

The Transformer follows this overall architecture using stacked self-attention and point-wise fully connected layers for both the encoder and decoder.

---

### 3.1 Encoder and Decoder Stacks

The encoder is composed of a stack of six identical layers.
Each layer has two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network.

Residual connections are employed around each sub-layer, followed by layer normalization.

The decoder is also composed of six identical layers.
In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer which performs multi-head attention over the output of the encoder stack.

Masking is applied in the decoder self-attention to prevent positions from attending to subsequent positions.

---

### 3.2 Attention

An attention function maps a query and a set of key-value pairs to an output.

The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

---

#### 3.2.1 Scaled Dot-Product Attention

Scaled dot-product attention computes the dot products of the query with all keys, divides each by the square root of the key dimension, and applies a softmax function to obtain weights.

The attention output is the weighted sum of the values.

---

#### 3.2.2 Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

Instead of performing a single attention function, the queries, keys, and values are projected multiple times with different learned linear projections.

The results are concatenated and projected again to form the final output.

---

### 3.3 Position-wise Feed-Forward Networks

Each layer contains a fully connected feed-forward network applied to each position separately and identically.

This consists of two linear transformations with a ReLU activation in between.

---

### 3.4 Embeddings and Softmax

Learned embeddings are used to convert input and output tokens to vectors of fixed dimension.

The same weight matrix is shared between the two embedding layers and the pre-softmax linear transformation.

---

### 3.5 Positional Encoding

Since the model contains no recurrence or convolution, positional information is injected using positional encodings.

These encodings are added to the input embeddings and are based on sine and cosine functions of different frequencies.

---

## 4. Why Self-Attention

Self-attention allows for shorter paths between long-range dependencies compared to recurrent or convolutional layers.

It provides better parallelization and lower computational cost for typical sequence lengths used in machine translation.

---

## 5. Training

The model is trained on standard WMT 2014 English-German and English-French datasets.

Training uses the Adam optimizer with a custom learning rate schedule and employs dropout and label smoothing for regularization.

---

## 6. Results

The Transformer achieves state-of-the-art BLEU scores on both English-to-German and English-to-French translation tasks.

It also generalizes well to English constituency parsing, outperforming many existing models.

---

## 7. Conclusion

We presented the Transformer, a sequence transduction model based entirely on attention mechanisms.

The model achieves superior performance while being significantly more parallelizable and efficient to train.

We plan to apply the Transformer to other modalities and investigate restricted attention mechanisms for very long sequences.
