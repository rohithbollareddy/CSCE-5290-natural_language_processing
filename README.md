# Text Summarization using GPT-2 and Pytorch

## Motivation

The meaning of a text can be determined by two ways: rephrasing the text also known as abstractive summarization and identifying the most important keywords also known as extractive summarization. The second way is not so much credited as it often fails to organize the sentences in a meaningful way and many times fails to convey the context of the content. Deep Learning language models like GPT-2, GPT-3 and BERT which are pre-trained use Deep Learning to model natural language and to produce text. We finetune these models on a dataset of articles and their summaries to make  Text Create Summarizer to help summarize the large portion of the text to a shorter and meaningful version. 


## Significance

GPT-2 (Generative Pre-trained Transformer 2) is a state of art transformer model created by open AI which implements a decoder-only transformer architecture based neural network that uses attention rather than recurrent and convolutional based models. This mechanism allows the model to focus on most relatively significant parts which allows increased parallelization rather than other models. This GPT- 2 has been trained with 1.5 billion parameters.

The main purpose of this project is to summarize the overall content of the text to a meaningful text which is relatively smaller than the later. Before the advent of deep learning, the approach to Natural Language processing was mostly rule based with simple Machine Learning algorithms. But, today, with deep learning and topics of Natural language processing enabled models to be trained and analyze the text in a non sequential manner increased greatly. 

The attention mechanism that is used in GPT-2 proved to be extremely rewardful in drawing the generalized context of the text. In today’s world, a lot of business entities take the reviews and surveys of their products and services which enables them to develop and modify according to the market and in customer interest. One such use of text summarization also comes in use to draw important and valid information.


## Objectives:

In this project, we will work on an Abstractive text summarization approach to train the text summariser and then fine tune the pre trained models without adding any new parameters.
We’ll be using a non anonymised Dataset which contains text or articles
Data Preprocessing: We’ll clean the data and remove all the unnecessary characters and to acquire a fixed length input for which we’ll take the average length (Double the average is the full length of the text.
We then tokenize the text and send it to the GPT-2 model and initialize an optimiser.
We create a simple data set that extends the pytorch dataset class.
Then we train the model where the loss is consistently decreased with each epoch or iteration.
A review is initially fed into the model where the top n- choices are selected and added to the summary which is later fed into the model. This process of selecting top N choices, adding it to the summary and feeding it into the model is repeated until we reach the max length or end of sentence
We will experiment by tuning with different hyper parameters like learning rate, optimiser, number of epochs, max_grad_norm, etc.., and determine the optimal hyperparameters for the GPT-2 model.
Expected Result: We are trying to achieve a summary which is meaningful and conveys the entire context of the input text by modifying the fine tuning parameters. (like learning rate, optimizer, training schedule).

## Features

Language models are probabilistic models that predict the next token in a sequence using the token that comes before it. These models learn the probability of occurrence based on examples from training.
GPT- 2 model has only the decoder part of transformer networks that uses multi headed masked self attention which allows it to look at first n- tokens at a given time.
Instead of processing tokens sequentially, GPT-2 processes it parallely.
The language model such as GPT-2 can easily be fine tuned as per our requirement.

## References:

https://jalammar.github.io/illustrated-gpt2/#part-3-beyond-language-modeling
https://towardsdatascience.com/conditional-text-generation-by-fine-tuning-gpt-2-11c1a9fc639d
https://huggingface.co/docs/transformers/model_doc/gpt2#gpt2lmheadmodel
https://huggingface.co/blog/how-to-generate
