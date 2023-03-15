# Project Overview

The goal of this project/repository is to identify **Hate Speech** in a tweet using deep learning based Transfomer models. Following models were used to indentify the hate speech in tweets:
* GRU
* LSTM
* Bi-Directional LSTM
* LSTM with attention
* Transfer Learning using **ALBERT transfomer**

# Data Source

You can get the data from https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset.

# API

Using FastAPI,docker and azure I have created a web server that exposes a /predict route. Clients can post their hate speech identification request to the /predict and get the classification results.

**URL to use API** : http://20.237.46.134/docs


