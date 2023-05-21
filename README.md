# Project Overview

The goal of this project/repository is to identify **Hate Speech** in a tweet using deep learning models. Following models were used to indentify the hate speech in tweets:
* GRU
* LSTM
* Bi-Directional LSTM
* LSTM with attention
* Transfer Learning using **ALBERT transfomer**

ALBERT transformer gave the best results with a Precision score of 0.91 on the test data

# Data Source

You can get the data from https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset.

# API

Using FastAPI,docker and azure I have created a web server that exposes a /predict route. Clients can post their hate speech identification request to the /predict and get the classification results.

**URL to use API** : http://20.241.135.69/docs

**Streamlit WebApp**: https://shubham29mehta-hate-speech-detection-app-app-tewndo.streamlit.app/


