import streamlit as st
import numpy as np
from transformers import AlbertModel,AlbertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import re


#Transformer class
class Transformer(nn.Module):
    def __init__(self, transformer, dropout_p, embedding_dim, num_classes):
        super(Transformer, self).__init__()
        self.transformer = transformer
        self.dropout = torch.nn.Dropout(dropout_p)
        self.fc1 = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs):
        ids, masks = inputs
        seq, pool = self.transformer(input_ids=ids, attention_mask=masks,return_dict=False)
        z = self.dropout(pool)
        z = self.fc1(z)
        z = F.log_softmax(z, dim = 1)
        return z


#get model and tokenizer
@st.cache_data
def get_model():
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    transformer = AlbertModel.from_pretrained('albert-base-v2')
    embedding_dim = transformer.config.hidden_size
    
    device = torch.device("cpu")
    checkpoint = "./hate_speech_model.pt"
    model = Transformer(transformer=transformer, dropout_p=0.5,embedding_dim=embedding_dim, num_classes=2)
    model.load_state_dict(torch.load(checkpoint,map_location=device))
    return tokenizer,model

#cleaning the input tweet/text
def clean_tweet(text):

    #lowercase the tweets and remove trailing & ending space
    text = text.lower().strip()                

    # Removes words followed by @
    text = re.sub("(@[A-Za-z0-9]+)", "", text)

    # Removes words at start of string 
    text = re.sub("([^0-9A-Za-z \t])", "", text)

    # remove non alphanumeric chars 
    text = re.sub("[^A-Za-z0-9]+", " ", text)

    # remove multiple spaces
    text = re.sub(" +", " ", text)

    return text

tokenizer,model = get_model()

# front end elements of the web page
st.title('Hate Speech Detection!')

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")


d = {
  1:'This tweet/text is Hate Speech',
  0: 'This tweet/text is not Hate Speech'
}


if user_input and button:

    cleaned_text = clean_tweet(user_input)

    encoded_input = tokenizer([cleaned_text], padding=True, truncation=True, max_length=512,return_tensors='pt')
    ids,masks = encoded_input["input_ids"],encoded_input["attention_mask"]
    
    # Forward pass w/ inputs
    inputs = ids,masks
    model.eval()
    z = model(inputs)

    # Output probababilites
    y_prob = torch.exp(z).detach().cpu().numpy()[0]
    y_pred = np.argmax(y_prob)

    st.write(d[y_pred])
