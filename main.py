from fastapi import FastAPI,Request
from pydantic import BaseModel
import uvicorn
import numpy as np
from transformers import AlbertModel,AlbertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import nltk
from nltk.corpus import stopwords


nltk.download("stopwords")

app = FastAPI()

#for validating the input
class HateSpeechPredict(BaseModel):
    text: str

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

    #remove stopwords
    STOPWORDS = stopwords.words("english")
    words = [word for word in text.split() if word not in STOPWORDS]
    text = " ".join(words)

    # remove multiple spaces
    text = re.sub(" +", " ", text)

    return text

tokenizer,model = get_model()

#mapping the results
d = {
  1:'This tweet/text is a hate speech',
  0: 'This tweet/text is not a hate speech'
}


## FASTAPI ROUTES

#first route
@app.get("/")
def get_root():
    return {"message": "Welcome to the Hate Speech Detection API"}

#predict route
@app.post("/predict")
def hate_speech(hp:HateSpeechPredict):

    cleaned_text = clean_tweet(hp.text)

    encoded_input = tokenizer([cleaned_text], padding=True, truncation=True, max_length=512,return_tensors='pt')
    ids,masks = encoded_input["input_ids"],encoded_input["attention_mask"]
    
    # Forward pass w/ inputs
    inputs = ids,masks
    model.eval()
    z = model(inputs)

    # Output probababilites
    y_prob = torch.exp(z).detach().cpu().numpy()[0]
    y_pred = np.argmax(y_prob)  
    response = {"Recieved Text": hp.text,"Prediction": d[y_pred]}
    return response

# if __name__ == "__main__":
#     uvicorn.run("main:app",host='0.0.0.0', port=8080, reload=True, debug=True)