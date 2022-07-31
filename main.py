# Importing necessary libraries
from cProfile import label
from itertools import count
from statistics import mode
import sys, os
import string
import uvicorn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import datetime
import pickle
from pydantic import BaseModel
from fastapi import FastAPI,Request,HTTPException
from fastapi.middleware.cors import CORSMiddleware
import configparser
import tweepy 
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
import re

# read configs
config = configparser.ConfigParser()
config.read('config.ini')
sid_obj= SentimentIntensityAnalyzer()
api_key = config['twitter']['api_key']
print(api_key)
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

#client=tweepy.Client(bearer_token="AAAAAAAAAAAAAAAAAAAAAAf4fAEAAAAA4JmaNHkSvMzjRJxAU2GiZMWdxNc%3DzyO6LOVWvzewvg0D0xX6MTAfYkxxnKb1nisnrf0QogADLlSDEA", consumer_key=api_key,consumer_secret= api_key_secret,access_token=access_token,access_token_secret= access_token_secret, wait_on_rate_limit=True)
# authentication
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
client = tweepy.Client(bearer_token="AAAAAAAAAAAAAAAAAAAAANmGfQEAAAAAvOLvSxGRHXCSyQHKs%2BmQRu3ma1A%3DfGjdh6ytI2xsdUsb1KgMF9ssTd2FUvIyUsWTpL8qRV1KFwVBye",consumer_key= api_key,consumer_secret= api_key_secret,access_token= access_token,access_token_secret= access_token_secret,wait_on_rate_limit=True)
def getOnlyDate(dt):
    return dt.strftime('%Y-%m-%d')

def getPolarity(txt):
    return sid_obj.polarity_scores(txt)['compound']

# Initializing the fast API server
app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading up the trained model
#model = pickle.load(open('../model/hireable.pkl', 'rb'))

# Defining the model input types
class Candidate(BaseModel):
    gender: int
    bsc: float
    workex: int
    etest_p: float
    msc: float

def toTime(ms):
    return datetime.datetime.fromtimestamp(ms / 1000)

def cleanText(txt):
    txt=re.sub(r'@[^\s]*\s+','',txt)
    txt=re.sub(r'RT','',txt)
    txt=re.sub(r'#','',txt)
    txt=re.sub(r'\s+https?:\/\/.*\s*','',txt)
    return txt
def analys(score):
    if score<0:
        return "negative"
    if score>0:
        return "positive"
    if score == 0:
        return "neutral"
# Setting up the home route
@app.get("/prediction/")
def read_root(request: Request):
    try:
        #id = 57741058
        #print(client.get_user(id=id))
        print("hellllo")
        query = 'ebadipour'
        tweets  = tweepy.Paginator(client.search_recent_tweets, query=request.query_params['query'] ,
        tweet_fields=["id", "author_id",  "created_at", "text", "source", "lang", "in_reply_to_user_id", "conversation_id", "public_metrics", "referenced_tweets", "reply_settings"],
        user_fields=["name", "username", "location", "verified", "description", "created_at"],
        place_fields=["full_name", "id", "country", "country_code", "geo", "name", "place_type"],
        expansions="geo.place_id,author_id",max_results=100).flatten(limit=int(request.query_params['num']))
        #for tweet in tweets:
            #print(client.get_user(id=tweet.author_id).data.name)
        
        #public_tweets1=tweepy.Cursor(api.search_tweets, q=request.query_params['query']).items(int(request.query_params['num']))
        #public_tweets1=api.home_timeline()

        columns = ['Time', 'User', 'Tweet']
        data = []
        for tweet in tweets:
            name =client.get_user(id=tweet.author_id).data.name
           
            data.append([tweet.created_at,name , tweet.text])
        print(data) 
        
        df = pd.DataFrame(data, columns=columns)
        print(df)
        df['Tweet']=df['Tweet'].apply(cleanText)
        df['polarity']=df['Tweet'].apply(getPolarity)
        df['analysis']=df['polarity'].apply(analys)
        df['Date']=df['Time'].apply(getOnlyDate)
        qq=df.groupby(['analysis'])['polarity'].count()
        ff = df.groupby(['Time'])['polarity'].mean()
        for_7_day = df.groupby(['Date'])['polarity'].mean()
        gg=pd.DataFrame({'Time':ff.index, 'polarity':ff.values})
        for_send=pd.DataFrame({'Time':for_7_day.index, 'polarity':for_7_day.values})
        pie=pd.DataFrame({'analysis':qq.index, 'count':qq.values})
        print(pie)
        allWords= ' '.join([twts for twts in df['Tweet']])
        size_elem = len(pie.index)
        print(size_elem)
        neu_c=str( pie.iloc[2]['count'])if (size_elem==3) else "0"
        n_c= str(pie.iloc[1]['count']) if (size_elem>=2) else "0"
        p_c=str(pie.iloc[0]['count']) if(size_elem>=1)  else "0"
        counts = dict()
        words = allWords.split()

        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
    
        return {"data": df , "count": counts,"control":gg,"seven":for_send,"positive_count":p_c,"negative_count":n_c,"neutral_count":neu_c}
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        raise HTTPException(status_code=404, detail="GOT A PROBLEM HERE...")






# Setting up the prediction route
@app.post("/prediction/")
async def get_predict():
    print("hello axios")
    
    

    hired = 4

    return {
        "data": {
            'prediction': hired,
            'interpretation': 'Candidate can be hired.' if hired == 1 else 'Candidate can not be hired.'
        }
    }

# Configuring the server host and port
if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')