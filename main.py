# Importing necessary libraries
from itertools import count
from statistics import mode
import string
import uvicorn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import twint
import datetime
import pickle
from pydantic import BaseModel
from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
import configparser
import tweepy 
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
import re


sid_obj= SentimentIntensityAnalyzer()

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

@app.get("/")
def hello():
    return {"message":"Hello TutLinks.com"}    
# Setting up the home route
@app.get("/prediction/")
def read_root(request: Request):
    
    #public_tweets1 = api.search_full_archive(
       # label='FullArchive',
             # query="#"+request.query_params['query'],
             # )
   # public_tweets2 = tweepy.Paginator(client.search_recent_tweets, query="#"+request.query_params['query']+" -is:retweet ",
                               #tweet_fields=["id", "author_id",  "created_at", "text", "source", "lang", "in_reply_to_user_id", "conversation_id", "public_metrics", "referenced_tweets", "reply_settings"],
        #user_fields=["name", "username", "location", "verified", "description", "created_at"],
        #place_fields=["full_name", "id", "country", "country_code", "geo", "name", "place_type"],
        #expansions="geo.place_id,author_id", max_results=100).flatten(limit=50)
    x=0
    
    # create dataframe
    c = twint.Config()
    c.Limit = request.query_params['num']
    c.Search=request.query_params['query']
    c.Pandas = True

    twint.run.Search(c)

    df1 = twint.storage.panda.Tweets_df
    columns = ['Time', 'User', 'Tweet']
    #data = []
    #for tweet in public_tweets:
     #data.append([tweet.created_at, tweet.user.screen_name, tweet.text])
    #print(len(data))

    df = pd.DataFrame()
    df['Time1']=df1['created_at']
    df['User']=df1['username']
    df['Tweet']=df1['tweet']
    
    df['Tweet']=df['Tweet'].apply(cleanText)
    
    df['polarity']=df['Tweet'].apply(getPolarity)
    df['analysis']=df['polarity'].apply(analys)
    df['Time']=df['Time1'].apply(toTime)
    df['Date']=df['Time'].apply(getOnlyDate)
    qq=df.groupby(['analysis'])['polarity'].count()
    ff = df.groupby(['Time'])['polarity'].mean()
    for_7_day = df.groupby(['Date'])['polarity'].mean()
    gg=pd.DataFrame({'Time':ff.index, 'polarity':ff.values})
    for_send=pd.DataFrame({'Time':for_7_day.index, 'polarity':for_7_day.values})
    pie=pd.DataFrame({'analysis':qq.index, 'count':qq.values})
    print(pie)
    
    
    allWords= ' '.join([twts for twts in df['Tweet']])
    p_c =str( pie.iloc[2]['count'])
    n_c=str(pie.iloc[0]['count'])
    neu_c=str(pie.iloc[1]['count'])
    
    counts = dict()
    words = allWords.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    
    return {"data": df , "count": counts,"control":gg,"seven":for_send,"positive_count":p_c,"negative_count":n_c,"neutral_count":neu_c}

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
