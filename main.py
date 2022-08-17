# Importing necessary libraries
from cProfile import label
from itertools import count
from statistics import mode
import sys, os
import string
import uvicorn
from PIL import Image,ImageDraw,ImageFont
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fastapi.responses import FileResponse
import datetime
import pickle
from wordcloud import WordCloud, STOPWORDS
import imageio
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
#client = tweepy.Client(bearer_token="AAAAAAAAAAAAAAAAAAAAAMEGfgEAAAAAJkYph5vmC2ZoOqv38jqwpmyEaSs%3Ddn0f6DXENk7ud9n7KqKjyC191ajrtBmgSgy21LfWvcNoSK06O8",consumer_key= api_key,consumer_secret= api_key_secret,access_token= access_token,access_token_secret= access_token_secret,wait_on_rate_limit=True)
client = tweepy.Client(bearer_token="AAAAAAAAAAAAAAAAAAAAAMEGfgEAAAAAP707CqP6NrIWaLWxuaYNlAJ4OjA%3DUVOxA6YlGuJ2b2Xp4iCpC6GewAfIGAchPgRa2G61rZ4baMWlfU",consumer_key= api_key,consumer_secret= api_key_secret,access_token= access_token,access_token_secret= access_token_secret,wait_on_rate_limit=True)
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

def only_positive(txt):
    if(txt=="positive"):
        return 1.0
    return 0.0
def only_negative(txt):
    if(txt=="negative"):
        return 1.0
    return 0.0
def only_neutral(txt):
    if(txt=="neutral"):
        return 1.0
    return 0.0
def get_Date_hour(time):
    st=time.strftime('%d %B,%Y _ Hour %H:%M:%S')
    print(st)
    for_return=st.split(":")[0]
    return for_return

def toTime(ms):
    return datetime.datetime.fromtimestamp(ms / 1000)

def cleanText(txt):
    txt=re.sub(r'@[^\s]*\s+','',txt)
    txt=re.sub(r'RT','',txt)
    txt=re.sub(r'#','',txt)
    txt=re.sub(r'\s+https?:\/\/.*\s*','',txt)
    return txt
def analys(score):
    if score<=-0.05:
        return "negative"
    elif score>=0.05:
        return "positive"
    else:
        return "neutral"

@app.get("/image")
async def main(request: Request):
    print(request.query_params['query'])
    return FileResponse("my_twitter_wordcloud_1.png")



# Setting up the home route
@app.get("/prediction/")
async def read_root(request: Request):
    try:
        
        
        tweets  = client.search_recent_tweets(query=request.query_params['query']+' '+"lang:en" ,
        media_fields=['preview_image_url', 'url'],
        tweet_fields=["id","context_annotations", "author_id",  "created_at", "text", "source", "lang", "in_reply_to_user_id", "conversation_id", "public_metrics", "referenced_tweets", "reply_settings","geo"],
        user_fields=["name", "username", "location", "verified", "description", "created_at","profile_image_url"],
        place_fields=["full_name", "id", "country", "country_code", "geo", "name", "place_type"],
        expansions=["geo.place_id,author_id","entities.mentions.username","attachments.media_keys"],max_results=100)
        #for tweet in tweets:
            #print(client.get_user(id=tweet.author_id).data.name)
        
        #public_tweets1=tweepy.Cursor(api.search_tweets, q=request.query_params['query']).items(int(request.query_params['num']))
        #public_tweets1=api.home_timeline()
        #print(tweets.data)
        columns = ['Time', 'User', 'Tweet','img_url','tweet_url']
        data = []
        users={}
        c=0
        
        users.update({u["id"]: u for u in tweets.includes['users']})
        

        
        
        x=0
        for tweet in tweets.data:
            x=x+1
            
            user = users[tweet.author_id]
            ur = 'https://twitter.com/'+user.username+'/status/'+str(tweet.id)
            
            if('full_text' in tweet):
                
                data.append([tweet.created_at,user.username, tweet.full_text,user.profile_image_url,ur])
            else:
                data.append([tweet.created_at,user.username, tweet.text,user.profile_image_url,ur])

            
            
        
        df = pd.DataFrame(data, columns=columns)
        
        df=df.head(int(request.query_params['num']))
        
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
        df['Hour_Date']=df['Time'].apply(get_Date_hour)
        df['positive_count']=df['analysis'].apply(only_positive)
        df['negative_count']=df['analysis'].apply(only_negative)
        df['neutral_count']=df['analysis'].apply(only_neutral)
        for_daily_hour_positive=df.groupby(['Hour_Date'])['positive_count'].sum()
        for_daily_hour_negative=df.groupby(['Hour_Date'])['negative_count'].sum()
        for_daily_hour_neutral=df.groupby(['Hour_Date'])['neutral_count'].sum()
        mile_positive=pd.DataFrame({'Time':for_daily_hour_positive.index, 'count':for_daily_hour_positive.values})
        mile_negative=pd.DataFrame({'Time':for_daily_hour_negative.index, 'count':for_daily_hour_negative.values})
        mile_neutral=pd.DataFrame({'Time':for_daily_hour_neutral.index, 'count':for_daily_hour_neutral.values})
        pie=pie.set_index('analysis')
        
        allWords= ' '.join([twts for twts in df['Tweet']])

        no_urls_no_tags = " ".join([word for word in allWords.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and word != 'RT'
                            ])

        twitter_mask = imageio.imread('https://raw.githubusercontent.com/rasbt/datacollect/master/dataviz/twitter_cloud/twitter_mask.png')
        wordcloud = WordCloud(
                      font_path='cabin-sketch-v1.02/CabinSketch-Bold.ttf',
                      stopwords=STOPWORDS,
                      background_color='white',
                      width=1800,
                      height=1400,
                      mask=twitter_mask
            ).generate(no_urls_no_tags)
        plt.imshow(wordcloud)
        plt.axis("off")
        wordcloud.to_file('my_twitter_wordcloud_1.png')
        
        
        size_elem = len(pie.index)
        
        p_c=str( pie.loc['positive']['count'])if ('positive' in pie.index) else "0"
        n_c= str(pie.loc['negative']['count']) if ('negative' in pie.index) else "0"
        neu_c=str(pie.loc['neutral']['count']) if('neutral' in pie.index)  else "0"
        counts = dict()
        words = allWords.split()

        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
    
        return {"data": df , "count": counts,"mile_positive":mile_positive,"mile_negative":mile_negative,"mile_neutral":mile_neutral,"control":gg,"seven":for_send,"positive_count":p_c,"negative_count":n_c,"neutral_count":neu_c}
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print( repr(e))
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