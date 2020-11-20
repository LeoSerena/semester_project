import os
import sys
import pandas as pd
import numpy as np
import re

import nltk

import spacy
from spacy_lefff import LefffLemmatizer, POSTagger


def load(tweets_file, rtt_file):
    """
    Reads the files corresponding to the tweets, the users and the retweets
    """
    try:
        tw_df = pd.read_csv(tweets_file)
        rtt_df = pd.read_csv(rtt_file)
    except:
        print('one or several files were not found')
        sys.exit()

    return tw_df, rtt_df

def clean_headers(df):
    """
    Changes the names of the headers, removing the ' characters and empty spaces
    """
    filtered_headers = [header.replace("'",'').replace(' ', '').replace('(', '').replace(')', '').replace('.', '').replace('[', '').replace(']', '') for header in df.columns]
    map_to_new_headers = {}
    for i in range(len(df.columns)):
        map_to_new_headers[df.columns[i]] = filtered_headers[i]

    return df.rename(columns = map_to_new_headers)

def remove_columns(df):
    """
    Removes all the columns that have no or a single distinct element.
    It also replaces sentiments that are none with the average sentiment.
    """
    avg = np.mean(df[df['sentiment'] != 'None']['sentiment'].astype('float'))
    df['sentiment'] = df['sentiment'].replace('None', avg).astype('float')

    to_remove = []
    print('column(s) removed: ')
    for column in df.columns:
        print(column)
        if(np.unique(df[column][df[column].notnull()]).shape[0] < 2):
            print(column)
            to_remove.append(column)
    
    return df.drop(columns = to_remove)

def join(tw_df, rtt_df):
    """
    Joins the 3 dataframes together such that each row is a tweet, coupled with its poster
    and the list of the ids of users that retweeted as well as when they retweeted it.
    """
    original_tw_id = []
    author_ids = []
    rtt_dates = []
    groups = rtt_df.groupby('original_tweet_id').groups
    for k in groups.keys():
        l_a = []
        l_r = []
        original_tw_id.append(k)
        for index in groups[k]:
            line = rtt_df.iloc[[index]]
            l_a.append(int(line['author_id']))
            l_r.append(str(line['retweet_date']))
        author_ids.append(l_a)
        rtt_dates.append(l_r)
    
    df_temp = pd.DataFrame()
    df_temp['natural_key'] = original_tw_id
    df_temp['rtt_author_ids'] = author_ids
    df_temp['retweet_dates'] = rtt_dates
    df_temp = df_temp.set_index('natural_key')
    tw_df = tw_df.set_index('natural_key')
    return tw_df.join(df_temp)

def filter_df(df):
    """
    Filters out tweets that have no retweets, non FR tweete,
    authors with less than 1000 favs
    """
    filtered_df = df[df['rtt_author_ids'].notnull()]
    filtered_df = filtered_df[filtered_df['user_country'] == 'FR']
    filtered_df = filtered_df[filtered_df['retweet_count'] > 1]
    filtered_df = filtered_df[filtered_df['favourites_count'] > 1000]
    return filtered_df

def rmv_hyperlink(text):
    ## removes all hyperlinks of the text
    return re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)

def lemmatize(serie):
    pos = POSTagger()
    french_lemmatizer = LefffLemmatizer(after_melt = True)
    
    nlp = spacy.load('fr_core_news_sm')
    nlp.add_pipe(pos, name = 'pos', after = 'parser')
    nlp.add_pipe(french_lemmatizer, name = 'lefff', after = 'pos')
    
    lemmatized = serie.map(
        lambda x : [doc.lemma_ for doc in nlp(x)]
    )
    return lemmatized

def rmv_char(text):
    ## removes all unwanted characters of the text
    text = re.sub("""[\n´]""", " ", text)
    text = re.sub("""["'+"*%&/()"=¦@#°§¬|¢\[\]\-\_\—‘’“”•`\^{}~¥©?.,:!$;«»<>]""", "", text)
    return re.sub("""\d""", " ", text)

def clean_tweets(serie):
    serie = serie.map(rmv_hyperlink)
    serie = lemmatize(serie)
    serie = serie.map(lambda x : " ".join(x))
    serie = serie.map(lambda x : x.lower()).map(rmv_char)
    serie = serie.map(nltk.word_tokenize)
    stopwords = nltk.corpus.stopwords.words('french')
    serie = serie.map(lambda tokens : [x for x in tokens if x not in stopwords])
    serie = serie.map(lambda tokens : [x for x in tokens if x.isalpha()])
    return serie

if __name__ == '__main__':

    assert(len(sys.argv) == 3)

    tw_df, rtt_df = load(sys.argv[1], sys.argv[2])

    tw_df = clean_headers(tw_df)
    
    
    # This is hust to manage sometimes messy data with weird column names
    N = len(rtt_df.columns)
    if N > 4:
        l = ['id', 'original_tweet_id', 'author_id', 'retweet_date']
        for k in range(N-4):
            l.append(str(k))
        map_ = {}
        for (i,c) in enumerate(rtt_df.columns):
            map_[c] = l[i]
        rtt_df = rtt_df.rename(columns = map_)
        rtt_df.drop(columns = l[4:])

    
    tw_df = remove_columns(tw_df)

    joined_df = join(tw_df, rtt_df)

    joined_df = filter_df(joined_df)

    joined_df['body'] = clean_tweets(joined_df['body'])

    print('name of the file: ')
    file_name = input()

    joined_df.to_csv(file_name)

