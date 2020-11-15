import os
import sys
import pandas as pd
import numpy as np

## Constants specific to given .csv



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

if __name__ == '__main__':

    assert(len(sys.argv) == 3)

    tw_df, rtt_df = load(sys.argv[1], sys.argv[2])

    tw_df = clean_headers(tw_df)
    
    l = ['id', 'original_tweet_id', 'author_id', 'retweet_date', 'a','b','c','d']
    map_ = {}
    for (i,c) in enumerate(rtt_df.columns):
        map_[c] = l[i]
    rtt_df = rtt_df.rename(columns = map_)
    rtt_df.drop(columns = l[4:])

    
    tw_df = remove_columns(tw_df)

    joined_df = join(tw_df, rtt_df)

    joined_df = filter_df(joined_df)

    print('name of the file: ')
    file_name = input()

    joined_df.to_csv(file_name)

