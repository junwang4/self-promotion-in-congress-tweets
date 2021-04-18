import re, datetime, time, glob, os, sys, json
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

gender_palette_map = {'F':'tab:red', 'M':'tab:blue'}
gender_palette = list(gender_palette_map.values())
folder_output_figure = '/tmp'

folder_data = '../data'
folder_pred = f'../code/working/pred'
folder_big_data = '/Users/jwang72/git/congresstweets' # with data pulled from: https://github.com/alexlitel/congresstweets 

fpath_pred = f'{folder_pred}/20210331_apply_epochs3.csv.v3' # generated in twitter_classifier.py > apply_one_full_model_to_new_sentences()

fpath_members = f'{folder_data}/congress_member_202103.csv' # generated in step_1
fpath_member_followers = f'{folder_data}/bio_id_followers_202103.csv' # generated in step_2

fpath_member_congress_classes = f'{folder_data}/open_data/congress_114_115_116_117.csv'
# downloaded from: https://bioguide.congress.gov/search

fpath_tweet_big = f'{folder_data}/tweets_with_has_i_and_from_known_congress_members.csv' # generated in step_4
#id,screen_name,bio_id,date,retweet_count,favorite_count,text,has_i
# check data_preprocess.py > step_4_filter_tweets_posted_by_known_congress_members__and__enrich_with_hasI_bioId_likes_retweets()

fpath_final_data_for_analysis = f'{folder_data}/final_data_for_regression_analysis.csv' # the ultimate output of this whole script


def save_csv(df, fpath, drop_by=None, float_format='%.3f'):
    if drop_by is not None:
        if type(drop_by)==str:
            print(f'- cnt before drop duplicated "{drop_by}": {len(df)}')
            df = df.drop_duplicates(drop_by)
        elif type(drop_by)==list and len(drop_by)==2:
            print(f'- cnt before drop duplicated "{drop_by[0]}": {len(df)}  (keep = {drop_by[1]})')
            df = df.drop_duplicates(drop_by[0], keep=drop_by[1])
    df.to_csv(fpath, index=False, float_format=float_format)
    print('- saved to:', fpath, '      rows:', len(df))

def df_column_switch(df, column1, column2):
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df = df[i]
    return df

def batch_read_csv_files(glob_pattern, usecols=None, nrows=None, dtype=None):
    if type(usecols)==str: usecols=usecols.split()
    df = None
    #nrows = nrows if nrows is None else nrows
    #for fpath in glob_files:
    for fpath in sorted(glob.glob(glob_pattern)):
        print(fpath)
        tmp = pd.read_csv(fpath, usecols=usecols, nrows=nrows, dtype=dtype, keep_default_na=False)
        df = tmp if df is None else pd.concat((df, tmp), sort=False)
    print(f'\n- total count of batch csv read: {len(df):,}\n')
    return df

def read_congress_member():
    df_member = pd.read_csv(fpath_members)
    print('cnt of members archived by https://github.com/alexlitel/congresstweets:', len(df_member))
    def get_age(x): return int( (datetime.datetime.now()-datetime.datetime.strptime(x, '%Y-%m-%d')).days/365.25)
    df_member['age'] = df_member['birthday'].apply(get_age)
    df_member_followers = pd.read_csv(fpath_member_followers)
    df_member = df_member.merge(df_member_followers)
    print('cnt of members who have followers:', len(df_member)) 
    print('    for those without followers, their accounts might have been suspended/deleted when I hydrated their tweets\n')

    df_member['followers_log'] = df_member['followers'].apply(lambda x: np.log(x+1))
    return df_member.drop(['twitter_accounts_json'], axis=1)

def read_congress_classes(class_recent=117):
    df = pd.read_csv(fpath_member_congress_classes, usecols='id congresses'.split())
    df['classes'] = df.congresses.apply(lambda x: ' '.join(map(str, sorted([cls['congressNumber'] for cls in json.loads(x) if cls['congressNumber']<=class_recent]))))
    df['num_terms'] = df.classes.apply(lambda x: len(x.split()))
    df.rename(columns={'id':'bio_id'}, inplace=True)
    print(f'- members up to class of {class_recent} is: {len(df)}\n')
    for cls in ('114', '115', '116', '117'):
        print(f'Total number of class {cls}: {len(df[df.classes.str.find(cls)>=0])}')
    for cls in ('115', '116', '117'):
        cls_prev = str(int(cls)-1)
        df[f'{cls}new'] = df.classes.apply(lambda x: int(x.startswith(cls)))
        df[f'{cls}in'] = df.classes.apply(lambda x: int(x.find(cls_prev)<0 and x.find(cls)>=0) )
        df[f'{cls}out'] = df.classes.apply(lambda x: int(x.find(cls_prev)>=0 and x.find(cls)<0))

        for cat in ('new', 'in', 'out'):
            attr = f"{cls}{cat}"
            print(f" - {attr:8s}: {len(df[df[attr]==1]):3d}")
    return df.drop('congresses', axis=1)

def read_tweets(first_date, last_date, tweets_TH = 10):
        #id,screen_name,bio_id,date,retweet_count,favorite_count,text,has_RT,has_i
        usecols = 'id bio_id date retweet_count favorite_count has_i'.split()
        df_tweet = pd.read_csv(fpath_tweet_big, dtype={'id':str}, usecols=usecols)
        df_tweet['date'] = df_tweet['date'].apply(lambda x:x[:10])
        print('- total tweets:', len(df_tweet))
        if first_date is not None and last_date is not None:
            df_tweet = df_tweet[(df_tweet.date>=first_date) & (df_tweet.date<=last_date)]
            print('- within the date range:', len(df_tweet))

        dg_bid_tweets = df_tweet.groupby('bio_id').agg(tweets=('id', 'count')).reset_index()
        print('- Congresspeople in tweets:', len(dg_bid_tweets))
        print(f'- Those with at least {tweets_TH} tweets:', len(dg_bid_tweets[dg_bid_tweets.tweets>=tweets_TH]))

        #bio_id_with_TH_tweets = set(dg_bid_tweets[dg_bid_tweets.tweets>=0].bio_id.tolist())
        #df_tweet = df_tweet[df_tweet.bio_id.isin(bio_id_with_TH_tweets)]
        print(f'- tweets after restricting to those people with {tweets_TH} tweets:', len(df_tweet))
        return df_tweet# bio_id_with_TH_tweets

def read_prediction():
        #usecols = 'id winner confidence'.split()
        usecols = 'id winner'.split()
        df_pred = pd.read_csv(fpath_pred, dtype={'id':str}, usecols=usecols)
        print('\n- cnt of predictions:', len(df_pred))
        # is_sp: is predicted as self-promotion (sp: self-promotion)
        df_pred['is_sp'] = df_pred.winner.apply(lambda x: int(x=='c1')) # c1: YES, c0: NO
        return df_pred[['id', 'is_sp']]

def generate_final_data_for_regression_analysis(first_date='2017-07-01', last_date='2021-03-31', debug=True, output=False):
    print(f'\n- date range: from {first_date} to {last_date}\n')

    df_member = read_congress_member()

    df_bioId_classes = read_congress_classes(class_recent=117)
    df_member = df_member.merge(df_bioId_classes)

    print(df_member.groupby(['gender']).size())
    print(df_member.groupby(['gender', 'chamber', 'party']).size())
    print(df_member.iloc[0])

    df_tweet = read_tweets(first_date, last_date)

    df_pred = read_prediction()
    df_pred = df_pred.merge(df_tweet[['id']])

    # merge data
    df = df_pred.merge(df_tweet)
    df = df.merge(df_member)
    print(len(df))

    print(df.iloc[0])

    if 0:
        df['newly_elected'] = df.apply(lambda x: '_117' if x['117new']==1 \
                                    else '_116' if x['116new']==1 \
                                    else '_115' if x['115new']==1 \
                                    else '_114b' , axis=1)
        print(df.drop_duplicates('bio_id')['newly_elected'].value_counts())

    df = df[df.party.isin(('R','D'))]
    print(f'\n- cnt of tweets after limiting to party of R/D: {len(df)}\n')

    '''
        id,is_sp,bio_id,date,retweet_count,favorite_count,has_i,chamber,party,gender,age,followers,classes,num_terms,
        115new,115in,115out,116new,116in,116out,117new,117in,117out,ym
    '''

    df.rename(columns={'is_sp': 'self_promotion_as_predicted_by_BERT_model'}, inplace=True)
    cols = 'self_promotion_as_predicted_by_BERT_model,bio_id,date,chamber,party,gender,age,followers_log,num_terms'.split(',')
    df_out = df[cols]
    print(df_out.head())

    if 0:
        save_csv(df_out, fpath_final_data_for_analysis)



def generate_sample_id_text_csv_file__and__output_all_tweet_ids():
    df_member = read_congress_member()
    print(len(df_member))
    df_member = df_member[df_member.party.isin(('R','D'))]
    print('within party of R or D:', len(df_member))


    bio_ids_valid = set(df_member.bio_id.tolist())

    df_pred = pd.read_csv(fpath_pred, usecols=['id'])

    df = pd.read_csv(fpath_tweet_big, usecols='id bio_id text has_RT date'.split())#, nrows=9999)
    df = df[~df.has_RT].drop(['has_RT'], axis=1)
    print(len(df))
    df = df.merge(df_pred)
    print('- after merge with df_pred:', len(df))

    df = df[df.bio_id.isin(bio_ids_valid)]
    print('- after limiting to valid members (party: R/D and with active twitter accounts):', len(df))
    date_start = '2017-07-01'
    df = df[df.date>=date_start]
    print(f'- after limiting to >= {date_start}:', len(df))
    date_end = '2021-04-01'
    df = df[df.date<date_end]
    print(f'- after limiting to < {date_end}:', len(df))

    df = df.drop(['date'], axis=1)
    df = df.rename(columns={'id':'tweet_id'})
    df_sample = df.sample(n=1000)

    if 0:
        folder_out = '../data'
        df[['tweet_id']].to_csv(f'{folder_out}/tweet_ids_all.csv', index=False, header=False)
        df_sample.to_csv(f'{folder_out}/sample_tweetid_bioid_text.csv', index=False)



def plot_trend(data=None, y=None, y_label='', x='ym', x_label='', hue='gender', style=None, style_order=None, marker=None, markersize=None,
               x_rotation=90, figsize=(7, 3.0), ylim=None,
               legend_loc='upper left', xticks_fs=10, figname='', estimator=np.mean):
    dgg = data
    if data is None or y is None:
        print('- wrong data or y parameters')
        return
    plt.figure(figsize=figsize)
    if style is None:
        style='gender'
        style_order=['M', 'F']
    ax = sns.lineplot(x=x, y=y, hue='gender', hue_order=gender_palette_map.keys(), palette=gender_palette,
        data=dgg, marker=marker, markersize=markersize, style=style, style_order=style_order, estimator=estimator)

    if ylim is not None:
        plt.ylim(ylim)

    for special_yy in ('2017', '2018', '2019', '2020'):
        ax.axvline(x=f'{special_yy}-12', color='tab:green', linewidth=1)
    #plt.xticks(rotation=x_rotation, fontsize=xticks_fs)
    ticks, labels = [], []
    mm_ch = {1:'J', 2:'F', 3:'M', 4:'A', 5:'M', 6:'J', 7:'J', 8:'A', 9:'S', 10:'O', 11:'N', 12:'D'}
    for year in range(2017, 2022):
        plt.text(f'{year}-0{7 if year>2017 else 9}', ax.get_ylim()[0], f'{year}    ', horizontalalignment='center')
        for mm in range(1 if year>2017 else 7, 13 if year<2021 else 4, 1):
            ticks.append(f'{year}-{mm:02d}')
            labels.append(f'{mm}')
            #labels.append(f'{mm_ch[mm]}')
    plt.xticks(ticks=ticks, labels=labels, fontsize=10, fontname="Times New Roman")

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:], loc=legend_loc)
    save_fig(figname)


def save_fig(figname, fig=None):
    if figname:
        outfig = f'{figname}'
        if outfig.find('/')<0:
            outfig = f'{folder_output_figure}/{outfig}'
        plt.tight_layout()
        if fig is not None:
            fig.savefig(outfig)
        else:
            plt.savefig(outfig)
        print('\n- figure saved to:', outfig, '\n')


def plot_trend_of_gender_difference():
    sns.set()
    df = pd.read_csv(fpath_final_data_for_analysis)
    df['ym'] = df['date'].apply(lambda x: x[:7])
    dg = df.groupby(['ym', 'gender', 'bio_id']).agg(monthly_cnt = ('bio_id', 'count')).reset_index()
    print(dg.shape)
    print('members:', len(dg.bio_id.unique()))

    dg_gender = dg.groupby(['ym', 'gender']).agg(monthly_cnt_median = ('monthly_cnt', 'median')).reset_index()
    
    print(dg_gender.groupby('gender').mean())

    figname = 'trend_gender_tweets.pdf'
    plot_trend(dg_gender, 'monthly_cnt_median', y_label='', legend_loc='upper left', figname=figname)


def test_tweepy_api():
    api = setup_tweepy_api()
    results = api.search_users('Justin Amash')
    print(results[0])

def setup_tweepy_api():
    import tweepy
    from configparser import ConfigParser
    config = ConfigParser()
    fpath_config = os.path.expanduser('~/.twarc')
    if not os.path.exists(fpath_config):
        print(f'- error: file "{fpath_config}" not exists.\n')
        print('- You can create the file manually in the following format (you need to sign up as a Twitter developer to get the following information):\n')
        print('''
[default]
consumer_key = your_consumer_key
consumer_secret = your_consumer_secret
access_token = your_access_token
access_token_secret = your_access_secret\n''')
        sys.exit()

    config.read(fpath_config)
    my_twitter_account = 'default'
    consumer_key = config.get(my_twitter_account, 'consumer_key')
    consumer_secret = config.get(my_twitter_account, 'consumer_secret')
    access_token = config.get(my_twitter_account, 'access_token')
    access_token_secret = config.get(my_twitter_account, 'access_token_secret')

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api


def step_1_create_congress_member_csv__by_merge__historical_users_filtered_json__with__gender_birth_data__via_bioguide_id():
    # https://theunitedstates.io/congress-legislators/legislators-current.csv
    # https://theunitedstates.io/congress-legislators/legislators-historical.csv

    fpath_alexlitel_members = f'{folder_data}/open_data/historical-users-filtered.json-2021'
    # https://github.com/alexlitel/congresstweets-automator/blob/master/data/historical-users-filtered.json

    out = []
    for data in json.load(open(fpath_alexlitel_members)):
        type = data['type']
        if not type == 'member':
            continue
        name = data['name']
        bioguide_id = data['id']['bioguide']
        chamber = data['chamber']
        party = data['party']
        tw_accounts = json.dumps(data['accounts'])
        out.append({'bioguide_id':bioguide_id, 'name':name, 'chamber':chamber, 'party':party, 'twitter_accounts_json':tw_accounts})
    df_alex = pd.DataFrame(out)
    print(df_alex.head())

    glob_pattern = f'{folder_data}/open_data/legislators-*.csv-2021'
    # downloaded from https://github.com/unitedstates/congress-legislators
    usecols = 'birthday,gender,bioguide_id'.split(',')
    nrows = None
    df_gender_birth = batch_read_csv_files(glob_pattern, usecols=usecols, nrows=nrows, dtype={'id':str})
    print(df_gender_birth.head())

    df = df_alex.merge(df_gender_birth)
    df.rename(columns={'bioguide_id':'bio_id'}, inplace=True)
    fpath_out = f'{folder_data}/congress_member.csv-tmp'
    save_csv(df, fpath_out)


def step_3a_get_tweet_id_only__from_alex_github():
    return

    ids_of_last_time = []
    for fpath_out_last_time in glob.glob(f'{folder_big_data}/congress_tweet_ids.dat-*'):
        if fpath_out_last_time.find(helpers.hydrate_version_curr_time) >= 0:
            continue
        ids_of_last_time += [e.strip() for e in open(fpath_out_last_time).readlines()]
    ids_of_last_time = set(ids_of_last_time)

    fpath_out = f'{folder_big_data}/congress_tweet_ids.dat-{helpers.hydrate_version_curr_time}'
    if os.path.exists(fpath_out):
        return fpath_out

    ids = []
    for fpath in sorted(glob.glob(f'{folder_big_data}/data/*.json')):
        print(fpath)
        if os.path.getsize(fpath) > 100:
            data = json.load(open(fpath))
            for d in data:
                if 'id' in d:
                    if str(d['id']) not in ids_of_last_time:
                        ids.append(str(d['id']))
            #break
    with open(fpath_out, 'w') as fout:
        fout.write('\n'.join(ids))
        print('- out:', fpath_out)


def step_3b_hydrate_to_get_comprehensive_information_of_the_tweets():
    tweet_ids_file = f'{folder_big_data}/congress_tweet_ids.dat-{helpers.hydrate_version_curr_time}'
    #fpath_out = f"{folder_big_data}/congress_tweets_with_full_information-{helpers.get_current_date_str()}.json"
    fpath_out = f"{folder_big_data}/congress_tweets_with_full_information-{helpers.hydrate_version_curr_time}.json"
    cmd = f"twarc hydrate {tweet_ids_file} > {fpath_out}"
    print(cmd)
    os.system(cmd)


def step_3c_extract_likes_and_retweets_from_the_above_hydrated_tweets():
    #print('check file under: microsoft_scisip/code/coling2020/newskit/tweets_utils.py')
    #folder = '/home/byu/tweets/kelly_congress_tweets'
    #from tqdm import tqdm

    fpath_huge = f'{folder_big_data}/congress_tweets_with_full_information-{helpers.hydrate_version_curr_time}.json'
    print(fpath_huge)

    out = []
    out_user = []
    for line in open(fpath_huge):
        data = json.loads(line)
        id = data['id']
        retweet_count = data['retweet_count']
        favorite_count = data['favorite_count']
        text = data['full_text']
        created_at = data['created_at']
        created_at = datetime.datetime.strftime(datetime.datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d %H:%M:%S')
        replied_to = data['in_reply_to_screen_name']
        is_quote_status = data['is_quote_status']
        user_followers_count = data['user']['followers_count'] # this is the #followers at the time of tweets retrieval
        user_screen_name = data['user']['screen_name']
        user_name = data['user']['name']
        user_id = data['user']['id']
        out.append({'id':id,
            'screen_name':user_screen_name,
            'user_id':user_id,
            'created_at':created_at,
            'replied_to':replied_to,
            'is_quote_status':is_quote_status,
            'retweet_count':retweet_count,
            'favorite_count':favorite_count,
            'text':text
        })
        out_user.append({'screen_name':user_screen_name, 'user_id':user_id, 'followers_count':user_followers_count, 'name':user_name})
        #if len(out)>100000:break

    df_out = pd.DataFrame(out)
    print('- tweets:', df_out.shape)
    df_user = pd.DataFrame(out_user)
    print('- users:', df_user.shape)
    df_user = df_user.drop_duplicates()
    print('- unique combination of screen_name/twitter_name/followers_cnt:', df_user.shape)

    fpath_out = f'{folder_big_data}/tweets_all_{helpers.hydrate_version_curr_time}.csv'
    fpath_out_user = f'{folder_big_data}/users_all_{helpers.hydrate_version_curr_time}.csv'
    df_out.to_csv(fpath_out, index=False)
    df_user.to_csv(fpath_out_user, index=False)
    print('out:', fpath_out)
    print('out:', fpath_out_user)


def read_tweet_user_id_and_followers():
    glob_pattern = f'{folder_big_data}/users_all_20*.csv'
    print(glob_pattern)
    usecols = 'user_id,followers_count'.split(',')
    df_user_followers = helpers.batch_read_csv_files(glob_pattern, usecols=usecols, dtype={'user_id':str, 'followers_count':int})
    # keep recent followers_count which is assumed to be bigger than the one previously retrieved
    df_user_followers = df_user_followers.sort_values('followers_count').drop_duplicates('user_id', keep='last')
    uid_followers = {uid:followers for uid, followers in zip(df_user_followers.user_id, df_user_followers.followers_count)}
    print('- unique user ids:', len(uid_followers))
    return uid_followers, df_user_followers

def read_tweet_user_id_and_congress_bio_id():
    #df_demographics = pd.read_csv(f'{folder_congress}/congress_member_full_demographics.csv-2021')
    df_demographics = pd.read_csv(fpath_members)

    # bioguide_id,name,chamber,party,twitter_accounts_json,birthday,gender
    # twitter_accounts_json: [{""id"": ""37007274"", ""screen_name"": ""repdonyoung"", ""account_type"": ""office""}, ...]
    user_id_bio_id = {}
    for bid, twitter_accounts_json in zip(df_demographics.bio_id, df_demographics.twitter_accounts_json):
        for acnt in json.loads(twitter_accounts_json):
            user_id = str(acnt['id'])
            user_id_bio_id[user_id] = bid

    df_userId_bioId = pd.DataFrame(user_id_bio_id.items(), columns=['user_id', 'bio_id'])
    return user_id_bio_id, df_userId_bioId

def step_3d_export_bio_id__with__followers():
    user_id_followers, df_userId_followers = read_tweet_user_id_and_followers()
    user_id_bio_id, df_userId_bioId = read_tweet_user_id_and_congress_bio_id()
    df = df_userId_bioId.merge(df_userId_followers)
    dg = df.groupby('bio_id').agg(followers = ('followers_count', 'sum')).reset_index()
    print(dg.head())
    #save_csv(dg, fpath_member_followers)


def step_2_fetch_followers_for_each_bio_id_via_tweepy():
    # TODO: should only query those whose previous followers == 0 (due to exceptions)

    api = setup_tweepy_api()
    #result = api.get_user(37007274)
    #followers_count = result.followers_count
    #print(followers_count)
    fpath_members = f'{folder_data}/congress_member_202103.csv'
    df = pd.read_csv(fpath_members)
    out = []
    #query_mode = 'id'
    query_mode = 'screen_name'
    for bio_id, twitter_accounts_json in zip(df.bio_id, df.twitter_accounts_json):
        followers_count = 0
        for twitter_account in json.loads(twitter_accounts_json):
            account_id = twitter_account['id']
            screen_name = twitter_account['screen_name'] 
            if query_mode == 'id':
                try:
                    result = api.get_user(account_id)
                    followers_count_ = result.followers_count
                    followers_count += followers_count_
                except:
                    print(f'- user id [{account_id}] not found in tweepy for bio_id: {bio_id}')
            elif query_mode == 'screen_name':
                try:
                    result = api.get_user(screen_name)
                    followers_count_ = result.followers_count
                    followers_count += followers_count_
                except:
                    print(f'- screen_name [{screen_name}] not found in tweepy for bio_id: {bio_id}')

        out.append({'bio_id':bio_id, 'followers':followers_count})
        time.sleep(2)
        #if len(out)>2: break

    fpath_member_followers_tmp = f'{folder_data}/bio_id_followers.csv-tmp_by_screen_name' 
    df_out = pd.DataFrame(out)
    save_csv(df_out, fpath_member_followers_tmp)


# NOTE: this is used in generating feature "has_i"
def create_feature_of_having_specific_word_in_text(df, word='i', start_with_word=False):
    cols = df.columns.tolist()
    patterns_begin = {
        'we': re.compile(r'We\b'),
        'our': re.compile(r'Our\b'),
        'i': re.compile(r'I\b'),
        'my': re.compile(r'My\b')
    }
    patterns_clause = {
        'we': re.compile(r',\s*we\b'),
        'our': re.compile(r',\s*our\b'),
        'i': re.compile(r',\s*I\b'),
        'my': re.compile(r',\s*my\b')
    }
    patterns_genderal = {
        'we': re.compile(r'\bwe\b', flags=re.IGNORECASE),
        'our': re.compile(r'\bour\b', flags=re.IGNORECASE),
        'i': re.compile(r'\bi\b', flags=re.IGNORECASE),
        'my': re.compile(r'\bmy\b', flags=re.IGNORECASE),
    }

    def get_first_sentence_with_we_our_i_my(ss, w):
        for s in ss:
            s1 = s.strip()
            if patterns_begin[w].match(s1) or patterns_clause[w].search(s1):
                return s
        return ''

    feature_has = f'has_{word}'
    df[feature_has] = df.text.apply(lambda x: int(bool(patterns_genderal[word].search(x))))
    if start_with_word:
        df['sentences'] = df.text.apply(lambda x: sent_detector.tokenize(x.strip()))
        feature_sentence = f'sentence_{word}'
        df[feature_sentence] = df.sentences.apply(lambda x: get_first_sentence_with_we_our_i_my(x, w))
        df[feature_has] = df[feature_sentence].apply(lambda x: int(not x==''))
    print()
    print(df[feature_has].value_counts())
    print(f'\n- percentage of has_{word}: {100 * len(df[df[feature_has]==1]) / len(df): .1f}%\n')
    df = df[cols+[feature_has]]
    return df

def _read_those_tweets_posted_by_known_congress_members():
    user_id_bio_id, df_userId_bioId = read_tweet_user_id_and_congress_bio_id()

    nrows = 1000
    nrows = None

    glob_pattern = f'{folder_big_data}/tweets_all_20*.csv'
    usecols = 'id,screen_name,user_id,created_at,retweet_count,favorite_count,text'.split(',')
    df = batch_read_csv_files(glob_pattern, usecols=usecols, nrows=nrows,
                    dtype={'id':str, 'user_id':str, 'retweet_count':str, 'favorite_count':str})
    df.rename(columns={'created_at':'date'}, inplace=True)

    print('- cnt of all tweets:', len(df))
    df_out = df[df.user_id.isin(user_id_bio_id.keys())].copy()
    df_out['bio_id'] = df_out.user_id.apply(lambda x: user_id_bio_id[x])
    print('- cnt of all tweets from accounts with known gender/chamber/born/party:', len(df_out))
    print(f'- percentage left: {100*len(df_out)/len(df):.1f}%\n')
    df_out = df_out.dropna()
    print('- after dropna:', df_out.shape)
    return df_column_switch(df_out, 'user_id', 'bio_id').drop('user_id', axis=1)

def step_4_filter_tweets_posted_by_known_congress_members__and__enrich_with_hasI_bioId_likes_retweets():
    df0 = _read_those_tweets_posted_by_known_congress_members()

    df0['has_RT'] = df0.text.apply(lambda x: x.startswith('RT '))
    df = df0[~df0.has_RT].copy()
    print(f'- after drop RT: {len(df)} (from {len(df0)})   percentage: {100*len(df)/len(df0):.1f}%')
    df = df.drop('has_RT', axis=1)
    print(df.iloc[0])

    #return
    df = create_feature_of_having_specific_word_in_text(df, word='i')
    fpath_out = f'{folder_data}/tweets_with_has_i_and_from_known_congress_members.csv-tmp'
    save_csv(df, fpath_out, drop_by='id')



def main_preprocess():
    pass
    #test_tweepy_api()

    # output: data/congress_members.csv 
    #   // "bio_id,name,chamber,party,twitter_accounts_json"
    #step_1_create_congress_member_csv__by_merge__historical_users_filtered_json__with__gender_birth_data__via_bioguide_id()

    # DONE
    #   read "bio_id,twitter_accounts_json" from "congress_member.csv" (the result of step_1_...)
    #   for each "screen_name/id" in "twitter_accounts_json", call tweepy api to get the followers of the bio_id 
    #step_2_fetch_followers_for_each_bio_id_via_tweepy()


    # TODO
    # the following 3 sub_steps (2a, 2b, 2c) need to revise 
    #   instead of using "twarc", we could use tweepy api for easy to update/fetch/parse data
    #   // the code are put here for reference

    #step_3a_get_tweet_id_only__from_alex_github()
    #step_3b_hydrate_to_get_comprehensive_information_of_the_tweets()
    #step_3c_extract_likes_and_retweets_from_the_above_hydrated_tweets()
    # Outdated_step_3d_export_bio_id__with__followers()


    # NOTE: this one needs to read data created from step_3
    #step_4_filter_tweets_posted_by_known_congress_members__and__enrich_with_hasI_bioId_likes_retweets()



def main():
    pass
    #generate_sample_id_text_csv_file__and__output_all_tweet_ids()
    #generate_final_data_for_regression_analysis(first_date='2017-07-01', last_date='2021-03-31', debug=True, output=False)

    #plot_trend_of_gender_difference()


if __name__ == '__main__':
    tic = time.time()
    main_preprocess()
    #main()
    print(f'\n- time used: {time.time()-tic:.1f} seconds\n')
