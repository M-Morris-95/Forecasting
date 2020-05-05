import pandas as pd
import os
import tqdm
import numpy as np


for month in months:
    os.chdir('/Users/michael/Documents/geocrawler-tweets/' + month)
    total_tweets = 0
    total_users = 0
    files = os.listdir()
    created = False
    
    for file in tqdm.tqdm(files):
        try:
            csv = pd.read_csv(file, header=None)
            if not created:
                created = True
                user_ids = np.array(csv[0].unique())
            else:
                user_ids = np.append(user_ids, csv[0].unique())

            total_tweets = total_tweets + csv.shape[0]
        except:
            print('failed')
            pass
    print('total tweets in ' + month + " " + str(total_tweets))
    print('total users in ' + month + " " + str(np.unique(user_ids.astype(str)).shape[0]))
