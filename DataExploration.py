# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:31:15 2022

@author: d_all
"""
import collections
import glob
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.signal
import seaborn as sns


data_dir = 'data/'
filenames = [os.path.splitext(f)[0] for f in sorted(os.listdir(data_dir))]
fs = 256

print(filenames) 

subjects_per_class = collections.defaultdict(set)
data = []
df_data = []
for f in filenames:
    subject = f.split('_')[0]
    activity = f.split('_')[1]
    subjects_per_class[activity].add(subject)
    path = os.path.join(data_dir, f + '.csv')
    df = pd.read_csv(path)
    df = df.loc[: df.last_valid_index()]
    data.append((subject, activity, df))
    df_tmp = pd.melt(df,var_name='directions',value_name='acc')
    df_tmp = pd.concat([pd.DataFrame({'subject': np.repeat(subject,len(df_tmp)),'activity': np.repeat(activity,len(df_tmp))}),
                        df_tmp],axis=1)
    if len(df_data)==0:
        df_data=df_tmp
    else: 
        df_data = pd.concat([df_data,df_tmp],ignore_index='True')
    
{k: len(v) for k, v in subjects_per_class.items()}
  
df.head()
df.shape[0] / fs / 60

#%% compute and plot samples per class
samples_per_class = collections.defaultdict(int)
for subject, activity, df in data:
    samples_per_class[activity] += len(df)
    

activity, n_samples = list(zip(*samples_per_class.items()))
plt.figure(figsize=(6, 4))
plt.bar(range(3), n_samples)
plt.xticks(range(3), activity);

# Plotting by combining subjects data
plt.figure(figsize=(6, 4))
g = sns.FacetGrid(df_data.drop(columns=["subject"]), col="directions", hue="activity")
g.map(sns.kdeplot, "acc", multiple="stack")
g.add_legend()
g.set(xlim=(-20, 20))

plt.figure(figsize=(6, 4))
ax = sns.violinplot(x="directions", y="acc", hue="activity",
                    data=df_data.drop(columns=["subject"]),inner='quartile', width=0.8, linewidth=0.5)

# plotting individual subject data ( a little too busy)..
g = sns.catplot(x='activity', y="acc",
                hue="subject", col="directions",
                data=df_data, kind="violin", 
                height=5, aspect=.8, linewidth=.5);
g.add_legend()





#%% plot and look at the data
# type this to properly execute the following: %matplotlib auto

for subject, activity, df in sorted(data, key=lambda x: x[1]):
    ts = np.arange(len(df)) / fs
    plt.clf() 
    #fig, (ax1, ax2) = plt.subplots(2,1,1, sharey='all')
    plt.plot(ts, df.accx, label='x')
    plt.plot(ts, df.accy, label='y')
    plt.plot(ts, df.accz, label='z')
    plt.plot(ts,np.sqrt(df.accx**2+ df.accy**2+ df.accz**2),'k')
    plt.title('{}_{}'.format(subject, activity))
    plt.legend()
    plt.ylim((-25, 25))
    plt.draw() 
    plt.show()
    plt.pause(2)
    plt.xlim(0,30)
    plt.pause(5)
    #while not plt.waitforbuttonpress(timeout=1):
    #    pass




