# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:00:33 2024 by Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
from os.path import join, realpath, dirname
import matplotlib.pyplot as plt

# Settings
REGION = 'SSp-bfd'

# Load in results
sig_time_period = pd.read_csv(join(dirname(realpath(__file__)), f'significant_time_period_{REGION}.csv'))
sig_time_period = sig_time_period.sort_values('max_time_bin').reset_index()
sig_neurons = pd.read_csv(join(dirname(realpath(__file__)), f'significant_neurons_{REGION}.csv'))

# Plot
f, ax1 = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
for i in sig_time_period.index.values:
    ax1.plot([sig_time_period.loc[i, 'begin_sig_period'], sig_time_period.loc[i, 'end_sig_period']],
             [i, i], color='gold', zorder=0)
    ax1.scatter(sig_time_period.loc[i, 'max_time_bin'], i, color='darkorange', s=1, zorder=1)
    
ax1.set(title=f'{REGION}', ylabel='Neurons', xlabel='Time (s)', xticks=[0, 0.1, 0.2, 0.3, 0.4])
sns.despine(trim=True)
plt.tight_layout()
