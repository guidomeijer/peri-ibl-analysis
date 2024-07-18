# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 13:28:32 2024 by Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join, realpath, dirname
from one.api import ONE
import pingouin as pg
from iblutil.numerical import ismember
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas, BrainRegions
from brainbox.population.decode import get_spike_counts_in_bins

# Settings
REGION = 'SSp-bfd'
ALPHA = 0.05
CONS_BIN_CRIT = 5
MIN_FR = 0.1

# Bins of 50 ms incremented by 10 ms
intervals = np.vstack((np.arange(0, 400, 10), np.arange(50, 450, 10))).T  # in ms
#intervals = np.vstack((np.arange(0, 900, 10), np.arange(50, 950, 10))).T  # in ms
intervals = intervals / 1000  # in seconds
time_bins = (intervals[:, 0] + intervals[:, 1]) / 2  # time bin centers

# Initialize ONE connection
one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international',
          silent=True, cache_rest=None)
ba = AllenAtlas()
br = BrainRegions()

# Query PERI recordings
insertions = one.alyx.rest('insertions', 'list', atlas_acronym=REGION)
pids = np.array([i['id'] for i in insertions])

sig_neurons, sig_time_period = pd.DataFrame(), pd.DataFrame()
for i, pid in enumerate(pids):
    print(f'Starting recording {i+1} of {len(pids)}')
    
    # Get eid and probe name
    eid, probe = one.pid2eid(pid)
    
    # Load in trials
    data = one.load_object(eid, 'trials')
    
    # Convert trials to dataframe
    data = {your_key: data[your_key] for your_key in [
        'stimOn_times', 'probabilityLeft', 'contrastLeft', 'contrastRight', 'feedbackType', 'choice']}
    trials = pd.DataFrame(data=data)
    trials['signed_contrast'] = trials['contrastRight']
    trials.loc[trials['signed_contrast'].isnull(), 'signed_contrast'] = -trials['contrastLeft']
    
    # Select only correct trials
    hit_trials = trials[trials['feedbackType'] == 1]
    
    # Now let's select 240 trials, counterbalanced between left and right choices 
    # - 80 100% contrast trials 
    # - 80 25% contrast trials
    # - 80 12.5% contrast trials
    trials_100 = pd.concat((hit_trials[hit_trials['signed_contrast'] == -1][:40],
                            hit_trials[hit_trials['signed_contrast'] == 1][:40]))
    trials_25 = pd.concat((hit_trials[hit_trials['signed_contrast'] == -0.25][:40],
                           hit_trials[hit_trials['signed_contrast'] == 0.25][:40]))
    trials_125 = pd.concat((hit_trials[hit_trials['signed_contrast'] == -0.125][:40],
                            hit_trials[hit_trials['signed_contrast'] == 0.125][:40]))
    use_trials = pd.concat((trials_100, trials_25, trials_125))
    if use_trials.shape[0] != 240:
        print('Not enough trials!')
        continue
        
    # Load in neural data
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)
    
    # Remap to Beryl acronyms
    _, inds = ismember(br.acronym2id(clusters['acronym']), br.id[br.mappings['Allen']])
    acronyms = br.get(br.id[br.mappings['Beryl'][inds]])['acronym']
    peri_neurons = np.where(acronyms == REGION)[0]
    
    # Exclude extremely low firing rate neurons
    low_fr_neurons = np.zeros(peri_neurons.shape[0]).astype(bool)
    for nn, neuron_id in enumerate(peri_neurons):
        firing_rate = np.sum(spikes.clusters == neuron_id) / spikes.times[-1]
        if firing_rate < MIN_FR:
            low_fr_neurons[nn] = True
    peri_neurons = peri_neurons[~low_fr_neurons]
    
    # Loop over neurons
    object_cell = np.zeros(peri_neurons.shape[0]).astype(int)
    for kk, neuron_id in enumerate(peri_neurons):
        this_neuron_df = pd.DataFrame()
        these_spikes = spikes.times[spikes.clusters == neuron_id]
        
        # Loop over trials
        for ii in use_trials.index.values:
            
            # Get spike counts for this neuron and this trial
            this_stim_on = use_trials.loc[ii, 'stimOn_times']
            these_intervals = this_stim_on + intervals
            these_spike_counts, _ = get_spike_counts_in_bins(these_spikes,
                                                             np.ones(these_spikes.shape[0]).astype(int),
                                                             these_intervals)
            
            # Add to dataframe
            this_neuron_df = pd.concat((this_neuron_df, pd.DataFrame(data={
                'spike_counts': these_spike_counts[0].astype(int),
                'side': np.ceil(use_trials.loc[ii, 'signed_contrast']),
                'contrast': np.abs(use_trials.loc[ii, 'signed_contrast']),
                'timebin': time_bins})))
                
        # Do two way ANOVA per timebin
        result = np.empty((intervals.shape[0], 2))
        for jj, t in enumerate(np.unique(this_neuron_df['timebin'])):
            if this_neuron_df.loc[this_neuron_df['timebin'] == t, 'spike_counts'].sum() > 0:
                anova_result = pg.anova(data=this_neuron_df[this_neuron_df['timebin'] == t],
                                        dv='spike_counts', between=['side', 'contrast'])
                result[jj, 0] = anova_result.loc[0, 'p-unc']
                result[jj, 1] = anova_result.loc[0, 'np2']
            else:
                result[jj, 0] = 1
                result[jj, 1] = 0
            
        # find time bins with significant p-val in ANOVA
        sig_bin = np.where(result[:,0]<ALPHA)[0]
        if sig_bin.shape[0] == 0:
            continue

        # find the peak selectivity time bin among significant bins (using eta-squared)
        max_bin = sig_bin[np.argmax(result[sig_bin,1])]    
        # consider the maximum eta-squared bin
        object_bin = []
        for i in range(max_bin):
            if result[max_bin-i,0] < ALPHA:
                object_bin.append(max_bin-i)
            else:
                break
        for i in range(max_bin+1, time_bins.shape[0]):
            if result[i,0] < ALPHA:
                object_bin.append(i)
            else:
                break    
        object_bin = np.array(object_bin)
        if len(object_bin) < CONS_BIN_CRIT:
            object_bin = np.array([])
            continue
        else:
            object_cell[kk] = 1
            
            # Add significant time period to dataframe
            sig_time_period = pd.concat((sig_time_period, pd.DataFrame(index=[sig_time_period], data={
                'max_time_bin': time_bins[max_bin],
                'begin_sig_period': np.min(time_bins[object_bin]),
                'end_sig_period': np.max(time_bins[object_bin])})))
    
    print(f'{np.sum(object_cell)} of {object_cell.shape[0]} significant neurons '
          f'{np.round((np.sum(object_cell) / object_cell.shape[0]) * 100, decimals=1)}%')
    
    # Add to dataframe
    sig_neurons = pd.concat((sig_neurons, pd.DataFrame(index=[sig_neurons.shape[0]], data={
        'pid': pid, 'n_neurons': object_cell.shape[0], 'sig_neurons': np.sum(object_cell),
        'perc_sig': np.round((np.sum(object_cell) / object_cell.shape[0]) * 100, decimals=1)})))
    
    # Save to disk
    sig_time_period.to_csv(join(dirname(realpath(__file__)), f'significant_time_period_{REGION}.csv'),
                           index=False)
    sig_neurons.to_csv(join(dirname(realpath(__file__)), f'significant_neurons_{REGION}.csv'),
                       index=False)
    

    