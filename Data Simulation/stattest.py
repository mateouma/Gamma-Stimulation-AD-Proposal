import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.signal as sig
from scipy import fft
from scipy import stats


def shapiro_wilk_test(sample, x_lab, **kwargs):
    '''
    Performs a Shaprio-Wilk test for normality on a given sample. Plots histogram and returns W-statistic and p-value.
    
    params:
    sample (array_like) - Sample in question to be tested.
    x_lab (str) - Metric of sample.
    kwargs (key, value mappings) - Other keyword arguments passed through to seaborn.histplot().
    
    returns:
    ax (matplotlib Axes) - Object with the plot drawn onto it.
    wstat (float) - Test statistic.
    pval (float) - The p-value for the hypothesis test.
    '''
    wstat, pval = stats.shapiro(sample)
    
    sns.set_theme(style='dark')
    ax = sns.histplot(sample, **kwargs)
    ax.set(xlabel=x_lab)
    
    return ax, (wstat, pval)

def t_test(samples, test_type, sample_names, y_lab, **kwargs):
    '''
    Performs a t-test (one sample, two sample, or paired) on two samples. Plots boxplot and significance stars.
    
    params:
    samples (2-tuple of array-likes) - Samples to be tested.
    test_type (str) - Type of t-test used.
    sample_names (2-tuple of str's) - Labels on samples.
    y_lab (str) - Metric of samples
    kwargs (key, value mappings) - Other keyword arguments passed through to seaborn.boxplot().
    
    returns:
    ax (matplotlib Axes) - Object with the plot drawn onto it.
    vals (numpy array) - Concatenated samples.
    pval (float) - The p-value for the hypothesis test.
    '''
    samp1, samp2 = samples
    name1, name2 = sample_names
    
    samp1 = np.asarray(samp1)
    samp2 = np.asarray(samp2)
    
    n_1 = samp1.size
    n_2 = samp2.size
    
    vals = np.hstack((samp1, samp2))
    
    stat_sig = False
    
    d = {
        'vals': vals,
        'label': np.hstack((np.repeat(name1, n_1), np.repeat(name2, n_2)))
    }
    df = pd.DataFrame(data=d)
    
    if test_type == 'students':
        ttest = stats.ttest_ind(samp1, samp2)
    elif test_type == 'within-subjects':
        ttest = stats.ttest_rel(samp1, samp2)
    pval = ttest.pvalue
    
    if pval < 0.001:
        stars = "***"
        stat_sig = True
    elif pval < 0.01:
        stars = "**"
        stat_sig = True
    elif pval < 0.05:
        stars = "*"
        stat_sig = True
    
    sns.set_theme()
    ax = sns.boxplot(data=df, x='label', y='vals', **kwargs)#, hue='label')
    ax.set(xlabel='', ylabel=y_lab)
    
    if stat_sig:
        vspace = stats.tstd(vals) / 10
        y, h, col = np.max(vals) + vspace, vspace, '#3F3F3F'
    
        plt.plot([0, 0, 1, 1], [y, y+h, y+h, y], lw=1.5, c=col)
        plt.text((0+1)*.5, y+(h*0.7), stars, ha='center', va='bottom', color=col)
        
    return ax, (vals, pval)

