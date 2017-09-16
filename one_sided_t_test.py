# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:52:01 2016

@author: stevegoodman
one sided T-tests
https://www.linkedin.com/in/volodymyr-kazantsev-3307321?authType=OUT_OF_NETWORK&authToken=RfAk&locale=en_US&srchid=1696200741461061993887&srchindex=4&srchtotal=172&trk=vsrp_people_res_name&trkInfo=VSRPsearchId%3A1696200741461061993887%2CVSRPtargetId%3A4793940%2CVSRPcmpt%3Aprimary%2CVSRPnm%3Afalse%2CauthType%3AOUT_OF_NETWORK

"""
import numpy as np
import pandas as pd
from statsmodels import stats
import scipy.stats
import math
#code from scratch
def standard_error(sample):
    n = sample.shape[0]
    sd = math.sqrt(np.power((sample - sample.mean()), 2).sum()/(n - 1))
    return sd/math.sqrt(n)


def one_sample_ttest(sample, popmean, one_sided=False):
    sample =np.asarray(sample)
    xbar = sample.mean()
    n = sample.shape[0]
    
    se = standard_error(sample)
    
    t = (xbar - popmean) / se
    p = 1 - scipy.stats.t.cdf(abs(t), n-1)
    p *= (2 - one_sided) # if not one_sided: p *= 2
    return t, p, se

if __name__ =="__main__":
    
    samp = np.array([4.5,4,3.5,4.5,3,2])
    t, p, se = one_sample_ttest(samp, popmean=3.0, one_sided=True)
    print(' mean = {xbar} \n t-stat = {t} \n p-value ={p}'.format(xbar=samp, t=t, p=p ))

    # auto version    
    t,p = scipy.stats.ttest_1samp(samp, popmean=3.0 )
    print(' mean = {xbar} \n t-stat = {t} \n p-value ={p}'.format(xbar=samp, t=t, p=p/2))
    
    
    # and the same with z-test from statsmodels
    import statsmodels.stats.weightstats as wstats
    z, p = wstats.ztest(samp, value=3.0, usevar='pooled', ddof=1.0)
    
    print('z-statistics= {t} \n p-value = {p}'.format(t=t, p=p/2))


