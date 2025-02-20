from datetime import datetime
import math
import csv
import numpy as np
import pandas as pd
from os import listdir
import os
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.patches as mpatches
import time
from config import *


tf.compat.v1.disable_eager_execution()


# %%
#@title Read in PPMI data
LEVDATA=pd.read_csv(f"{YOUR_PATH_TO_PPMI}/LEDD_Concomitant_Medication_Log_Coded_Synthetic.csv")
LEVDATA=LEVDATA[np.isnan(LEVDATA.LEDD)==False]
LEVDATA=LEVDATA[LEVDATA.LEDD>0]
LEVDATA=LEVDATA[~(LEVDATA.STARTDT==LEVDATA.STOPDT)]
LEVDATA["Duplicate"]=False
patients=np.unique(LEVDATA.PATNO)
for i in range(len(patients)):
  duplicates=LEVDATA[(LEVDATA.PATNO==patients[i]) & (LEVDATA.LEVCODE!=0)][["LEVCODE","FRQ","LEDD","STARTDT","STOPDT"]].duplicated()
  LEVDATA.loc[(LEVDATA.PATNO==patients[i]) & (LEVDATA.LEVCODE!=0),"Duplicate"]=duplicates
LEVDATA=LEVDATA[LEVDATA.Duplicate==False]
  #Max-norm type-0 LEDD 
nonlev_max=LEVDATA.loc[(LEVDATA.LEVCODE==0) & ~(LEVDATA.LEDTRT.isin(['SAFINAMIDE','XADAGO','ONSTRYV (SAFINAMIDE)','XADAGO 50MG','SAFINAMIDA'])),"LEDD"].max()
LEVDATA.loc[LEVDATA.LEVCODE==0,"LEDD"]=LEVDATA.loc[LEVDATA.LEVCODE==0,"LEDD"]/nonlev_max
LEVDATA.loc[LEVDATA.LEVCODE==0,"LEDD"]=np.clip(LEVDATA.loc[LEVDATA.LEVCODE==0,"LEDD"],0,1)

UPDRSDATA=pd.read_csv(f"{YOUR_PATH_TO_PPMI}/UPDRS Data Synthetic.csv")
UPDRSDATA=UPDRSDATA[(UPDRSDATA.PDSTATE!="OFF") & (UPDRSDATA.EVENT_ID!="SC")] #eliminate 'off' assessments and screening assessments (since screening is redundant with baseline)
UPDRSDATA=UPDRSDATA[~np.isnan(UPDRSDATA.NP3TOT)] #eliminate missing total scores
duplicate_updrs_filter=np.flip(UPDRSDATA.iloc[::-1][["PATNO","INFODT"]].duplicated())
UPDRSDATA=UPDRSDATA[~duplicate_updrs_filter]
#Recall that UPDRSDATA is sorted by patno, then by date (did this in Excel)

LEVDATA=LEVDATA[LEVDATA.PATNO.isin(UPDRSDATA.PATNO)]

# %%
#@title Create matrix of gender, age, years since PD
age_data=pd.read_csv(f"{YOUR_PATH_TO_PPMI}/Age Synthetic.csv")
gender_data=pd.read_csv(f"{YOUR_PATH_TO_PPMI}/Demographics Synthetic.csv")
diagnosis_data=pd.read_csv(f"{YOUR_PATH_TO_PPMI}/PD Diagnosis History Synthetic.csv")

patients=np.unique(LEVDATA.PATNO)

ages=[]
gender=[]
years=[]
for p in patients:
  ages.append(np.min(age_data[age_data.PATNO==p]["AGE_AT_VISIT"]))
  gender.append(np.min(gender_data[gender_data.PATNO==p]["SEX"]))
  if np.sum(diagnosis_data.PATNO==p)==0:
    years_since=-1
  else:
    sc_date=pd.to_datetime(diagnosis_data[diagnosis_data.PATNO==p]["INFODT"].iloc[0])
    dx_date=pd.to_datetime(diagnosis_data[diagnosis_data.PATNO==p]["PDDXDT"].iloc[0])
    years_since=(np.timedelta64(sc_date-dx_date,"D")/np.timedelta64(1, 'D')) / 365
  years.append(years_since)

covar_data=pd.DataFrame({
    'PATNO': patients,
    'Gender': gender,
    'Age': ages,
    'Years_PD': years
})

covar_data.loc[covar_data.Years_PD==-1,"Years_PD"]=np.median(covar_data.loc[covar_data.Years_PD!=-1,"Years_PD"])
#Regression predicting from age was not significant
covar_data.Age=covar_data.Age/np.max(covar_data.Age)


# %%
#@title Show patient inclusion filter and calculate 'LEVDATA_TRIMMED' (filtered to patients taking L-dopa)

#There are 743 patients, before data cleaning
patno_all=np.unique(LEVDATA.PATNO)
print(len(patno_all))
#There are 610 patients taking levodopa IR, CR, or Rytary
patno_123=np.unique(LEVDATA[LEVDATA.LEVCODE.isin([1,2,3])].PATNO)
print(len(patno_123))
#There are 511 patients NOT taking an unspecified levodopa medication or COMT-inhibitors (but may be taking non-L-Dopa medication)
patno_all=np.unique(LEVDATA.PATNO)
patno_4=np.unique(LEVDATA[LEVDATA.LEVCODE==4].PATNO)
patno_5=np.unique(LEVDATA[LEVDATA.LEVCODE==5].PATNO)
patno_123=np.unique(LEVDATA[LEVDATA.LEVCODE.isin([1,2,3])].PATNO)

LEVDATA_TRIMMED=LEVDATA[(LEVDATA.PATNO.isin(patno_123) & ~LEVDATA.PATNO.isin(patno_4) & ~LEVDATA.PATNO.isin(patno_5))]

print(len(np.unique(LEVDATA_TRIMMED.PATNO)))
patients=np.unique(LEVDATA_TRIMMED.PATNO)
print([len(np.unique(LEVDATA_TRIMMED[LEVDATA_TRIMMED.LEVCODE==c].PATNO)) for c in [1,2,3]])

# %%
#@title Plot individual LEDD doses
i =  5#@param {type:"number"}

patients=np.unique(LEVDATA_TRIMMED.PATNO)

updrs_scores=UPDRSDATA[UPDRSDATA.PATNO==patients[i]].NP3TOT
updrs_dates=pd.to_datetime(UPDRSDATA[UPDRSDATA.PATNO==patients[i]].INFODT)

levcodes=LEVDATA_TRIMMED[LEVDATA_TRIMMED.PATNO==patients[i]].LEVCODE
ledd=LEVDATA_TRIMMED[LEVDATA_TRIMMED.PATNO==patients[i]].LEDD
startdates=pd.to_datetime(LEVDATA_TRIMMED[LEVDATA_TRIMMED.PATNO==patients[i]].STARTDT)
stopdates=pd.to_datetime(LEVDATA_TRIMMED[LEVDATA_TRIMMED.PATNO==patients[i]].STOPDT)
stopdates[stopdates.isna()]=np.max([np.max(stopdates),np.max(startdates),np.max(updrs_dates)])

fig, ax1=plt.subplots()

[ax1.plot([np.array(startdates)[levcodes==0][i],np.array(stopdates)[levcodes==0][i]],[np.array(ledd)[levcodes==0][i],np.array(ledd)[levcodes==0][i]],
              color="gray") for i in range(len(levcodes[levcodes==0]))]
[ax1.plot([np.array(startdates)[levcodes==1][i],np.array(stopdates)[levcodes==1][i]],[np.array(ledd)[levcodes==1][i],np.array(ledd)[levcodes==1][i]],
              color="blue") for i in range(len(levcodes[levcodes==1]))]
[ax1.plot([np.array(startdates)[levcodes==2][i],np.array(stopdates)[levcodes==2][i]],[np.array(ledd)[levcodes==2][i],np.array(ledd)[levcodes==2][i]],
              color="green") for i in range(len(levcodes[levcodes==2]))]
[ax1.plot([np.array(startdates)[levcodes==3][i],np.array(stopdates)[levcodes==3][i]],[np.array(ledd)[levcodes==3][i],np.array(ledd)[levcodes==3][i]],
              color="orange") for i in range(len(levcodes[levcodes==3]))]

#[plt.plot([np.array(startdates)[i],np.array(stopdates)[i]],[np.array(ledd)[i],np.array(ledd)[i]],color="blue",label="Nonlev") for i in range(len(ledd))]
ax1.set_ylim(bottom=0)

patch1 = mpatches.Patch(color='gray', label='Non-L-dopa')
patch2 = mpatches.Patch(color='blue', label='L-dopa IR')
patch3 = mpatches.Patch(color='green', label='L-dopa CR')
patch4 = mpatches.Patch(color='orange', label='Rytary')
patch5 = mpatches.Patch(color='black', label='UPDRS')

ax1.legend(handles=[patch1,patch2,patch3,patch4,patch5],bbox_to_anchor=[1.4,.7])
ax1.set_ylabel("LEDD")

ax2=ax1.twinx()
ax2.set_ylabel("UPDRS")
ax2.plot(updrs_dates,updrs_scores,"x",color="black",linestyle="--")

# %%
#@title Define smoothing function (analysis does not actually smooth UPDRS scores - but this was written in an earlier project iteration, and is now used with the smoothing parameter turned off)

def tricube_ma(x,y,n): #note that n is the number of PAST points
  if n==0:
    smoothed=np.array(y)
  else:
    smoothed=[]
    for i in range(0,np.min([n,len(y)])):
      smoothed.append(y[i])
    if len(y)>n:
      for i in range(n,len(y)):
        delta=1.0001*np.clip(np.abs(x[i]-x[i-n]),a_min=1,a_max=None)
        weights=(1-(np.abs(x[i]-x[(i-n):(i+1)])/delta)**3)**3
        weights=weights/np.sum(weights)
        smoothed.append(np.sum(weights*y[(i-n):(i+1)]))
  return smoothed

# %%
#@title Define logit and sigmoid functions
def logit(array,xmin,xmax):
  proportion=(array-xmin)/(xmax-xmin)
  return -1*np.log(1/proportion-1)

def sigmoid(array,xmin,xmax):
  proportion=1/(1+np.exp(-1*array))
  return xmin+proportion*(xmax-xmin)

# %%
#Average number of years available
dates=[pd.to_datetime(UPDRSDATA[UPDRSDATA.PATNO==p].INFODT) for p in patients]
np.mean([(d.max()-d.min())/np.timedelta64(365,'D') for d in dates])
np.std([(d.max()-d.min())/np.timedelta64(365,'D') for d in dates])

# %%
#@title Plot example L-dopa / UPDRS trajectory
i =  3 #@param {type:"number"}
smooth =  1#@param {type:"number"}
use_smoothing =  0#@param {type:"number"}

patients=np.unique(LEVDATA_TRIMMED.PATNO)

updrs_scores=UPDRSDATA[UPDRSDATA.PATNO==patients[i]].NP3TOT
updrs_dates=pd.to_datetime(UPDRSDATA[UPDRSDATA.PATNO==patients[i]].INFODT)

levcodes=LEVDATA_TRIMMED[LEVDATA_TRIMMED.PATNO==patients[i]].LEVCODE
ledd=LEVDATA_TRIMMED[LEVDATA_TRIMMED.PATNO==patients[i]].LEDD
startdates=pd.to_datetime(LEVDATA_TRIMMED[LEVDATA_TRIMMED.PATNO==patients[i]].STARTDT)
stopdates=pd.to_datetime(LEVDATA_TRIMMED[LEVDATA_TRIMMED.PATNO==patients[i]].STOPDT)
stopdates[stopdates.isna()]=np.max([np.max(stopdates),np.max(startdates),np.max(updrs_dates)])

total_ledd=[]
dates=[]

for type in [0,1,2,3]:
  alldates=np.unique(np.sort(np.concatenate([startdates[levcodes==type],stopdates[levcodes==type]])))
  alldates=np.stack([alldates[:-1],alldates[1:]],axis=1)
  dates.append(alldates)

  _ledd=ledd[levcodes==type]
  _indices=[(startdates[levcodes==type]<=alldates[i,0]) & (stopdates[levcodes==type]>=alldates[i,1]) for i in range(alldates.shape[0])]
  total_ledd.append([np.sum(_ledd[i]) for i in _indices])
  

fig, ax1=plt.subplots()

ax1.set_ylabel("MDS-UPDRS-III")
np.random.seed(2025)


ax1.plot(updrs_dates,updrs_scores,"x",color="black",linestyle="--")


from statsmodels.nonparametric.smoothers_lowess import lowess
daydiffs=np.array((updrs_dates - updrs_dates.iloc[0]).astype('timedelta64[h]')/24)

smoothed=tricube_ma(x=np.array(daydiffs),y=np.array(updrs_scores),n=smooth)

if use_smoothing==1:
  ax1.plot(updrs_dates,smoothed,"o",color="orange",linestyle="-")

ax2=ax1.twinx()


np.random.seed(1000)
total_ledd[1] = [t+np.random.choice([-200,-100,-50,-25,25,50,75,100]) for t in total_ledd[1]]
total_ledd[2] = [t+np.random.choice([-200-100,-50,-25,25,50,75,100]) for t in total_ledd[2]]
total_ledd[3] = [t+np.random.choice([-200,-100,-50,-25,25,50,75,100]) for t in total_ledd[3]]

for i in range(len(total_ledd[1])): dates[1][i,0] = dates[1][i,0] + np.timedelta64(round(np.random.normal(0,60)),'D')
for i in range(len(total_ledd[2])): dates[2][i,0] = dates[2][i,0] + np.timedelta64(round(np.random.normal(0,60)),'D')
for i in range(len(total_ledd[3])): dates[3][i,0] = dates[3][i,0] + np.timedelta64(round(np.random.normal(0,60)),'D')

[ax2.plot([dates[1][i,0],dates[1][i,1]],[total_ledd[1][i],total_ledd[1][i]],
              color="black") for i in range(len(total_ledd[1]))] #or "black" "#FF0066"
[ax2.plot([dates[2][i,0],dates[2][i,1]],[total_ledd[2][i],total_ledd[2][i]],
              color='0.5') for i in range(len(total_ledd[2]))] #or 0.5 "#FF6600"
[ax2.plot([dates[3][i,0],dates[3][i,1]],[total_ledd[3][i],total_ledd[3][i]],
              color='0.8') for i in range(len(total_ledd[3]))] #or 0.8 "#C8A200"

ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)

patch2 = mpatches.Patch(color='black', label='L-dopa IR')
patch3 = mpatches.Patch(color='0.5', label='L-dopa CR')
patch4 = mpatches.Patch(color='0.1', label='Rytary')
patch5 = mpatches.Patch(color='black', label='UPDRS')
from matplotlib.lines import Line2D

line1 = Line2D([0], [0], label='Immediate', color='black')#'#FF0066')
line2 = Line2D([0], [0], label='Controlled', color='0.5')#'#FF6600')
line3 = Line2D([0], [0], label='Extended', color='0.8')#'#C8A200')
line4 = Line2D([0], [0], label='MDS-UPDRS-III', color='black',linestyle="--")


ax2.legend(handles=[line1,line2,line3,line4],prop={'size': 9},loc='lower left',
           ncol=2)
ax2.set_ylabel("LEDD")



# %%
#@title Write function for compiling final LEDD and UPDRS matrix
tlag=1 #set to 1 to only use current time point, 2 to use one previous month, etc.

def create_data(tlag=tlag):

  patients=np.unique(LEVDATA_TRIMMED.PATNO)
  num_updrs=[len(UPDRSDATA[UPDRSDATA.PATNO==patients[i]].NP3TOT) for i in range(len(patients))]


  array=[]
  updrs_init=[] #These will be arctanh-transformed updrs scores

  for i in range(len(patients)):
    if i%100==0:
      print(i)

    updrs_scores=UPDRSDATA[UPDRSDATA.PATNO==patients[i]].NP3TOT
    #ARCTANH TRANSFORM
    lbound=0;ubound=53 #53 is 99th percentile, np.quantile(UPDRSDATA.NP3TOT,.99)
    
    updrs_normed=(np.clip(updrs_scores,a_min=0,a_max=53)-lbound)/(ubound-lbound) #Range between 0 and 1
    updrs_normed=updrs_normed*2-1 #Range between -1 and 1
    updrs_normed=np.clip(updrs_normed,a_min=-.999,a_max=.999)
    updrs_transformed=np.arctanh(updrs_normed)
    updrs_diff=np.array([updrs_transformed.iloc[i+1]-updrs_transformed.iloc[i]
                        for i in range(len(updrs_transformed)-1)])  
    
    updrs_init.append(updrs_transformed.iloc[0])
    
    updrs_dates=pd.to_datetime(UPDRSDATA[UPDRSDATA.PATNO==patients[i]].INFODT)
    updrs_lags=np.array([np.timedelta64(updrs_dates.iloc[i+1]-updrs_dates.iloc[i],"D")/np.timedelta64(1, 'D')
          for i in range(len(updrs_dates)-1)])
    
    updrs_diff=updrs_diff*180/updrs_lags 
    updrs_diff=np.array([0]+list(updrs_diff))

    levcodes=LEVDATA_TRIMMED[LEVDATA_TRIMMED.PATNO==patients[i]].LEVCODE
    ledd=LEVDATA_TRIMMED[LEVDATA_TRIMMED.PATNO==patients[i]].LEDD
    freq=LEVDATA_TRIMMED[LEVDATA_TRIMMED.PATNO==patients[i]].FRQ
    startdates=pd.to_datetime(LEVDATA_TRIMMED[LEVDATA_TRIMMED.PATNO==patients[i]].STARTDT)
    stopdates=pd.to_datetime(LEVDATA_TRIMMED[LEVDATA_TRIMMED.PATNO==patients[i]].STOPDT)
    stopdates[stopdates.isna()]=np.max([np.max(stopdates),np.max(startdates),np.max(updrs_dates)])

    total_ledd=[]
    freqs=[]
    dates=[]

    for ltype in [0,1,2,3]:
      alldates=np.unique(np.sort(np.concatenate([startdates[levcodes==ltype],stopdates[levcodes==ltype]])))
      alldates=np.stack([alldates[:-1],alldates[1:]],axis=1)
      dates.append(alldates)

      _ledd=ledd[levcodes==ltype]
      _freq=freq[levcodes==ltype]
      _indices=[(startdates[levcodes==ltype]<=alldates[i,0]) & (stopdates[levcodes==ltype]>=alldates[i,1]) for i in range(alldates.shape[0])]
      total_ledd.append(np.array([np.sum(_ledd[i]) for i in _indices]))
      freqs.append(np.array([np.max(_freq[i]) for i in _indices]))
      freqs[-1][np.isnan(freqs[-1])]=0 #if there is no matching interval, impute frequency of 0

    _all_ledd_matrices=[]
    _all_freq_matrices=[]
    for t in range(6):
      _type_ledd=[[],[],[],[]]
      _type_freq=[[],[],[],[]]

      for ydate in updrs_dates.iloc[tlag:]: 
        ydate=ydate-pd.Timedelta(t*30, unit='D')
        for ltype in [0,1,2,3]:
          if len(dates[ltype])>0:
            type_index=np.array([(ydate>dates[ltype][i,0]) & (ydate<=dates[ltype][i,1]) for i in range(dates[ltype].shape[0])])
            if np.sum(type_index)>0:
              _type_ledd[ltype].append(np.array(total_ledd[ltype])[type_index])
              _type_freq[ltype].append(np.array(freqs[ltype])[type_index])
            else:
              _type_ledd[ltype].append(np.array([0.]))
              _type_freq[ltype].append(np.array([0.]))

          else:
            _type_ledd[ltype].append(np.array([0.]))
            _type_freq[ltype].append(np.array([0.]))

      _ledd_matrix=np.array(_type_ledd).reshape([4,-1]).T
      _totledd=np.sum(_ledd_matrix[:,1:],axis=1) #sum only type 1, 2, 3
      ledd_denom=np.clip(_totledd.reshape(-1,1),a_min=.1,a_max=None) #so you don't divide by 0
      _normed_ledd_matrix=_ledd_matrix[:,1:] / ledd_denom  #don't normalize LEDD0
      _ledd_matrix=np.concatenate([_ledd_matrix[:,0].reshape(-1,1), #LEDD 0 
                                   _normed_ledd_matrix, #Normed LEDD 1, 2, 3
                                  _totledd.reshape(-1,1)],axis=1) #LEDD total
      
      _freq_matrix=np.array(_type_freq).reshape([4,-1]).T[:,1:] #remove frequency information for non-lev medication
      _maxfreq=np.max(_freq_matrix,axis=1)
      _freq_matrix=np.concatenate([_freq_matrix,_maxfreq.reshape(-1,1)],axis=1)

      _all_ledd_matrices.append(_ledd_matrix)
      _all_freq_matrices.append(_freq_matrix)

    _final_ledd_matrix=np.concatenate(_all_ledd_matrices,axis=1)
    _final_freq_matrix=np.concatenate(_all_freq_matrices,axis=1)

    #YSCORE
    _yscores=updrs_diff[tlag:] 
    
    _ylagfactors=180/updrs_lags[(tlag-1):]
    
    
    #DIFFERENCING
    _xscores=np.array([
              np.array(updrs_transformed)[((tlag-1)-d):(-1-d)] for d in range(tlag)
    ])


    _xscores=_xscores.T #now most recent is first column    
    o_time = np.array([time.mktime(dt.timetuple()) / (3600 * 24) for dt in updrs_dates.iloc[tlag:]])
    array.append(np.concatenate([o_time.reshape(-1,1),
                                 np.ones([len(_yscores),1])*patients[i],
                                _yscores.reshape(-1,1),
                                _ylagfactors.reshape(-1,1),
                                _xscores.reshape(-1,tlag),
                                _final_ledd_matrix[:,:],
                                _final_freq_matrix[:,:]],axis=1))

  xscorenames=['XSCORE_'+str(i) for i in range(1,tlag+1)]
  leddnames=['LEDD0','LEDD1','LEDD2','LEDD3','LEDDTOT']
  freqnames=['FREQ1','FREQ2','FREQ3','FREQMAX']
  all_leddnames=list(np.concatenate([[l+"_"+str(t) for l in leddnames] for t in range(6)]))
  all_freqnames=list(np.concatenate([[f+"_"+str(t) for f in freqnames] for t in range(6)]))

  data=pd.DataFrame(np.concatenate(array,axis=0),
        columns=list(['DATETIME','PATNO','YSCORE','LAGFACTOR']+xscorenames+
                all_leddnames+
                all_freqnames))
  
  return data, updrs_init


# %%
#@title Define levodopa tracker functions (copied from script MS1)
time_interval=30

class lev_tracker():
    def __init__(self,hours_to_peak,hold_hours,cmax_per_mg,half_life,mg,time_interval):
        self.t=0
        self.hours_to_peak=hours_to_peak;self.hold_hours=hold_hours;self.cmax_per_mg=cmax_per_mg;self.half_life=half_life
        self.mg=mg
        self.lev=[0]
        self.rate=.5**(time_interval/(self.half_life*60))   #rate for 2 minutes
    def advance(self):
        self.t=self.t+time_interval
        if self.t<=(self.hours_to_peak*60):
            self.lev.append(self.lev[-1]+((self.cmax_per_mg*self.mg)/(self.hours_to_peak*60))*time_interval)
        elif self.t<=(self.hours_to_peak*60+self.hold_hours*60):
            self.lev.append(self.lev[-1])
        elif self.t>(self.hours_to_peak*60+self.hold_hours*60):
            self.lev.append(self.lev[-1]*self.rate)
    def multi_advance(self,minutes):
        for i in range(int(minutes/time_interval)):
            self.advance()
    def auc(self):
        return np.sum(np.array(self.lev)*time_interval/60)
    def reset(self):
        self.t=0
        self.lev=[0]
       



# %%
#@title Write levodopa simulation function (produces daily L-dopa profile based on L-dopa regimen)

def simulate_levday(daily_dose,dose_freq,levcode,waking_hours=17,time_interval=30):
  trackers=[]
  
  split_fractions=np.array(range(0,dose_freq))/dose_freq
  split_minutes=split_fractions*waking_hours*60
  med_times=np.round(split_minutes/time_interval)*time_interval
  dose=daily_dose/dose_freq
  times=list(range(0,waking_hours*60+1,time_interval))
  
  for t in times:
    [track.advance() for track in trackers]

    if t in med_times:
      if levcode==1:
        trackers.append(lev_tracker(hours_to_peak=1,hold_hours=0,cmax_per_mg=10.9,half_life=1.6,
                                    mg=dose,time_interval=time_interval))
      if levcode==2:
        trackers.append(lev_tracker(hours_to_peak=1.5,hold_hours=0,cmax_per_mg=8.55,half_life=1.6,
                                    mg=dose,time_interval=time_interval))
      if levcode==3:
        trackers.append(lev_tracker(hours_to_peak=1,hold_hours=3.5,cmax_per_mg=3.4,half_life=1.6,
                                    mg=dose,time_interval=time_interval))
  
  lev_list=[tr.lev for tr in trackers]
  lev_matrix=np.zeros([len(lev_list),len(times)])
  for i in range(lev_matrix.shape[0]):
    lev_matrix[i,-len(lev_list[i]):]=lev_list[i]

  return np.sum(lev_matrix,axis=0)

test1=simulate_levday(daily_dose=300,dose_freq=4,levcode=1,waking_hours=16,time_interval=30)
test2=simulate_levday(daily_dose=300,dose_freq=4,levcode=2,waking_hours=16,time_interval=30)
test3=simulate_levday(daily_dose=300,dose_freq=4,levcode=3,waking_hours=16,time_interval=30)

plt.plot(test1)
plt.plot(test2)
plt.plot(test3)

# %%
#@title Write function to create array of levodopa profiles
def create_levarray(data,tlag=6,waking_hours=17,time_interval=30):

  data_lev=np.zeros([data.shape[0],int(waking_hours*60/time_interval+1),tlag])

  for i in range(data.shape[0]):
    if i%500==0:
      print(i)

    for l in range(tlag):
      ir_dose=data.loc[data.index[i],'LEDD1_'+str(l)]*data.loc[data.index[i],'LEDDTOT_'+str(l)]
      cr_dose=data.loc[data.index[i],'LEDD2_'+str(l)]*data.loc[data.index[i],'LEDDTOT_'+str(l)]/.7
        #Make sure you divide by 0.7 to convert to original dose - this is the LEDD factor PPMI used for L-dopa CR and Rytary
      ryt_dose=data.loc[data.index[i],'LEDD3_'+str(l)]*data.loc[data.index[i],'LEDDTOT_'+str(l)]/.7
      ir_freq=data.loc[data.index[i],'FREQ1_'+str(l)]
      cr_freq=data.loc[data.index[i],'FREQ2_'+str(l)]
      ryt_freq=data.loc[data.index[i],'FREQ3_'+str(l)]
      ir_seq=simulate_levday(daily_dose=ir_dose, dose_freq=int(np.max([ir_freq,1])), levcode=1, waking_hours=waking_hours, time_interval=time_interval)
      cr_seq=simulate_levday(daily_dose=cr_dose, dose_freq=int(np.max([cr_freq,1])), levcode=2, waking_hours=waking_hours, time_interval=time_interval)
      ryt_seq=simulate_levday(daily_dose=ryt_dose, dose_freq=int(np.max([ryt_freq,1])), levcode=3, waking_hours=waking_hours, time_interval=time_interval)
      
      total_lev=1*ir_seq+.75*cr_seq+.5*ryt_seq

      data_lev[i,:,l]=total_lev
  return data_lev
  


# %%
#@title Define training, validation, and testing patients

np.random.seed(262)
ryt_patients=np.unique(LEVDATA_TRIMMED[LEVDATA_TRIMMED.LEVCODE==3].PATNO);np.random.shuffle(ryt_patients)
cr_patients=np.unique(LEVDATA_TRIMMED[(~LEVDATA_TRIMMED.PATNO.isin(ryt_patients)) & (LEVDATA_TRIMMED.LEVCODE==2)].PATNO);np.random.shuffle(cr_patients)
ir_patients=np.unique(LEVDATA_TRIMMED[(~LEVDATA_TRIMMED.PATNO.isin(cr_patients)) & (~LEVDATA_TRIMMED.PATNO.isin(ryt_patients))].PATNO);np.random.shuffle(ir_patients)

ryt_patient_folds=np.array_split(ryt_patients,10)
cr_patient_folds=np.array_split(cr_patients,10)
ir_patient_folds=np.array_split(ir_patients,10)
def create_fold(indices):
  patient_ids=[]
  for i in indices:
    patient_ids.append(ryt_patient_folds[i])
    patient_ids.append(cr_patient_folds[i])
    patient_ids.append(ir_patient_folds[i])
  return np.concatenate(patient_ids)

train_fold1=create_fold([0,1,2,3,4,5,6,7])
val_fold1=create_fold([8])
test_fold1=create_fold([9])

train_fold2=create_fold([0,1,2,3,4,5,8,9])
val_fold2=create_fold([6])
test_fold2=create_fold([7])

train_fold3=create_fold([0,1,2,3,6,7,8,9])
val_fold3=create_fold([4])
test_fold3=create_fold([5])

train_fold4=create_fold([0,1,4,5,6,7,8,9])
val_fold4=create_fold([2])
test_fold4=create_fold([3])

train_fold5=create_fold([2,3,4,5,6,7,8,9])
val_fold5=create_fold([0])
test_fold5=create_fold([1])

train_folds=[train_fold1,train_fold2,train_fold3,train_fold4,train_fold5]
val_folds=[val_fold1,val_fold2,val_fold3,val_fold4,val_fold5]
test_folds=[test_fold1,test_fold2,test_fold3,test_fold4,test_fold5]

# %%
#@title Create datasets for given updrs_lag (takes a few minutes to run)
include_nonlev=True

data_updrs_list=[]
data_lev_list=[]
data_cov_list=[]

waking_hours=17
time_interval=30

for updrs_lag in [1,2,3,4,5,6]: # [1,2,3,4,5,6]
  data, updrs_init=create_data(tlag=updrs_lag)
  covar_index=np.concatenate([np.where(covar_data.PATNO==p)[0] for p in np.array(data.PATNO)])
  data_covar=covar_data.iloc[covar_index,:]
  if include_nonlev==True:
    data_covar.loc[:,"Nonlev"]=np.array(data["LEDD0_0"])

  data_lev=create_levarray(data=data,tlag=6,waking_hours=17,time_interval=30) #Use all six past months, then filter as needed for different tlags. Remember [:,:,0] is most recent and [:,:,5] is oldest
  
  data_lev_list.append(data_lev)
  data_updrs_list.append(data)
  data_cov_list.append(data_covar)

  data_covar.to_csv(f"{OUTPUT_PATH}/data_cov_ulag"+str(updrs_lag)+".csv")
  data.to_csv(f"{OUTPUT_PATH}/data_updrs_ulag"+str(updrs_lag)+".csv")
  np.savetxt(f"{OUTPUT_PATH}/data_lev_ulag"+str(updrs_lag)+".csv",data_lev.reshape(data_lev.shape[0],-1),delimiter=",")
  np.savetxt(f"{OUTPUT_PATH}/updrs_transformed_init.txt",np.array(updrs_init),delimiter=",") 


# %%
#@title Load datasets for each updrs_lag (computed in previous block)
data_updrs_list=[]
data_lev_list=[]
data_cov_list=[]

waking_hours=16
time_interval=30

for updrs_lag in [1,2,3,4,5,6]:
  data_lev=np.loadtxt(f"{OUTPUT_PATH}/data_lev_ulag"+str(updrs_lag)+".csv",delimiter=",")
  data_cov=pd.read_csv(f"{OUTPUT_PATH}/data_cov_ulag"+str(updrs_lag)+".csv")
  data_updrs=pd.read_csv(f"{OUTPUT_PATH}/data_updrs_ulag"+str(updrs_lag)+".csv")

  #Write over any nan or inf values (as a result of synthetic dataset)
  data_lev[data_lev==np.inf] = np.nanquantile(data_lev, 0.95)
  data_updrs[data_updrs==np.inf] = np.nanquantile(data_updrs, 0.95)
  data_cov[data_cov==np.inf] = np.nanquantile(data_cov, 0.95)
  data_lev[np.isnan(data_lev)] = np.nanquantile(data_lev, 0.95)
  data_updrs[np.isnan(data_updrs)] = np.nanquantile(data_updrs, 0.95)
  data_cov[np.isnan(data_cov)] = np.nanquantile(data_cov, 0.95)

  data_updrs_list.append(data_updrs)
  data_cov_list.append(data_cov)
  data_lev_list.append(data_lev.reshape(data_lev.shape[0],-1,6))

updrs_init=np.loadtxt(f"{OUTPUT_PATH}/updrs_transformed_init.txt",delimiter=",") 


# %%
#@title Create 'dummy dataset' with tlag=1, to compute descriptive statistics
updrs_data_descriptive = pd.DataFrame(data_updrs_list[0])
ledd_ir_vecs=[];ledd_cr_vecs=[];ledd_ryt_vecs=[];ledd_tot_vecs=[]
freq_vec=[]

patients = np.unique(updrs_data_descriptive.PATNO)

for p in patients:
  ledd_ir_data, ledd_cr_data, ledd_ryt_data, ledd_tot_data = [np.array(updrs_data_descriptive.loc[updrs_data_descriptive.PATNO==p , updrs_data_descriptive.columns.str.startswith(prefix)]) for prefix in ["LEDD1","LEDD2","LEDD3","LEDDTOT"] ]
  
  freq_data = np.array(updrs_data_descriptive.loc[updrs_data_descriptive.PATNO==p, updrs_data_descriptive.columns.str.startswith('FREQMAX')])
  [lvec.append(np.concatenate([np.flip(ldata[0,1:]), ldata[:,0]])) for lvec, ldata in zip([ledd_ir_vecs,ledd_cr_vecs,ledd_ryt_vecs,ledd_tot_vecs],[ledd_ir_data,ledd_cr_data,ledd_ryt_data,ledd_tot_data])]
  freq_vec.append(np.concatenate([np.flip(freq_data[0,1:]), freq_data[:,0]]))

ledd_avg = np.array([np.mean(np.trim_zeros(l)) for l in ledd_tot_vecs])
ir_mg_avg = np.array([np.mean(np.trim_zeros(l)) for l in ledd_ir_vecs])/1
cr_mg_avg = np.array([np.mean(np.trim_zeros(l)) for l in ledd_cr_vecs])/0.7
ryt_mg_avg = np.array([np.mean(np.trim_zeros(l)) for l in ledd_ryt_vecs])/0.7
freq_avg = np.array([np.mean(np.trim_zeros(f)) for f in freq_vec])

ledd_avg[np.isnan(ledd_avg)] = 0
ir_mg_avg[np.isnan(ir_mg_avg)] = 0
cr_mg_avg[np.isnan(cr_mg_avg)] = 0
ryt_mg_avg[np.isnan(ryt_mg_avg)] = 0
freq_avg[np.isnan(freq_avg)] = 0

# %%
#@title Function to train and return linear regression model
from statsmodels.api import WLS
import statsmodels.api as sm

def linreg_models(updrs_lag=6,lev_lag=3,use_nonlev=False,pharmaco=True):
    data=data_updrs_list[updrs_lag-1]
    data_lev=data_lev_list[updrs_lag-1]
    data_lev=data_lev[:,:,0:lev_lag]
    data_covar=data_cov_list[updrs_lag-1]

    model_list=[]

    for f in range(5):
      train_y=data[data.PATNO.isin(train_folds[f])]["YSCORE"]
      val_y=data[data.PATNO.isin(val_folds[f])]["YSCORE"]

      if use_nonlev==True:
        train_x1=np.array(data_covar[data_covar.PATNO.isin(train_folds[f])][["Gender","Age","Years_PD","Nonlev"]])
      else:
        train_x1=np.array(data_covar[data_covar.PATNO.isin(train_folds[f])][["Gender","Age","Years_PD"]])
      train_x2=np.array(data.loc[data.PATNO.isin(train_folds[f]),data.columns.str.startswith('XSCORE')])
      
      
      if pharmaco==True:
        train_x3=data_lev[data.PATNO.isin(train_folds[f]),:,:].reshape(-1,data_lev.shape[1]*data_lev.shape[2])
        #Remove this if not standardizing
        train_x3_std=(train_x3-np.mean(train_x3))/np.std(train_x3)
        #train_x3_std=train_x3/np.max([np.max(train_x3),.1])

      if pharmaco==False:
        train_x3_std =np.array(
            data.loc[data.PATNO.isin(train_folds[f]),
            data.columns.str.startswith('LEDD1') | 
            data.columns.str.startswith('LEDD2') |
            data.columns.str.startswith('LEDD3') |
            data.columns.str.startswith('FREQ1') | 
            data.columns.str.startswith('FREQ2') | 
            data.columns.str.startswith('FREQ3')])
   
      ryt_index=data.loc[data.PATNO.isin(train_folds[f]),"PATNO"].isin(ryt_patients)
      cr_index=data.loc[data.PATNO.isin(train_folds[f]),"PATNO"].isin(cr_patients)
      ir_index=data.loc[data.PATNO.isin(train_folds[f]),"PATNO"].isin(ir_patients)

      weights=np.ones(train_x1.shape[0])
      weights[ryt_index]=1/np.sum(ryt_index)
      weights[cr_index]=1/np.sum(cr_index)
      weights[ir_index]=1/np.sum(ir_index)

      model=WLS(train_y,(np.concatenate([train_x1,train_x2,train_x3_std],axis=1)),weights=weights)
      modelfit=model.fit()

      model_list.append(modelfit)

    return model_list  
      



# %%
#@title Linear regression RMSE. Note: Most of these are zeros, because of an earlier project iteration in which different L-dopa lags were tested.
#Current version only uses the most recent L-dopa profile, to be consistent with the PKG study. Shape of rmse array is maintained so as not to break code.

rmse_reg=np.array([[[0.84530735, 0.58669352, 0.67335095, 0.88067296, 0.53955139],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ]],

       [[0.85462734, 0.57907505, 0.62352246, 0.821264  , 0.54448583],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ]],

       [[0.52199508, 0.59051559, 0.63010695, 0.70195382, 0.52919351],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ]],

       [[0.46131564, 0.56111642, 0.63764849, 0.6688221 , 0.53667165],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ]],

       [[0.43332501, 0.53159303, 0.56760173, 0.69117238, 0.52730651],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ]],

       [[0.44998786, 0.55820914, 0.55381153, 0.69093449, 0.4944361 ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ]]])

rmse_reg=np.tanh(rmse_reg)

# %%
#@title Create Convolutional Neural Net (CNN) Model Class.
#Note: Filter sizes here refer to number of times points (which are every half hour). So filter size k=8 corresponds to 4 hours, k=24 to 12 hours.
from keras.initializers import GlorotUniform
from tensorflow.keras import regularizers

class conv_model():
  def __init__(self,updrs_lag=6,lev_lag=3,lev_seq_length=35,use_nonlev=False,
    weight_file="",seed=2022,ksize=8,l2=0):
    if use_nonlev==True:
      ncov=4
    else:
      ncov=3
    
    np.random.seed(seed)
    tf.random.set_seed(seed)
      
    self.lev_input=tf.keras.Input(shape=(lev_seq_length,lev_lag))

    self.lev_layer1=tf.keras.layers.Conv1D(filters=lev_lag,kernel_size=ksize,activation='sigmoid',
                    kernel_initializer=GlorotUniform(seed=np.random.choice(range(1000))),
                    kernel_regularizer=regularizers.l2(l2)
                    )
    self._lev_layer1=self.lev_layer1(self.lev_input)
    
    self.lev_layer2=tf.keras.layers.MaxPooling1D(pool_size=lev_seq_length-ksize+1) #reduces to 7
    
    self._lev_layer2=self.lev_layer2(self._lev_layer1)

    self.lev_output=keras.layers.Reshape(target_shape=(lev_lag,))
 
    self._lev_output = self.lev_output(self._lev_layer2)

    self.x_input=tf.keras.Input(shape=(updrs_lag+ncov)) #number of xscore columns, plus 4 demographic covariates 
    self.all_input=tf.keras.layers.Concatenate(axis=-1)([self.x_input,self._lev_output])
    self.hidden1=tf.keras.layers.Dense(np.floor((updrs_lag+lev_lag+ncov)/2),activation='sigmoid',
                          kernel_initializer=GlorotUniform(seed=np.random.choice(range(1000))),
                          kernel_regularizer=regularizers.l2(l2)
                          )
    
    self._hidden1=self.hidden1(self.all_input)
    
    self.tanh_output=tf.keras.layers.Dense(1,activation='tanh',
          kernel_initializer=GlorotUniform(seed=np.random.choice(range(1000))))
    
    self._tanh_output=self.tanh_output(self._hidden1)

    self.final_output=tf.keras.layers.Lambda(lambda x: x * 5.0)(self._tanh_output) #5 is the .998 and .002 percentile

    self.model=keras.Model([self.lev_input,self.x_input],self.final_output)
    self.model.compile(loss='mse',optimizer='Adam')
    if weight_file!="":
      self.model.load_weights(weight_file)



# %%
#@title Create LSTM Model Class
from keras.initializers import GlorotUniform
from tensorflow.keras import regularizers

class lstm_model():
  def __init__(self,updrs_lag=6,lev_lag=3,lev_seq_length=35,use_nonlev=False,
    weight_file="",seed=2022,layers = 1,l2=0,ksize=8): #not actually using ksize in this model, since it doesn't use a CNN.
    if use_nonlev==True:
      ncov=4
    else:
      ncov=3
    
    np.random.seed(seed)
    tf.random.set_seed(seed)
      
    self.lev_input=tf.keras.Input(shape=(lev_seq_length,lev_lag))
    
    if layers==1:

      self.lev_layer_lstm = tf.keras.layers.LSTM(units=3, 
                activation='tanh',
                use_bias = True,
                kernel_initializer=GlorotUniform(seed=np.random.choice(range(1000))),
                kernel_regularizer = regularizers.l2(l2),
                return_sequences = False
                )
      
      self._lev_layer_lstm = self.lev_layer_lstm(self.lev_input)
      
      
      self.lev_output=tf.keras.layers.Reshape(target_shape=(3*lev_lag,))
      self._lev_output = self.lev_output(self._lev_layer_lstm)
 
    
    self.x_input=tf.keras.Input(shape=(updrs_lag+ncov)) #number of xscore columns, plus 4 demographic covariates 
    self.all_input=tf.keras.layers.Concatenate(axis=-1)([self.x_input,self._lev_output])
    self.hidden1=tf.keras.layers.Dense(np.floor((updrs_lag+lev_lag+ncov)/2),activation='sigmoid',
                          kernel_initializer=GlorotUniform(seed=np.random.choice(range(1000))),
                          )
    
    self._hidden1=self.hidden1(self.all_input)
    
    self.tanh_output=tf.keras.layers.Dense(1,activation='tanh',
          kernel_initializer=GlorotUniform(seed=np.random.choice(range(1000))))
    
    self._tanh_output=self.tanh_output(self._hidden1)

    self.final_output=tf.keras.layers.Lambda(lambda x: x * 5.0)(self._tanh_output) #5 is the .998 and .002 percentile

    self.model=keras.Model([self.lev_input,self.x_input],self.final_output)
    self.model.compile(loss='mse',optimizer='Adam')
    if weight_file!="":
      self.model.load_weights(weight_file)




# %%
#@title Save final LSTM model weights.
layers =  1#@param {type:"number"}
l2 = 0 #@param {type:"number"}
ksize =  8 #@param {type:"number"}
lev_lag=1


seed = 2025

rmse_lstm=np.zeros([6,5])

for updrs_lag in [5]:
  use_nonlev=True
  epochs=100

  tf.keras.backend.clear_session()

  data=data_updrs_list[updrs_lag-1]
  data_lev=data_lev_list[updrs_lag-1]
  data_lev=data_lev[:,:,0:lev_lag]

  data_covar=data_cov_list[updrs_lag-1] 

  for f in range(5):
      print(f)
      train_y=data[data.PATNO.isin(train_folds[f])]["YSCORE"]
      val_y=data[data.PATNO.isin(val_folds[f])]["YSCORE"]

      if use_nonlev==True:
        train_covs=np.array(data_covar[data_covar.PATNO.isin(train_folds[f])][["Gender","Age","Years_PD","Nonlev"]])
      else:
        train_covs=np.array(data_covar[data_covar.PATNO.isin(train_folds[f])][["Gender","Age","Years_PD"]])
      
      train_x=np.array(data.loc[data.PATNO.isin(train_folds[f]),data.columns.str.startswith('XSCORE')])
      train_lev=data_lev[data.PATNO.isin(train_folds[f]),:,:]

      train_lev_std=(train_lev-np.mean(train_lev))/np.std(train_lev)


      if use_nonlev==True:
        val_covs=np.array(data_covar[data_covar.PATNO.isin(val_folds[f])][["Gender","Age","Years_PD","Nonlev"]])
      else:
        val_covs=np.array(data_covar[data_covar.PATNO.isin(val_folds[f])][["Gender","Age","Years_PD"]])
      val_x=np.array(data.loc[data.PATNO.isin(val_folds[f]),data.columns.str.startswith('XSCORE')])
      val_lev=data_lev[data.PATNO.isin(val_folds[f]),:,:]

      val_lev_std=(val_lev-np.mean(train_lev))/np.std(train_lev)

      ryt_index=data.loc[data.PATNO.isin(train_folds[f]),"PATNO"].isin(ryt_patients)
      cr_index=data.loc[data.PATNO.isin(train_folds[f]),"PATNO"].isin(cr_patients)
      ir_index=data.loc[data.PATNO.isin(train_folds[f]),"PATNO"].isin(ir_patients)

      weights=np.ones(train_x.shape[0])
      weights[ryt_index]=1/np.sum(ryt_index)
      weights[cr_index]=1/np.sum(cr_index)
      weights[ir_index]=1/np.sum(ir_index)
    
      mod=lstm_model(updrs_lag=updrs_lag,lev_lag=lev_lag,
                    lev_seq_length=data_lev.shape[1],use_nonlev=use_nonlev,
                  layers = layers,
                  seed = seed, l2 = l2,
                  ksize = ksize
                  ) #set seed to get reproducible results

      history=mod.model.fit([train_lev_std,np.concatenate([train_covs,train_x],axis=1)],train_y,
                        epochs=epochs,verbose=0,
                        sample_weight=weights)
      
      mse=mod.model.evaluate([val_lev_std,np.concatenate([val_covs,val_x],axis=1)],val_y)
      
      rmse_lstm[updrs_lag-1,f]=np.sqrt(np.mean(mse))
      
      if use_nonlev==False:
        if ksize==8:
          mod.model.save_weights(f"{WEIGHT_PATH}val_plainlstm3_weights_layer"+str(layers)+"_reg"+str(l2)+"_fold"+str(f)+"_ulag"+str(updrs_lag)+"_llag"+str(lev_lag)+".h5")
        else:
          mod.model.save_weights(f"{WEIGHT_PATH}val_plainlstm3_weights_layer"+str(layers)+"_reg"+str(l2)+"_fold"+str(f)+"_ulag"+str(updrs_lag)+"_llag"+str(lev_lag)+"_k24.h5")
      if use_nonlev==True:
        if ksize==8:
          mod.model.save_weights(f"{WEIGHT_PATH}val_plainlstm3_nonlev_weights_layer"+str(layers)+"_reg"+str(l2)+"_fold"+str(f)+"_ulag"+str(updrs_lag)+"_llag"+str(lev_lag)+".h5")
        else:
          mod.model.save_weights(f"{WEIGHT_PATH}val_plainlstm3_nonlev_weights_layer"+str(layers)+"_reg"+str(l2)+"_fold"+str(f)+"_ulag"+str(updrs_lag)+"_llag"+str(lev_lag)+"_k24.h5")
      print(rmse_lstm)

