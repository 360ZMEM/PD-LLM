from __future__ import division
import os 
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import copy
from mcts import mcts
import time
import pandas as pd
import numpy as np
import joblib
from funcs import *
from problems import *
from config import *
import mcts as mcts
DEBUG = False
from io import StringIO
from collections import deque
from filelock import FileLock
import gc


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--patient", type=int, default=0, help="patient")
parser.add_argument("--quiet", action="store_true")
args, unknown = parser.parse_known_args()
if args.quiet == True:
    sys.stdout = StringIO()

import warnings
warnings.filterwarnings("ignore")


LEVDATA = pd.read_csv(f"{YOUR_PATH_TO_PPMI}/LEDD_Concomitant_Medication_Log_Coded.csv")


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

UPDRSDATA=pd.read_csv(f"{YOUR_PATH_TO_PPMI}/UPDRS Data.csv") 
UPDRSDATA=UPDRSDATA[(UPDRSDATA.PDSTATE!="OFF") & (UPDRSDATA.EVENT_ID!="SC")] #eliminate 'off' assessments and screening assessments (since screening is redundant with baseline)
UPDRSDATA=UPDRSDATA[~np.isnan(UPDRSDATA.NP3TOT)] #eliminate missing total scores
duplicate_updrs_filter=np.flip(UPDRSDATA.iloc[::-1][["PATNO","INFODT"]].duplicated())
UPDRSDATA=UPDRSDATA[~duplicate_updrs_filter]
#Recall that UPDRSDATA is sorted by patno, then by date (did this in Excel)

LEVDATA=LEVDATA[LEVDATA.PATNO.isin(UPDRSDATA.PATNO)]



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


age_data=pd.read_csv(f"{YOUR_PATH_TO_PPMI}/Age Synthetic.csv")
gender_data=pd.read_csv(f"{YOUR_PATH_TO_PPMI}/Demographics Synthetic.csv")
diagnosis_data=pd.read_csv(f"{YOUR_PATH_TO_PPMI}/PD Diagnosis History Synthetic.csv")

patients=np.unique(LEVDATA.PATNO)
patients_old = np.unique(LEVDATA.PATNO)
ages=[]
gender=[]
years=[]

disease_time = []
for p in patients:
  ages.append(np.min(age_data[age_data.PATNO==p]["AGE_AT_VISIT"]))
  gender.append(np.min(gender_data[gender_data.PATNO==p]["SEX"]))
  if np.sum(diagnosis_data.PATNO==p)==0:
    years_since=-1
    disease_time.append(-1) 
  else:
    sc_date=pd.to_datetime(diagnosis_data[diagnosis_data.PATNO==p]["INFODT"].iloc[0])
    dx_date=pd.to_datetime(diagnosis_data[diagnosis_data.PATNO==p]["PDDXDT"].iloc[0])

    disease_time.append(time.mktime(dx_date.timetuple()) / (3600 * 24))
    years_since=(np.timedelta64(sc_date-dx_date,"D")/np.timedelta64(1, 'D')) / 365
  years.append(years_since)

covar_data=pd.DataFrame({
    'PATNO': patients,
    'Gender': gender,
    'Age': ages,
    'Years_PD': years
})

covar_data.loc[covar_data.Years_PD==-1,"Years_PD"]=np.median(covar_data.loc[covar_data.Years_PD!=-1,"Years_PD"])
covar_data.Age=covar_data.Age/np.max(covar_data.Age)


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

train_fold1=create_fold([0,1,2,3,4,5,6,7]) # 8-1-1 fold
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




updrs_lag =  6  
sd_mult =  1 
ksize =  8  
l2 = 0 
use_reg = False 

use_nonlev = True 
lev_lag = 1  
l2 = 0 
lev_seq_length = 35
ntxt = 'nonlev_'
weight_files = [f"{WEIGHT_PATH}/val_plainlstm3_" + str(ntxt) + "weights_layer1" +
                "_reg" + str(l2) +
                "_fold" + str(f) + "_ulag" + str(updrs_lag) + "_llag" + str(lev_lag) + ".h5" for f in range(5)]
lstm_models = [lstm_model(updrs_lag=updrs_lag, lev_lag=lev_lag, lev_seq_length=lev_seq_length,
                          use_nonlev=use_nonlev, weight_file=w, ksize=ksize)
               for w in weight_files]

run = 2  
inits = [[np.loadtxt(f"{WEIGHT_PATH}fold" + str(fold) + "_run" + str(run) + "_weights1_dys.txt", delimiter=","),
          np.loadtxt(f"{WEIGHT_PATH}fold" + str(fold) + "_run" + str(run) + "_bias1_dys.txt", delimiter=","),
          np.loadtxt(f"{WEIGHT_PATH}fold" + str(fold) + "_run" + str(run) + "_weights2_dys.txt", delimiter=","),
          np.loadtxt(f"{WEIGHT_PATH}fold" + str(fold) + "_run" + str(run) + "_bias2_dys.txt", delimiter=",")] for fold in range(5)]

policy_simulators = [PPMIPolicySimulator(inits=inits[f]) for f in range(5)]





pd_synthetic_csv = pd.read_csv(f"{OUTPUT_PATH}/data_updrs_ulag6_time.csv")
data = pd_synthetic_csv 

covar_index=np.concatenate([np.where(covar_data.PATNO==p)[0] for p in np.array(data.PATNO)])
data_covar=covar_data.iloc[covar_index,:]
data_covar.loc[:,"Nonlev"]=np.array(data["LEDD0_0"])
data_lev_all=create_levarray(data=data,tlag=6,waking_hours=17,time_interval=30)

age_csv = pd.read_csv(f"{YOUR_PATH_TO_PPMI}/Age.csv") 
updrs_orig_csv = pd.read_csv(f'{YOUR_PATH_TO_PPMI}UPDRS Data.csv')

age_updrs_csv = age_csv[["PATNO","EVENT_ID","AGE_AT_VISIT"]].merge(updrs_orig_csv[["PATNO","EVENT_ID","INFODT","NP3TOT"]])



age_updrs_csv["INFODT"] = [time.mktime(i.timetuple()) / (3600 * 24) for i in pd.to_datetime(age_updrs_csv["INFODT"])]


patients=np.unique(LEVDATA_TRIMMED.PATNO)
patients_old_2=np.unique(LEVDATA_TRIMMED.PATNO)

birthday_patno = np.unique(age_updrs_csv.PATNO)
birthday = np.zeros_like(birthday_patno)
for idx, p in enumerate(birthday_patno):

    line = age_updrs_csv[age_updrs_csv.PATNO == p].iloc[0]
    birthday[idx] = line.INFODT - line.AGE_AT_VISIT * 365


dose_name = ['']

data_lev=np.loadtxt(f"{OUTPUT_PATH}/data_lev_ulag6.csv",delimiter=",")
data_cov=pd.read_csv(f"{OUTPUT_PATH}/data_cov_ulag6.csv")
data_updrs=pd.read_csv(f"{OUTPUT_PATH}/data_updrs_ulag6_time.csv")



ledd_ir_vecs=[];ledd_cr_vecs=[];ledd_ryt_vecs=[];ledd_tot_vecs=[]
notwork_idx = []
for i in range(len(patients)):

    PATNO = patients[i]
    ledd_ir_data, ledd_cr_data, ledd_ryt_data, ledd_tot_data = [np.array(data_updrs.loc[data_updrs.PATNO== PATNO , data_updrs.columns.str.startswith(prefix)]) for prefix in ["LEDD1","LEDD2","LEDD3","LEDDTOT"] ]
    if len(ledd_ir_data) == 0:
        notwork_idx.append(i)
        continue
    [lvec.append(np.concatenate([np.flip(ldata[0,1:]), ldata[:,0]])) for lvec, ldata in zip([ledd_ir_vecs,ledd_cr_vecs,ledd_ryt_vecs,ledd_tot_vecs],[ledd_ir_data,ledd_cr_data,ledd_ryt_data,ledd_tot_data])]
patients = np.delete(patients,notwork_idx)

ledd_avg = np.array([np.mean(np.trim_zeros(l)) for l in ledd_tot_vecs])
max_led_list = list(ledd_avg)

fin_qa = []; RAG_path = BASE_DIR + "/Rnew_RAG_datalib.txt"

RAG_path_parallel = BASE_DIR + '/Rnew_RAG_infor.txt'
example_input_str = []
example_output_str = []
fin_RAGprob = []



def simulate_sequence(p, conv_models, use_reg, updrs_start, policy_simulators=None, data_lev=None, updrs_lag=6, lev_lag=1, sd_list=[0]*5, use_nonlev=False,
                      diff=False, pharmaco=True, ryt=False, use_ppmi_simulator=False,
                      max_led=None, rand_ppmi=False,  saveRAG=True, 
                      pred_updrs = None, in_row = 0, lev_input = None, choice_rand_ratio = 1): 
    
    include_dose_num = int(RL_filter * choice_rand_ratio)

    lbound = 0; ubound = 53 
    updrs_start_ = copy.deepcopy(updrs_start)
    updrs_start = (np.clip(updrs_start, a_min=0, a_max=53) - lbound) / (ubound - lbound)  
    updrs_start = updrs_start * 2 - 1  
    updrs_start = np.clip(updrs_start, a_min=-.999, a_max=.999)  
    updrs_start = np.arctanh(updrs_start)  
    uinit = updrs_start[0] 
    if diff == True:  
        updrs_start = np.flip([0] + list(np.diff(updrs_start)))  
    if diff == False:  
        updrs_start = np.flip(updrs_start)  

    data = data_updrs 
    if data_lev is None: 
        data_lev = np.array(np.array(data_lev_all)[:, :, 0:lev_lag])  
    data_covar = data_cov
    patients = np.unique(data.PATNO)
    train_fold = int(np.where(~np.array([p in f for f in train_folds]))[0])
    sd = sd_list[train_fold]
    mod = conv_models[train_fold]
    if use_nonlev == True: 
        subset_covs = data_covar[data_covar.PATNO == p][["Gender", "Age", "Years_PD", "Nonlev"]] 
    else: 
        subset_covs = data_covar[data_covar.PATNO == p][["Gender", "Age", "Years_PD"]]  
    

    cov_time = data.loc[data.PATNO == p, 'DATETIME']
    non_lev_data = data.loc[data.PATNO == p, 'LEDD0_0'] * data.loc[data.PATNO == p, 'LEDDTOT_0']


    subset_x = data.loc[data.PATNO == p, data.columns.str.startswith('XSCORE')]  
    subset_lagfactors = data.loc[data.PATNO == p, 'LAGFACTOR']  
    subset_y = data.loc[data.PATNO == p, 'YSCORE']  
    subset_ledd = data.loc[data.PATNO == p,
                           data.columns.str.startswith('LEDD1') |
                           data.columns.str.startswith('LEDD2') |
                           data.columns.str.startswith('LEDD3') |
                           data.columns.str.startswith('FREQ1') |
                           data.columns.str.startswith('FREQ2') |
                           data.columns.str.startswith('FREQ3')]  
    if ryt == True:  
        subset_ledd.loc[:, subset_ledd.columns.str.startswith('LEDD1')] = 0  
        subset_ledd.loc[:, subset_ledd.columns.str.startswith('LEDD2')] = 0  
        subset_ledd.loc[:, subset_ledd.columns.str.startswith('LEDD3')] = 0.5  
        subset_ledd.loc[:, subset_ledd.columns.str.startswith('FREQ1')] = 0  
        subset_ledd.loc[:, subset_ledd.columns.str.startswith('FREQ2')] = 0  
        subset_ledd.loc[:, subset_ledd.columns.str.startswith('FREQ3')] = 5 
    subset_lev = np.array(data_lev[np.array(data.PATNO == p), :, :])  


    train_lev = np.array(data_lev[data.PATNO.isin(train_folds[train_fold]), :, 0:lev_lag])  
    subset_lev = (subset_lev - np.mean(train_lev)) / np.std(train_lev)  


    init_updrs = np.flip(updrs_start)  

    if subset_x.shape[0] == 1:  
        final_updrs = (subset_y.iloc[-1] / subset_lagfactors.iloc[-1]) + init_updrs[-1] 
        real_updrs = np.concatenate([init_updrs.reshape(-1), final_updrs.reshape(-1)])  
    else:  
        remaining_updrs = np.array(subset_x.iloc[1:, 0])  
        final_updrs = (subset_y.iloc[-1] / subset_lagfactors.iloc[-1]) + remaining_updrs[-1] 
        real_updrs = np.concatenate([init_updrs.reshape(-1), remaining_updrs.reshape(-1), final_updrs.reshape(-1)])  

    real_updrs = np.tanh(real_updrs)  
    real_updrs = (real_updrs + 1) / 2  
    real_updrs = real_updrs * 53  
    init_updrs = np.array(updrs_start)  
    remaining_updrs_diffs = np.array(subset_y).reshape(-1)  

    real_updrs2 = np.array(init_updrs)  
    for i in range(len(subset_y)):
        real_updrs2 = np.append(real_updrs2, real_updrs2[-1] + (subset_y.iloc[i] / subset_lagfactors.iloc[i]))  

    real_updrs2 = np.tanh(real_updrs2) 
    real_updrs2 = (real_updrs2 + 1) / 2  
    real_updrs2 = real_updrs2 * 53  


    subset_x.iloc[1:, :] = 0  
    if type(pred_updrs) == type(None): 
        pred_updrs = updrs_start 
        pred_updrs = np.tile(pred_updrs, (n * include_dose_num,1))
    else:

        pred_updrs = np.tile(pred_updrs, (include_dose_num, 1))
    cov_data = []
    x_data = []
    dose_data = []
    row = in_row
    all_rows = subset_x.shape[0]

    event_day = cov_time.iloc[row] 

    birthday_patient_idx = np.where(patients_old_2 == p)[0]
    disease_patient_idx = np.where(patients_old == p)[0]

    birthday_patient_no = birthday_patient_idx[0] if birthday_patient_idx.shape[0] != 0 else -1
    disease_patient_no = disease_patient_idx[0] if disease_patient_idx.shape[0] != 0 else -1

    patient_age = ((event_day - birthday[birthday_patient_no]) / 365.25) if birthday_patient_idx.shape[0] != 0 else None
    disease_tstep = disease_time[disease_patient_no] if disease_patient_idx.shape[0] != 0 else -2
    disease_years = ((event_day - disease_tstep) / 365.25) if disease_tstep not in [-1,-2] else (-999 if disease_tstep == -1 else None) 
    nonlev = non_lev_data.iloc[row]

    user_input = [patient_age, disease_years, nonlev]

    cov_input = np.tile(subset_covs.iloc[row, :], (n * include_dose_num, 1))  
    x_input = pred_updrs[:, 0:updrs_lag] 
    rag_str = [copy.deepcopy(rag_content_mcts) for _ in range(include_dose_num)]

    if lev_input == None:
        np.random.seed(int((time.time() - int(time.time())) * 1000000))
        actions = np.random.choice(np.arange(0,73),include_dose_num,replace=False)


        bad_examples = []
        lev_idx = actions

        lev_input = [];
        [[lev_input.append(action_dose_lists[a]) for _ in range(n)] for a in actions]

    raw_lev_input = [simulate_RL_levday(ir_seq=lev_input[sim][0],
                                                cr_seq=lev_input[sim][1],
                                                ryt_seq=lev_input[sim][2]) for sim in range(len(lev_input))]
    raw_lev_input = np.array(raw_lev_input).reshape(n * include_dose_num, 35, 1)  
    lev_data = (raw_lev_input - np.mean(train_lev)) / np.std(train_lev)
    pred = mod.model.predict([lev_data, np.concatenate([cov_input, x_input], axis=1)]).reshape(-1)  
    pred = pred_updrs[:, 0] + pred / subset_lagfactors.iloc[row] # 
    pred = np.arctanh(np.clip(np.tanh(pred) + np.random.normal(loc=0, scale=sd, size=pred.shape), -.999, .999)) 
    pred_updrs = np.concatenate([pred.reshape(-1, 1), pred_updrs], axis=1)
    cov_data=([cov_input.flatten().tolist()])
    x_data=([x_input.flatten().tolist()])

    
    rag_str = [gen_patientinf_readable(r,cov_input,x_input,user_input,use_x=True) for r in rag_str] 

    return real_updrs, pred_updrs, rag_str, lev_input, lev_idx, bad_examples, nonlev




def MCTS_expand(tree_node, PATNO):
    start_row = tree_node.depth
    real_updrs, pred_updrs, rag_str, lev_input, lev_idx, bad_examples, nonlev = simulate_sequence(PATNO, conv_models=lstm_models, use_ppmi_simulator=True, policy_simulators=policy_simulators, updrs_lag=updrs_lag, lev_lag=lev_lag, sd_list=sd_list, use_nonlev=use_nonlev, use_reg=use_reg, updrs_start=updrs[0:updrs_lag],data_lev=None,max_led=max_led_list[int(np.where(patients == PATNO)[0])], 
        in_row = start_row)
    tree_node.nonlev = nonlev 
    subpatient_idx = [n * idx for idx in range(RL_filter)]

    if tree_node.choice == None:
        tree_node.choice = (None,None,rag_str[0],None,None,bad_examples)

    for idx in subpatient_idx:
        newNode = mcts.treeNode(tree_node,depth=start_row + 1,choice= (real_updrs,pred_updrs[idx:idx+n],rag_str[int(idx/n)],lev_input[idx],lev_idx[int(idx/n)],bad_examples)) 
        tree_node.children[lev_idx[int(idx/n)]] = newNode; 

        real_updrs_, pred_updrs_, rag_str_, lev_input_, lev_idx_, bad_examples_, nonlev_ = simulate_sequence(PATNO, conv_models=lstm_models, use_ppmi_simulator=True,policy_simulators=policy_simulators, updrs_lag=updrs_lag, lev_lag=lev_lag, sd_list=sd_list, use_nonlev=use_nonlev, use_reg=use_reg, updrs_start=updrs[0:updrs_lag],data_lev=None,max_led=max_led_list[int(np.where(patients == PATNO)[0])], 
        pred_updrs = pred_updrs[idx:idx+n], in_row=start_row + 1, choice_rand_ratio = second_selection_rate)
        newNode.nonlev = nonlev_
        newNode.get_med_history()

        subsubpatient_idx = [n * i for i in range(int(RL_filter * second_selection_rate))]
        for subidx in subsubpatient_idx:
            newNode_ = mcts.treeNode(newNode, depth=start_row + 2, choice=(real_updrs_,pred_updrs_[subidx:subidx+n],rag_str_[int(subidx/n)],lev_input_[subidx],lev_idx_[int(subidx/n)],bad_examples_))
            newNode.children[lev_idx_[int(subidx/n)]] = newNode_;

            pred_updrs__ = pred_updrs_[subidx:subidx+n]

            for frow in range(2,MCTS_in_depth):
                real_updrs__, pred_updrs__, rag_str__, lev_input__, lev_idx__, bad_examples__, nonlev__ = simulate_sequence(PATNO, conv_models=lstm_models, use_ppmi_simulator=True,policy_simulators=policy_simulators, updrs_lag=updrs_lag, lev_lag=lev_lag, sd_list=sd_list, use_nonlev=use_nonlev, use_reg=use_reg, updrs_start=updrs[0:updrs_lag],data_lev=None,max_led=max_led_list[int(np.where(patients == PATNO)[0])],
                 pred_updrs = pred_updrs__, in_row=start_row + frow, choice_rand_ratio = (1 / RL_filter))
                newNode_.children[lev_idx[0]] = mcts.treeNode(newNode_,depth=start_row + frow,choice= (real_updrs__,pred_updrs__,rag_str__[0],lev_input__[0],lev_idx__[0],bad_examples__))
                newNode_.nonlev = nonlev__
                newNode_.get_med_history()
                newNode_ = newNode_.children[lev_idx[0]]
            newNode_.parent.getDeltUpdrs()
    return tree_node


kmeans_loaded = joblib.load(f'{WEIGHT_PATH}/kmeans_model.pkl')

sd_lstm = rmse_lstm[updrs_lag - 1, :] 
sd_list = sd_lstm
sd_list = np.array(sd_list) * sd_mult


n = 8  
RL_filter = 73 
MCTS_in_depth = 4 
second_selection_rate = 0.33
FIN_LENGTH = 400
FIN_LENGTH = min(len(patients),FIN_LENGTH) # truncate
if os.path.exists(RAG_path) and DEBUG == False and args.patient == 0:
    os.remove(RAG_path)
if os.path.exists(RAG_path_parallel) and DEBUG == False and args.patient == 0:
    os.remove(RAG_path_parallel)
patient_decision_tree = []
legal_patients = []
all_rag_str = []

iter_best_child_num = 1 

for i in [args.patient]:
    PATNO = patients[i]
    med_pat_idx = np.where(medicate_patient == PATNO)[0]
    med_pat_idx = med_pat_idx[0] if med_pat_idx.shape[0] != 0 else -1
    if med_pat_idx == -1:
        continue
    med_history = patient_medication_history[med_pat_idx]
    med_history_nonlev = patient_medication_history_nonlev[med_pat_idx]
    patient_decision_tree_ = mcts.treeNode(None, 0, med_history=(med_history, med_history_nonlev))

    dec_len = data.loc[data.PATNO == PATNO, data.columns.str.startswith('XSCORE')].shape[0]

    if dec_len < (MCTS_in_depth + 1):
        continue
    updrs = np.array(UPDRSDATA[UPDRSDATA.PATNO == PATNO].NP3TOT)
    legal_patients.append((i,PATNO)) 

    patient_decision_tree_ =  MCTS_expand(patient_decision_tree_,PATNO=PATNO)
    queue = deque([patient_decision_tree_])

    while len(queue) != 0:
        node = queue.popleft()
        children = node.getBestChild()[:iter_best_child_num]
        if node.depth + MCTS_in_depth < (dec_len - 1) and node.depth <= 2:
            queue.extend(children)
        rag_txt = node.getRAGtext()
        del node
        gc.collect()
        rag_txt_parallel = ''.join(rag_txt.splitlines(True)[:4])  

        rag_txt += '\n----\n'; rag_txt_parallel += '\n----\n'
        mode = 'a' if os.path.exists(RAG_path) else "w"
        lock1 = FileLock("lock1.lock")
        with lock1:
            with open(RAG_path, mode) as f:
                f.write(rag_txt)
            with open(RAG_path_parallel, mode) as f:
                f.write(rag_txt_parallel)
        print('-------------------------------------------')
        print('            RAG GENERATE OK!')
        print('-------------------------------------------')
        for idx, cdr in enumerate(children):
            children[idx].del_children() 
            children[idx] = MCTS_expand(cdr,PATNO=PATNO)
    patient_decision_tree.append(patient_decision_tree_) 


os._exit(0)
