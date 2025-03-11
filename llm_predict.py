# -------------------------------------------------------
import os
import pandas as pd
import numpy as np
import time
import sys
import copy
import joblib
import re
import pickle

import gc
import threading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from filelock import FileLock
from funcs import *
from problems import *
from config import *
import argparse
os.environ["HF_ENDPOINT"] = model_proxy_base


parser = argparse.ArgumentParser()
parser.add_argument("--similar_case_num", type=int, default=2, help="The number of similar cases")
parser.add_argument("--test_patient_num", type=int, default=400, help="The maximum number of patients used for predition.")
parser.add_argument("--parallel_num", type=int, default=100, help="The number of concurrent LLM calls. Set a smaller value if your API providers do not possess enough request quotas.")
args, unknown = parser.parse_known_args()

original_stdout = sys.stdout

def generate_report(question,answer,patno,row,token_inf):
    fin_str = ('-'*4 + '\n')
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    fin_str += (f'# {formatted_time} | PATNO {patno} | Row {row}' + '\n')
    fin_str += (f'**Token used : INPUT {token_inf[0]} | OUTPUT {token_inf[1]} | SUM {token_inf[0]+token_inf[1]}**' + '\n')
    fin_str += ('## QUESTION\n' + question)
    fin_str += ('## ANSWER\n' + answer)
    return fin_str


def format_print_list(in_list,nonlev = False):
    fin_str = ''
    for i in in_list:
        i = 'None' if i == [[],[],[]] else i
        fin_str += f'{i}' if nonlev == False else f'{round(i,0)}'
        fin_str += ' / ' if nonlev == False else ', '
    return fin_str[:-2]


def get_patient_medication_history(LEDD,LEDD_FREQ,SUM_DOSE):
    fin_med = []
    for i in range(LEDD[0].shape[0]):
        sub_med = []
        for j in [0,1,2]:
            if SUM_DOSE.iloc[i] == 0:
                LEDD_FREQ[j].iloc[i] = 0
            all_dose = SUM_DOSE.iloc[i] * LEDD[j].iloc[i]
            sub_med.append([round(all_dose / LEDD_FREQ[j].iloc[i]) for _ in range(int(LEDD_FREQ[j].iloc[i]))])
        fin_med.append(sub_med)
    return fin_med


pmi = data_updrs=pd.read_csv(f"{OUTPUT_PATH}/data_updrs_ulag1.csv")

medicate_patient = np.unique(pmi.PATNO)
patient_medication_history = []
patient_medication_history_nonlev = []

for p in medicate_patient:

    LEDD = []; LEDD_FREQ = [] 
    LEDD_0 = pmi.loc[pmi.PATNO == p, f'LEDD0_0']
    SUM_DOSE = pmi.loc[pmi.PATNO == p, f'LEDDTOT_0']
    patient_medication_history_nonlev.append(list(LEDD_0 * SUM_DOSE))
    SUM_DOSE = SUM_DOSE[:5]
    for i in [1,2,3]:
        LEDD.append(pmi.loc[pmi.PATNO == p, f'LEDD{i}_0'][:5])
        LEDD_FREQ.append(pmi.loc[pmi.PATNO == p, f'FREQ{i}_0'][:5])
    patient_medication_history.append(get_patient_medication_history(LEDD,LEDD_FREQ,SUM_DOSE))

def check_stridt(str1:str,str2:str) -> bool:
    beg1_idx1 = str1.find(":"); beg1_idx2 = str2.find(":")
    end1_idx1 = str1.find("|"); end1_idx2 = str2.find("|")
    beg2_idx1 = str1.find(":",beg1_idx1+1); beg2_idx2 = str2.find(":",beg1_idx2+1)
    end2_idx1 = str1.find("|",end1_idx1+1); end2_idx2 = str2.find("|",end1_idx2+1)
    val1_1 = str1[beg1_idx1+1:end1_idx1].lstrip().rstrip(); val1_2 = str1[beg1_idx2+1:end1_idx2].lstrip().rstrip()
    val2_1 = int(float(str1[beg2_idx1+1:end2_idx1].lstrip().rstrip()))
    val2_2 = int(float(str2[beg2_idx2+1:end2_idx2].lstrip().rstrip()))
    judgement1 = (val1_1 == val1_2) and (abs(val2_1 - val2_2) <= 2)
    sub_str1 = str1.lstrip().splitlines(True)[3]; sub_str2 = str2.lstrip().splitlines(True)[3]
    separator = ','
    beg_idx1 = sub_str1.find(":"); beg_idx2 = sub_str2.find(":")
    sub_str1 = sub_str1[beg_idx1+1:]; sub_str2 = sub_str2[beg_idx2+1:]
    sub_str1 = sub_str1.split(separator); sub_str2 = sub_str2.split(separator)
    sub_str1 = set([s.lstrip().rstrip() for s in sub_str1]); sub_str2 = set([s.lstrip().rstrip() for s in sub_str2])
    common_str12 = sub_str1 & sub_str2
    judgement3 = len(common_str12) >= 1 
    return (judgement1 and judgement3)

        
def find_strlist_idx(in_str:str, in_list:list[str]) -> int:
    for idx, list_str in enumerate(in_list):
        if list_str.find(in_str.lstrip().rstrip()) >= 0:
            return idx
    return -1

def getNewidx(rag_result:list[str], rag_lib:list[str]) -> list[int]:
    fin_idx:list[int] = []
    for rag_str in rag_result:
        find_idx = find_strlist_idx(rag_str, rag_lib)
        if find_idx >= 0:
            fin_idx.append(find_idx)
    return fin_idx
        
def text_abstract(llm_txt:str) -> str:
    judgement1 = '### Brief Analysis of Patient Information'
    judgement2 = '### Analysis of Treatment Plans'
    index1 = llm_txt.find(judgement1); index2 = llm_txt.find(judgement2)
    if (index1 and index2):
        return llm_txt[:index1] + llm_txt[index2:]
    else:
        return ''
    
# --------------------------------------------



LEVDATA = pd.read_csv(f"{YOUR_PATH_TO_PPMI}LEDD_Concomitant_Medication_Log_Coded.csv")

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

UPDRSDATA=pd.read_csv(f"{YOUR_PATH_TO_PPMI}UPDRS Data.csv") 
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



age_data=pd.read_csv(f"{YOUR_PATH_TO_PPMI}Age Synthetic.csv")
gender_data=pd.read_csv(f"{YOUR_PATH_TO_PPMI}Demographics Synthetic.csv")
diagnosis_data=pd.read_csv(f"{YOUR_PATH_TO_PPMI}PD Diagnosis History Synthetic.csv")

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

n = 60  
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

run = 4  
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

age_csv = pd.read_csv(f"{YOUR_PATH_TO_PPMI}Age.csv") 
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


fin_qa = []; RAG_path = BASE_DIR + "/Rnewnew_RAG_infor.txt" 
RAG_path_full = BASE_DIR + "/Rnewnew_RAG_datalib.txt"  
RAG_path_LLM = BASE_DIR + "/LLMinterpret_RAG_datalib.txt"

LOG_DIR = BASE_DIR + '/llm_log.md'

example_input_str = []
example_output_str = []
fin_RAGprob = []


from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import LLMChain, PromptTemplate

from langchain_core.messages import HumanMessage
from langchain.vectorstores import FAISS


with open(RAG_path,'r') as f:
    rag_str = f.read()

with open(RAG_path_full,'r') as f:
    rag_str_full = f.read()

with open(RAG_path_LLM,'r') as f:
    rag_str_LLM = f.read()


rag_lib = rag_str.split('\n----\n')
rag_lib_full = rag_str_full.split('\n----\n')
rag_lib_LLM = rag_str_LLM.split('\n----\n')

LLM_CHOICE = 1; OUT_DIR = './Health-LLM/medalpaca-7b/'


if model_type == 'api':
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=model,
        openai_api_key=model_api_key,
        openai_api_base=model_base,
        temperature=temperature
        )
elif model_type == 'local':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import pipeline
    model =  AutoModelForCausalLM.from_pretrained(model, device_map='auto', max_length = 1000)
    tokenizer = AutoTokenizer.from_pretrained(model)
else:
    raise NotImplementedError


from langchain_community.vectorstores import FAISS
if embed_model_type == 'local':
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddingsModel = HuggingFaceEmbeddings(
        model_name=embed_model
    ) 
elif embed_model_type == 'api':
    from langchain_openai import OpenAIEmbeddings # for example
    embeddingsModel = OpenAIEmbeddings(
        model=embed_model,
        openai_api_base=embed_model_base,
        api_key=embed_api_key
    )

vectorstore = FAISS.from_texts(rag_lib, embeddingsModel)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 10}
) 


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            llm_system_message,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
chain = prompt | llm

def simulate_sequence(p, conv_models, n, use_reg, updrs_start, policy_simulators=None, data_lev=None, updrs_lag=6, lev_lag=1, sd_list=[0]*5, use_nonlev=False,
                      diff=False, pharmaco=True, ryt=False, use_ppmi_simulator=False,
                      max_led=None, rand_ppmi=False, llmcontrol=False, med_pat_idx = 0):
    gen_ok = True
    taken_actions = []  
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

    pred_updrs = np.tile(updrs_start, (n, 1))  

    cov_data = []
    x_data = []
    dose_data = []
    row = 0
    llm_actions = []
    for row in range(subset_x.shape[0]):
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
        cov_input = np.tile(subset_covs.iloc[row, :], (n, 1))  
        x_input = pred_updrs[:, 0:updrs_lag]  
        # print("cov_input",cov_input)
        # print("x_input",x_input)

        if use_ppmi_simulator == False: 
            lev_input = np.tile(subset_lev[row, :, :], (n, 1, 1))  
            # print("len_lev_input",len(lev_input))
            # print("lev_input",lev_input)
            ledd_input = np.tile(subset_ledd.iloc[row, :], (n, 1))  
        if use_ppmi_simulator == True:  
            state = np.concatenate([cov_input, x_input], axis=1)  

            kmeans_loaded = joblib.load(f'{WEIGHT_PATH}/kmeans_model.pkl')  
            cluster_label = kmeans_loaded.predict(cov_input)  
            tree_output = np.array(cluster_label).reshape(-1, 1).astype(float)  
            enhanced_state = np.concatenate([state, tree_output], axis=1)  
            
            choiceprobs = policy_simulators[train_fold].model.predict(enhanced_state)  
            
            if rand_ppmi == True:  
                np.apply_along_axis(np.random.shuffle, arr=choiceprobs, axis=1)  

            choiceprobs = mask(choiceprobs, max_led)  
            actions = np.argmax(choiceprobs, axis=1)  

            actions[np.sum(choiceprobs, axis=1) == 0] = 584  

            dose_vec_lists = [action_dose_lists[a] for a in actions]
            



        if llmcontrol==True:
            e_input_str = gen_patientinf_readable(rag_content_mcts,cov_input,x_input,user_input,use_x=True)
            med_history = patient_medication_history[med_pat_idx]
            med_history_nonlev = patient_medication_history_nonlev[med_pat_idx]
            e_input_str = e_input_str.replace('<medication_history>',format_print_list(med_history[-5:])).replace('<nldopa_history>',format_print_list(med_history_nonlev[row:row+5],nonlev=True))

            e_input_str = ''.join(e_input_str.splitlines(True)[:4]) 
            rag_input_str = ''.join(e_input_str.splitlines(True)[:4]) 
            results = retriever.get_relevant_documents(rag_input_str)

            results = [r.page_content for r in results]

            #question = gen_patientinf_readable(problem,cov_input,x_input,user_input,use_x=True)

            if USE_LLM:
                question = llm_user_message.replace('<patient_information>', e_input_str)
            else:
                question = llm_user_message_worag.replace('<patient_information>', e_input_str)
            rep_rag_str = ''
            

            result_idt = [check_stridt(e_input_str, docs) for docs in results]

            results_ = []; [(results_.append(results[i]) if not result_idt[i] else None) for i in range(len(result_idt))]
            results = results_

            use_rag_lib = rag_lib_LLM if USE_LLM else rag_lib_full
            newidx = getNewidx(results,use_rag_lib)

            fin_results = [];
            if USE_LLM:
                [fin_results.append(text_abstract(use_rag_lib[i])) for i in newidx]
            else:
                [fin_results.append(use_rag_lib[i]) for i in newidx]

            for idx, doc in enumerate(fin_results[:NUM_RAG_PATIENT]): 
                rep_rag_str += f'### Similar Case {idx + 1}\n'
                rep_rag_str += doc.lstrip().rstrip() + '\n'
            question = question.replace('<rag_context>',rep_rag_str)

            try_time = 0
            connect_time = 0
            while True:
                while True:
                    try:
                        response = chain.invoke({"messages": [HumanMessage(content=question)]})
                        break
                    except: 
                        connect_time += 1
                        print(f'Patient {p} Connection error! Try connecting again {connect_time}.',end='\r')
                pattern_python = r"\```python\n(.+?)\n```"
                matches = re.findall(pattern_python, response.content, re.DOTALL)
                log_str: str = generate_report(question, response.content, p, row, (response.response_metadata['token_usage']['prompt_tokens'],
                response.response_metadata['token_usage']['completion_tokens'])).rstrip() + '\n'
                lock = FileLock("lock3.lock")
                with lock: 
                    with open(LOG_DIR,'a',encoding='utf-8') as f:
                        f.write(log_str)
                try:
                    if len(matches) != 1:
                        raise ValueError
                    out_list = eval(matches[0])
                    ok = out_list.__class__ == list and out_list.__len__() == 3
                    if not ok:
                        raise ValueError
                    with open(LOG_DIR,'a') as f: 
                        f.write('**Status: Success**\n')
                    break
                except:
                    try_time += 1
                    with open(LOG_DIR,'a') as f:
                        f.write('***Status: Fail***\n')
                    if try_time > 5:
                        print('LLM predict error! Use RL regiment.')
                        gen_ok = False
                        break
                    print("Predict error! Try again...")

            if try_time < 5:
                llm_actions.append(out_list)
                dose_vec_lists= [out_list for _ in range(n)]

        if use_ppmi_simulator or llmcontrol:
            med_history.append(dose_vec_lists[0]) 
            raw_lev_input = [simulate_RL_levday(ir_seq=dose_vec_lists[sim][0],
                                                cr_seq=dose_vec_lists[sim][1],
                                                ryt_seq=dose_vec_lists[sim][2]) for sim in range(len(dose_vec_lists))]
            raw_lev_input = np.array(raw_lev_input).reshape(n, 35, 1) 

            lev_input = (raw_lev_input - np.mean(train_lev)) / np.std(train_lev)
            dose_list = dose_vec_lists[0]
            dose_data=([dose_list])
        #sys.stdout = StringIO()
        pred = mod.model.predict([lev_input, np.concatenate([cov_input, x_input], axis=1)], verbose = 0).reshape(-1)  
        pred = pred_updrs[:, 0] + pred / subset_lagfactors.iloc[row]
        pred = np.arctanh(np.clip(np.tanh(pred) + np.random.normal(loc=0, scale=sd, size=pred.shape), -.999, .999)) 
        pred_updrs = np.concatenate([pred.reshape(-1, 1), pred_updrs], axis=1) 
        cov_data=([cov_input.flatten().tolist()])
        #print(cov_data[0][:4])
        x_data=([x_input.flatten().tolist()])
        
    pred_updrs = np.flip(pred_updrs, axis=1)  
    pred_updrs = np.tanh(pred_updrs)
    pred_updrs = (pred_updrs + 1) / 2 
    pred_updrs = pred_updrs * 53
    return real_updrs, pred_updrs, taken_actions, llm_actions, gen_ok

start_time = time.time() 
NUM_RAG_PATIENT = args.similar_case_num # 
USE_LLM = False 


all_LLM_pred_updrs = []  
all_real_updrs = []  
all_RL_pred_updrs = []
all_action_arrays = []  
all_LLM_predict = []
all_pred_updrs = []
sd_lstm = rmse_lstm[updrs_lag - 1, :] 
sd_list = sd_lstm
sd_list = np.array(sd_list) * sd_mult
execute_PATNO = [] 


def LLM_pred_func(PATNO:int):
    med_pat_idx = np.where(medicate_patient == PATNO)[0]
    med_pat_idx = med_pat_idx[0] if med_pat_idx.shape[0] != 0 else -1
    if med_pat_idx == -1:
        return -1
    updrs = np.array(UPDRSDATA[UPDRSDATA.PATNO == PATNO].NP3TOT)
    _, pred_updrs, _, _, _ = simulate_sequence(
        PATNO,
        conv_models=lstm_models,
        use_ppmi_simulator=False,  
        # use_ppmi_simulator=False,  
        llmcontrol=False,
        policy_simulators=policy_simulators,  
        updrs_lag=updrs_lag, lev_lag=lev_lag,
        n=n, sd_list=sd_list, use_nonlev=use_nonlev,
        use_reg=use_reg,
        updrs_start=updrs[0:updrs_lag],  
        max_led=max_led_list[int(np.where(patients == PATNO)[0])],
        med_pat_idx=med_pat_idx
    )
    real_updrs, pred_LLM_updrs, taken_actions, llm_actions, gen_ok = simulate_sequence(
        PATNO,
        conv_models=lstm_models,
        use_ppmi_simulator=False,  
        # use_ppmi_simulator=False,  
        llmcontrol=True,
        policy_simulators=policy_simulators,  
        updrs_lag=updrs_lag, lev_lag=lev_lag,
        n=n, sd_list=sd_list, use_nonlev=use_nonlev,
        use_reg=use_reg,
        updrs_start=updrs[0:updrs_lag],  
        data_lev=None,  
        max_led=max_led_list[int(np.where(patients == PATNO)[0])],
        med_pat_idx=med_pat_idx
    )
    lock2 = FileLock("lock4.lock")
    with lock2:
        all_LLM_pred_updrs.append(np.zeros([n, 30])) 
        all_RL_pred_updrs.append(np.zeros([n, 30])) 
        all_pred_updrs.append(np.zeros([n, 30]))
        all_real_updrs.append(np.zeros(30)) 
        execute_PATNO.append(PATNO)
        all_LLM_pred_updrs[-1][:, -len(real_updrs):] = np.array(pred_LLM_updrs)
        all_real_updrs[-1][-len(real_updrs):] = np.array(real_updrs)
        all_pred_updrs[-1][:, -len(real_updrs):] = np.array(pred_updrs)
        all_action_arrays.append([])
        all_LLM_predict.extend(llm_actions)
    # return np.array(pred_LLM_updrs), np.array(real_updrs), np.array(taken_actions).T, llm_actions

if __name__ == '__main__':
    FIN_LENGTH = min(len(patients),args.test_patient_num) 
    print(f'{FIN_LENGTH} patients to be analyzed')
    response = chain.invoke({'messages':[HumanMessage(content="Hello !")]})
    print(f'Connection success!')
    PARALLEL_NUMS = args.parallel_num
    START_IDX = 0
    alive = np.ones(PARALLEL_NUMS)
    all_threads: list[threading.Thread] = []
    pointer = min(START_IDX+PARALLEL_NUMS,FIN_LENGTH)
    all_idx = list(range(START_IDX,START_IDX+PARALLEL_NUMS))
    [all_threads.append(threading.Thread(target=LLM_pred_func, args=(patients[i],))) for i in range(START_IDX,pointer)]
    [p.start() for p in all_threads]
    print('START RETRIEVING !')
    try:
        while True:
            time.sleep(0.05)

            for idx_, thread_ in enumerate(all_threads):
                alive[idx_] = int(thread_.is_alive())
                if not alive[idx_]:
                    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    time_elapse = time.time() - start_time
                    print_str = f"PATIENT {all_idx[idx_]} (PATNO {patients[all_idx[idx_]]}) FINISH ! time {formatted_time} time_elapse {round(time_elapse/60,2)}"
                    if pointer < FIN_LENGTH:
                        all_threads[idx_] = threading.Thread(target=LLM_pred_func, args = (patients[pointer], ))
                        all_threads[idx_].start()
                        all_idx[idx_] = pointer
                        print_str += f' START PATIENT {pointer} (PATNO {patients[pointer]}).'
                        pointer += 1; alive[idx_] = 1
                    if all_idx[idx_] > 0:
                        print(print_str)
                    if not all_threads[idx_].is_alive():
                        all_idx[idx_] = -1
            gc.collect()
            if np.sum(alive) == 0:
                break
    except KeyboardInterrupt:
        pass 
formatted_time = time.strftime("%Y_%m_%d %H_%M_%S", time.localtime()) 
with open(f'{EXP_OUTPUT_PATH }/{formatted_time}.pkl', 'wb') as f:
    pickle.dump((execute_PATNO, action_dose_lists,all_action_arrays,all_LLM_predict,all_real_updrs,all_pred_updrs,all_LLM_pred_updrs),f)




