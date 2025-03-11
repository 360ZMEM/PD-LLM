import os
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from config import *
sns.set_style('whitegrid')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

updrs_lag = 6

fname = 'yyyy_mm_dd hh_mm_ss' # example

with open(f'{EXP_OUTPUT_PATH}/{fname}.pkl','rb') as f:
    execute_PATNO, action_dose_lists,all_action_arrays,all_LLM_predict,all_real_updrs,all_pred_updrs,all_LLM_pred_updrs = pickle.load(f)

with open(f'{EXP_OUTPUT_PATH}/result_5870.pkl','rb') as f:
    _, _, _, all_real_updrs_, _, _, all_RL_pred_updrs = pickle.load(f)

with open(f'{EXP_OUTPUT_PATH}/result_2570.pkl','rb') as f:
    action_dose_lists, all_action_arrays, _, _ ,_, _, _ = pickle.load(f)

med_take = [[action_dose_lists[aa] for aa in np.min(a,axis=0)] for a in all_action_arrays]
med_take_ = []; [med_take_.extend(aa) for aa in med_take]; med_take = med_take_

with open('exp_results/patient.pkl','rb') as f:
    patients = pickle.load(f)


patient_patno = np.zeros(399)

for i, value in enumerate(all_real_updrs):
    index = np.where(np.all(all_real_updrs_ == value, axis=1))[0]
    if index.size > 0:
        patient_patno[i] = patients[index[0]]
    pass

def get_dose_freq(in_list):
    #base_freq = max(len(in_list[0]),len(in_list[1]),len(in_list[2]))
    ir_freq = np.where(np.array(in_list[0]) != 0)
    cr_freq = np.where(np.array(in_list[1]) != 0)
    ryt_freq = np.where(np.array(in_list[2]) != 0)
    freq = np.union1d(ir_freq, cr_freq)
    freq = np.union1d(freq, ryt_freq)
    return freq.shape[0]

print('------ LLM ------')
llm_ir_time = np.sum(np.array([(1 if np.sum(np.array(l[0])) > 1 else 0) for l in all_LLM_predict])) / len(all_LLM_predict)
llm_cr_time = np.sum(np.array([(1 if np.sum(np.array(l[1])) > 1 else 0) for l in all_LLM_predict])) / len(all_LLM_predict)
llm_ryt_time = np.sum(np.array([(1 if np.sum(np.array(l[2])) > 1 else 0) for l in all_LLM_predict])) / len(all_LLM_predict)
llm_ir_dosage = np.sum(np.array([np.sum(np.array(l[0])) for l in all_LLM_predict])) / (len(all_LLM_predict) * llm_ir_time)
llm_cr_dosage = np.sum(np.array([np.sum(np.array(l[1])) for l in all_LLM_predict])) / (len(all_LLM_predict) * llm_cr_time)
llm_ryt_dosage = np.sum(np.array([np.sum(np.array(l[2])) for l in all_LLM_predict])) / (len(all_LLM_predict) * llm_ryt_time)
llm_ir_freq = np.sum(np.array([get_dose_freq(l) for l in all_LLM_predict])) / (len(all_LLM_predict))
print(llm_ir_time)
print(llm_cr_time)
print(llm_ryt_time)
print('---- Medication Dosage ----')
print(f'IR - {llm_ir_dosage}')
print(f'CR - {llm_cr_dosage}')
print(f'Rytary - {llm_ryt_dosage}')
print('---- Medication Frequency ----')
print(f'IR - {llm_ir_freq}')
print('------ RL ------')
rl_ir_time = np.sum(np.array([(1 if len(l[0]) > 0 else 0) for l in med_take])) / len(med_take)
rl_cr_time = np.sum(np.array([(1 if len(l[1]) > 0 else 0) for l in med_take])) / len(med_take)
rl_ryt_time = np.sum(np.array([(1 if len(l[2]) > 0 else 0) for l in med_take])) / len(med_take)
rl_ir_dosage = np.sum(np.array([np.sum(np.array(l[0])) for l in med_take])) / (len(med_take) * rl_ir_time)
rl_cr_dosage = np.sum(np.array([np.sum(np.array(l[1])) for l in med_take])) / (len(med_take) * rl_cr_time)
rl_ryt_dosage = np.sum(np.array([np.sum(np.array(l[2])) for l in med_take])) / (len(med_take) * rl_ryt_time)
rl_freq2 = np.sum(np.array([(1 if max(len(l[0]),len(l[1]),len(l[2])) > 0 else 0) for l in med_take])) / len(med_take)
rl_ir_freq = np.sum(np.array([max(len(l[0]),len(l[1]),len(l[2])) for l in med_take])) / (len(med_take) * rl_freq2)
print(rl_ir_time)
print(rl_cr_time)
print(rl_ryt_time)
print('---- Medication Dosage ----')
print(f'IR - {rl_ir_dosage}')
print(f'CR - {rl_cr_dosage}')
print(f'Rytary - {rl_ryt_dosage}')
print('---- Medication Frequency ----')
print(f'IR - {rl_ir_freq}')


RL_trimmed_predtrajec=[p[:,np.min(np.where(r>0)[0]):] for p,r  in zip(all_RL_pred_updrs, all_real_updrs_)]


trimmed_realtrajec=[r[np.min(np.where(r>0)[0]):] for r in all_real_updrs]
trimmed_realtrajec_=[r[np.min(np.where(r>0)[0]):] for r in all_real_updrs_]

rlen = np.array([len(r) - 6 for r in trimmed_realtrajec]); 
rlen_rl = np.array([len(r) - 6 for r in trimmed_realtrajec_])

trimmed_predtrajec = [p[:,np.min(np.where(r>0)[0]):] for p,r in zip(all_pred_updrs, all_real_updrs)]

LLM_predtrajec = [p.reshape(-1) for p in all_LLM_pred_updrs]
LLM_trimmed_predtrajec=[p[:,np.min(np.where(r>0)[0]):] for p,r  in zip(all_LLM_pred_updrs, all_real_updrs)]

real_updrs_change=[r[-1]-r[0] for r in trimmed_realtrajec]
real_updrs_avg=[np.mean(r[updrs_lag:]) for r in trimmed_realtrajec]
pred_LLM_updrs_change=[(np.mean(r,axis=0)[-1]-np.mean(r,axis=0)[0]) for r in LLM_trimmed_predtrajec]
pred_LLM_updrs_avg=[(np.mean(r[:,updrs_lag:])) for r in LLM_trimmed_predtrajec]
pred_updrs_change=[(np.mean(r,axis=0)[-1]-np.mean(r,axis=0)[0])  for r in trimmed_predtrajec]
pred_updrs_avg=[(np.mean(r[:,updrs_lag:])) for r in trimmed_predtrajec]

print('---- Avg UPDRS Change ----')
print(f'Real - {np.mean(np.array(real_updrs_change))}')
print(f'Expert - {np.mean(np.array(pred_updrs_change))}')
print(f'LLM - {np.mean(np.array(pred_LLM_updrs_change))}')
print('---- Improvement Percentage / % ----')
print(f'LLM - {np.mean(np.array(pred_LLM_updrs_change)<=0)}')

print('---- Clinically significant percentage / % ----')
print(f'LLM - {np.mean(np.array(pred_LLM_updrs_change)<=-2)}')

print('---- Med UPDRS Change ----')
print(f'Real - {np.median(np.array(real_updrs_change))}')
print(f'Expert - {np.median(np.array(pred_updrs_change))}')
print(f'LLM - {np.median(np.array(pred_LLM_updrs_change))}')

LLM_adv = np.array(pred_LLM_updrs_change) - np.array(pred_updrs_change)
LLM_c = np.array(pred_LLM_updrs_change)

plt.figure(figsize=(6,3.5)) 
plt.hist(LLM_adv,
            edgecolor='black',bins=15)

plt.tight_layout()
plt_fname = f'exp_results/LLM_{fname}.png'
plt.savefig(plt_fname,dpi=300)


print('---- Improvement Percentage / % ----')
print(f'LLM - {np.mean(LLM_adv<=0)}')

print('---- Clinically significant percentage / % ----')
print(f'LLM - {np.mean(LLM_adv<=-2)}')

print('---- LLM ----')
print(np.mean(LLM_adv))
print(np.std(LLM_adv))
print((np.quantile(LLM_adv, 0.25), np.quantile(LLM_adv,.75)))
print('---- LLM ----')
print(np.mean(LLM_c))
print(np.median(LLM_c))
print(np.std(LLM_c))
print((np.quantile(LLM_c, 0.25), np.quantile(LLM_c,.75)))


def list_sum(in_list):
    sum = 0;
    [(sum := sum + i) for i in in_list]
    return sum

dose_intake = []
for tl in all_LLM_predict:
    din = [list_sum(t) for t in tl]
    dose_intake.append(din)


dose_intake_RL = []
for tl in med_take:
    din = [list_sum(t) for t in tl]
    dose_intake_RL.append(din)


# dose_intake = np.array(dose_intake) 
# dose_intake_RL = np.array(dose_intake_RL)
# plot_title = ['L-dopa IR','L-dopa CR','Rytary']
# plt.figure(figsize=(8,2.5))
# for i in range(3):
#     ax = plt.subplot(130 + i + 1) # 131, 132, 133
#     ax.grid(linestyle='--',color='#cccccc',zorder=-100)
#     ax.hist(dose_intake[:,i],bins=20,range=(0,1000),weights=np.ones_like(dose_intake[:,i])/2382,edgecolor='black',zorder=100)
#     ax.set_title(plot_title[i])
#     ax.set_xlabel('Dosage / mg')
#     ax.set_ylabel('Density')
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(8,2.5))
# for i in range(3):
#     ax = plt.subplot(130 + i + 1) # 131, 132, 133
#     ax.grid(linestyle='--',color='#cccccc',zorder=-100)
#     ax.hist(dose_intake_RL[:,i],bins=20,range=(0,750),weights=np.ones_like(dose_intake[:,i])/2382,edgecolor='black',zorder=100)
#     ax.set_title(plot_title[i])
#     ax.set_xlabel('Dosage / mg')
#     ax.set_ylabel('Density')
# plt.tight_layout()
# plt.show()


TARG_PATIENT = 3507

RATIO = 0.2 # controlling overlap
pred_updrs_mean = [np.mean(r,axis=0)[5:] for r in trimmed_predtrajec] # Initial 5 records are used as history
pred_LLM_updrs_mean = [np.mean(r,axis=0)[5:] for r in LLM_trimmed_predtrajec]
pred_RL_updrs_mean = [np.mean(r,axis=0)[5:] for r in RL_trimmed_predtrajec]
pred_updrs_std = [np.std(r,axis=0)[5:] * RATIO for r in trimmed_predtrajec]
pred_LLM_updrs_std = [np.std(r,axis=0)[5:] * RATIO for r in LLM_trimmed_predtrajec]
pred_RL_updrs_std = [np.std(r,axis=0)[5:] * RATIO for r in RL_trimmed_predtrajec]

pat_idx = np.where(patient_patno == TARG_PATIENT)[0]
pat_idx_RL = np.where(patients == TARG_PATIENT)[0]

lidx1 = np.sum(rlen[:pat_idx[0]]); lidx2 = np.sum(rlen[:pat_idx[0]+1]) 
lidx1_ = np.sum(rlen_rl[:pat_idx_RL[0]]); lidx2_ = np.sum(rlen_rl[:pat_idx_RL[0]+1])

med_information = all_LLM_predict[lidx1:lidx2]
med_information_RL = med_take[lidx1_:lidx2_]
print('---- LLM ----')
print(repr(med_information).replace(']],',']],\n'))  
print('---- RL ----')
print(repr(med_information_RL).replace(']],',']],\n'))


from colour import Color

color = ['#B8E5FA','#B2DBB9','#F7B7D2']
color = [Color(c) for c in color]; 
[c.set_luminance(0.59) for c in color]
# color[1].set_saturation(0.59)
color[1].set_luminance(0.49)
color = [c.__str__() for c in color]

if pat_idx.size > 0:
    pat_idx = pat_idx[0]
    pat_idx_RL = pat_idx_RL[0]
    plt.figure(figsize=(3.5,2.8)) 
    plt.plot(pred_updrs_mean[pat_idx],'--',label='Physician',color=color[0],marker='o',markerfacecolor='none',linewidth=1.35)
    plt.fill_between(range(pred_updrs_mean[pat_idx].shape[0]),
                     pred_updrs_mean[pat_idx] - pred_updrs_std[pat_idx],
                     pred_updrs_mean[pat_idx] + pred_updrs_std[pat_idx],
                     color=color[0],alpha=0.2)
    plt.plot(pred_RL_updrs_mean[pat_idx_RL],'-.',label='RL',color=color[1],marker='p',markerfacecolor='none',linewidth=1.35)
    plt.fill_between(range(pred_RL_updrs_mean[pat_idx_RL].shape[0]),
                     pred_RL_updrs_mean[pat_idx_RL] - pred_RL_updrs_std[pat_idx_RL],
                     pred_RL_updrs_mean[pat_idx_RL] + pred_RL_updrs_std[pat_idx_RL],
                     color=color[1],alpha=0.2)
    plt.plot(pred_LLM_updrs_mean[pat_idx],linestyle=(0,(3,1,1,1)),label='LLM',color=color[2],marker='d',markerfacecolor='none',linewidth=1.35)
    plt.fill_between(range(pred_LLM_updrs_mean[pat_idx].shape[0]),
                     pred_LLM_updrs_mean[pat_idx] - pred_LLM_updrs_std[pat_idx],
                     pred_LLM_updrs_mean[pat_idx] + pred_LLM_updrs_std[pat_idx],
                     color=color[2],alpha=0.2)
    plt.grid(linestyle='--',color='#bfbfbf',zorder=-100,alpha=0.45)
    # plt.plot(trimmed_realtrajec[pat_idx],'--',label='Real',color='black')
    plt.xlabel('Cilinal Visit')
    plt.ylabel('UPDRS Score Change') 
    plt.legend()
    plt.tight_layout()
    plt_fname = f'exp_results/pred_{TARG_PATIENT}.png'
    plt.savefig(plt_fname,dpi=300)
    print(f'Pred plot saved as {plt_fname}')
plt.show()



# real_c = np.array(real_updrs_change)
# pred_c = np.array(pred_updrs_change)
# print((np.quantile(real_c, 0.25), np.quantile(real_c,.75)))
# print((np.quantile(pred_c, 0.25), np.quantile(pred_c,.75)))
# with open(ffname,'wb') as f:
#     pickle.dump(LLM_adv,f)
# print('---- RL ----')
# print(np.mean(RL_adv))
# print(np.std(RL_adv))
# print((np.quantile(RL_adv, 0.25), np.quantile(RL_adv,.75)))

print('---- T-test result ----')
from scipy.stats import ttest_rel
print('LLM vs. physician')
print(ttest_rel(np.array(pred_LLM_updrs_change), np.array(pred_updrs_change)))
# print('LLM vs. RL')
# print(ttest_rel(np.array(pred_LLM_updrs_change), np.array(pred_RL_updrs_change)))