from keras.initializers import GlorotUniform
from tensorflow.keras import regularizers
import tensorflow as tf
import numpy as np
import pandas as pd
from config import *


class lstm_model():
  def __init__(self,updrs_lag=6,lev_lag=3,lev_seq_length=35,use_nonlev=False,
    weight_file="",seed=2022,layers = 1,l2=0,ksize=8): #not actually using ksize in this
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

    self.model= tf.keras.Model([self.lev_input,self.x_input],self.final_output)
    self.model.compile(loss='mse',optimizer='Adam')
    if weight_file!="":
      self.model.load_weights(weight_file)


time_interval = 30

class lev_tracker():
    def __init__(self,hours_to_peak,hold_hours,cmax_per_mg,half_life,mg,time_interval):
        self.t=0
        self.hours_to_peak=hours_to_peak;self.hold_hours=hold_hours;self.cmax_per_mg=cmax_per_mg;self.half_life=half_life
        self.mg=mg
        self.lev=[0]
        self.rate=.5**(time_interval/(self.half_life*60))   #rate for 0.5 h
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

from keras.initializers import GlorotUniform

class PPMIPolicySimulator():
  def __init__(self,inits=None):
    if inits is None:
      init1, init2, init3, init4 = GlorotUniform(), GlorotUniform(), GlorotUniform(), GlorotUniform()
    else:
      init1 = tf.constant_initializer(inits[0])
      init2 = tf.constant_initializer(inits[1])
      init3 = tf.constant_initializer(inits[2])
      init4 = tf.constant_initializer(inits[3])

    # self.state = tf.keras.Input(shape=(10,))
    # self.hidden = tf.keras.layers.Dense(10,activation='relu',kernel_initializer = init1, bias_initializer = init2)
    self.state = tf.keras.Input(shape=(11,))
    self.hidden = tf.keras.layers.Dense(10,activation='relu',kernel_initializer = init1, bias_initializer = init2)

    self._hidden=self.hidden(self.state)
    self.regimens = tf.keras.layers.Dense(73,activation='softmax', kernel_initializer= init3, bias_initializer = init4)
    self._regimens = self.regimens(self._hidden)

    self.model= tf.keras.Model(self.state,self._regimens)
    self.model.compile(loss='mse',optimizer='Adam')


import itertools # 600, 386, 280, 204, 108

meds = ['IR 100','IR 250','CR 100','CR 200','RYT 95','RYT 195'] # ,'RYT 145'
med_combos = [c for c in itertools.permutations(meds,2)]+[(m,m) for m in meds]

action_dose_lists = []

for m in med_combos:
  for dose_freq in [1,2,4]:
    med1, dose1 = m[0].split(" "); med2, dose2 = m[1].split(" ");
    dose1 = int(dose1) ; dose2= int(dose2)

    if m[0] == m[1]: #single medication given all day
      ir_seq = [dose1] * (dose_freq*int(med1=="IR"))
      cr_seq = [dose1] * (dose_freq*int(med1=="CR"))
      ryt_seq = [dose1] * (dose_freq*int(med1=="RYT"))
      action_dose_lists.append([ir_seq,cr_seq,ryt_seq])
    elif dose_freq>1: #medications are different, but need at least 2 administrations
      space = [2] if dose_freq == 4 else range(1,dose_freq)
      for first_freq in space: #Need at least one administration for each med
        if med1==med2 and (dose_freq == 4):
          ir_seq = [dose1]*(first_freq*int(med1=="IR")) + [dose2]*( (dose_freq - first_freq)*int(med1=="IR"))
          cr_seq = [dose1]*(first_freq*int(med1=="CR")) + [dose2]*( (dose_freq - first_freq)*int(med1=="CR"))
          ryt_seq = [dose1]*(first_freq*int(med1=="RYT")) + [dose2]*( (dose_freq - first_freq)*int(med1=="RYT"))
          action_dose_lists.append([ir_seq,cr_seq,ryt_seq])
        elif med1 != med2:
          ir_seq = [dose1]*(first_freq*int(med1=="IR")) + [0]*( (dose_freq - first_freq)*int(med1=="IR"))
          cr_seq = [dose1]*(first_freq*int(med1=="CR")) + [0]*( (dose_freq - first_freq)*int(med1=="CR"))
          ryt_seq = [dose1]*(first_freq*int(med1=="RYT")) + [0]*( (dose_freq - first_freq)*int(med1=="RYT"))
          if ir_seq==[]:
            ir_seq = [0]*(first_freq*int(med2=="IR")) + [dose2]*( (dose_freq - first_freq)*int(med2=="IR"))
          if cr_seq==[]:
            cr_seq = [0]*(first_freq*int(med2=="CR")) + [dose2]*( (dose_freq - first_freq)*int(med2=="CR"))
          if ryt_seq==[]:
            ryt_seq = [0]*(first_freq*int(med2=="RYT")) + [dose2]*( (dose_freq - first_freq)*int(med2=="RYT"))

          action_dose_lists.append([ir_seq,cr_seq,ryt_seq])

action_dose_lists.append([[],[],[]])


def getlist_idx(in_list):
    return (len(in_list[0]) == 0) * 1 + (len(in_list[1]) == 0) * 2 + (len(in_list[2]) == 0) * 4

#print(str(action_dose_lists).replace(']],',']],\n'))
#print(len(action_dose_lists))
action_dose_idx = np.array([getlist_idx(a) for a in action_dose_lists])

action_list_led = np.array([np.sum(np.array([np.sum(m) for m in alist])*np.array([1,0.75,0.5])) for alist in action_dose_lists])
def mask(choiceprobs,daily_led):
  for i in range(choiceprobs.shape[0]):
    choiceprobs[i,action_list_led > daily_led] = 0 #any actions that exceed daily LEDD become zero
  return choiceprobs

def simulate_RL_levday(ir_seq,cr_seq,ryt_seq,waking_hours = 17, time_interval=30):
  levcode_list=[]
  if len(ir_seq)>0: levcode_list.append(1)
  if len(cr_seq)>0: levcode_list.append(2)
  if len(ryt_seq)>0: levcode_list.append(3)

  if levcode_list==[]:
    return np.zeros(waking_hours*2+1)
  else:
    final_lev_vecs=[]
    for levcode in levcode_list:
      med_seq = [ir_seq,cr_seq,ryt_seq][levcode-1]
      dose_freq = len(med_seq)
      split_fractions=np.array(range(0,dose_freq))/dose_freq
      split_minutes=split_fractions*waking_hours*60
      med_times=np.round(split_minutes/time_interval)*time_interval
      times=list(range(0,waking_hours*60+1,time_interval))
      trackers=[]

      dose_counter = -1
      for t in times:
        [track.advance() for track in trackers]

        if t in med_times:

          dose_counter+= 1
          if levcode==1:
            trackers.append(lev_tracker(hours_to_peak=1,hold_hours=0,cmax_per_mg=10.9,half_life=1.6,
                                        mg=med_seq[dose_counter],time_interval=time_interval))
          if levcode==2:
            trackers.append(lev_tracker(hours_to_peak=1.5,hold_hours=0,cmax_per_mg=8.55,half_life=1.6,
                                        mg=med_seq[dose_counter],time_interval=time_interval))
          if levcode==3:
            trackers.append(lev_tracker(hours_to_peak=1,hold_hours=3.5,cmax_per_mg=3.4,half_life=1.6,
                                        mg=med_seq[dose_counter],time_interval=time_interval))

      lev_list=[tr.lev for tr in trackers]
      lev_matrix=np.zeros([len(lev_list),len(times)])
      for i in range(lev_matrix.shape[0]):
        lev_matrix[i,-len(lev_list[i]):]=lev_list[i]

      final_lev_vecs.append(np.sum(lev_matrix,axis=0))

    return sum(final_lev_vecs)
  
rmse_lstm = np.array([
[0.8483485,  0.57864155, 0.64660944, 0.86005974, 0.51184296], #updated
[0.85095075, 0.57050448, 0.60825269, 0.80048528, 0.53291724], #updated
[0.52403872, 0.58158697, 0.62674313, 0.68281872, 0.50734132], #updated
[0.47316233, 0.55202494, 0.62931329, 0.65286054, 0.49990356], #updated
[0.43789161, 0.52387784, 0.56201504, 0.66625484, 0.51456696], #updated
[0.44567522,  0.56420531 , 0.54999444, 0.66591554, 0.48450921]]) #updated


rmse_lstm = np.tanh(rmse_lstm)

def updrs_unnormalize(score):
    score = np.tanh(score)
    score = (score + 1) / 2
    score = score * 53
    return score

def gen_patientinf(in_str, cov_input, x_input, use_x=False, round_digit=4):
    in_gender = 'female' if cov_input.flatten().tolist()[0] == 0 else ('male' if cov_input.flatten().tolist()[0] in [0,1] else 'Unknown')
    in_str = in_str.replace('<gender>',in_gender).replace('<age>',round(cov_input.flatten().tolist()[1],round_digit).__repr__()).replace('<years_pd>',round(cov_input.flatten().tolist()[2],round_digit).__repr__()).replace('<nldopa_dose>',round(cov_input.flatten().tolist()[3],round_digit).__repr__())
    if use_x:
        for i in [6,5,4,3,2,1]: 
            in_str = in_str.replace(f'<u{7-i}>',repr(round(x_input.flatten().tolist()[i-1],round_digit)))
    return in_str

def format_print_list(in_list):
    fin_str = ''
    for i in in_list:
        fin_str += f'{i}, '
    return fin_str[:-2]


def gen_patientinf_readable(in_str, cov_input, x_input, user_input, use_x=False, med_history=None, round_digit=2): 
    in_gender = 'female' if cov_input.flatten().tolist()[0] == 0 else ('male' if cov_input.flatten().tolist()[0] in [0,1] else 'Unknown')
    in_age = round(user_input[0],1).__repr__() if user_input[0] != None else 'Unknown'
    if user_input[1] == None: in_pd = 'Unknown'
    elif user_input[1] <= 0: in_pd = 'N/A'
    else: in_pd = round(user_input[1],1).__repr__()
    in_str = in_str.replace('<gender>',in_gender).replace('<age>',in_age).replace('<years_pd>',in_pd)#.replace('<nldopa_dose>',round(user_input[2],1).__repr__())
    if use_x:
        for i in [6,5,4,3,2,1]: 
            in_str = in_str.replace(f'<u{7-i}>',repr(round(updrs_unnormalize(x_input.flatten().tolist()[i-1]),round_digit)))
    return in_str


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
    LEDD_0 = pmi.loc[pmi.PATNO == p, f'LEDD0_0'][:5] 
    SUM_DOSE = pmi.loc[pmi.PATNO == p, f'LEDDTOT_0'][:5]
    patient_medication_history_nonlev.append(list(LEDD_0 * SUM_DOSE))
    for i in [1,2,3]:
        LEDD.append(pmi.loc[pmi.PATNO == p, f'LEDD{i}_0'][:5])
        LEDD_FREQ.append(pmi.loc[pmi.PATNO == p, f'FREQ{i}_0'][:5])
    patient_medication_history.append(get_patient_medication_history(LEDD,LEDD_FREQ,SUM_DOSE))