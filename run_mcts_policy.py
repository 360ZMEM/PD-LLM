import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import subprocess
import time
import numpy as np
import gc
from io import StringIO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--parallel_num", type=int, default=5, help="The number of subprocesses for MCTS simultaneous prediction. Change this to 5 for PCs with 32GB memory (~13hours) and 2 for PCs with 16GB memory (~32hours).")
args, unknown = parser.parse_known_args()


original_stdout = sys.stdout

SIM_RUN = args.parallel_num
PATIENT_NUM = 399
proc = [0] * SIM_RUN
start_time = time.time()

for i in range(SIM_RUN):
    args = ['python',f'{BASE_DIR}/mcts_policy.py','--patient',str(i),'--quiet']
    proc[i] = subprocess.Popen(args) #, stdout=subprocess.PIPE
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f'Patient {i} START RUNNING! time {formatted_time}')
sys.stdout = StringIO()

patient_idx = SIM_RUN
proc_alive = [1] * SIM_RUN
while True:
    time.sleep(0.05)

    for i in range(SIM_RUN):
        poll = proc[i].poll()
        proc_alive[i] = poll is None
        if not proc_alive[i]:
            if patient_idx < PATIENT_NUM:
                args = ['python',f'{BASE_DIR}/mcts_policy.py','--patient',str(patient_idx),'--quiet']
                proc[i] = subprocess.Popen(args)  # , stdout=subprocess.PIPE
                proc_alive[i] = 1
                formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                sys.stdout = original_stdout
                time_elapse = time.time() - start_time
                print(f'Patient {patient_idx} START RUNNING! time {formatted_time} time_elapse {round(time_elapse/60,2)} minutes')
                patient_idx += 1
                sys.stdout = StringIO()
    gc.collect()
    if np.sum(np.array(proc_alive)) == 0:
        break
    