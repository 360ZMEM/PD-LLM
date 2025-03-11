from __future__ import division
import numpy as np
import gc

def getRewardRepresentation(in_reward):
    per_45l = int(len(in_reward) * 0.45)
    in_reward = np.array(in_reward)
    in_reward_per45l = np.sort(np.array(in_reward))[:per_45l] 
    return np.mean(in_reward_per45l)

def gen_str(in_list):
    fin_str = ''
    for idx, children in enumerate(in_list):
        fin_str += f'Plan {idx+1} - {children.__repr__()} ({round(getRewardRepresentation(children.totalReward),3)}) /  '
    return fin_str[:-2] 

def format_print_list(in_list,nonlev = False):
    fin_str = ''
    for i in in_list:
        i = 'None' if i == [[],[],[]] else i
        fin_str += f'{i}' if nonlev == False else f'{round(i,0)}'
        fin_str += ' / ' if nonlev == False else ', '
    return fin_str[:-2]

class treeNode():
    def __init__(self, parent = None, depth = 0, choice = None, med_history = (None, None), nonlev = None):
        self.parent = parent
        self.numVisits = 0
        self.totalReward = [] 
        self.children = {}
        self.best_children_idx = None
        self.best_children = None
        self.bad_children = None
        self.bad_children_idx = None
        self.nonlev = nonlev
        self.depth = depth
        self.fully_expand = False 
        self.choice = choice 
        self.med_history, self.med_history_nonlev = med_history

    def del_children(self):
        del self.children
        gc.collect()
        self.children = {}

    def getDeltUpdrs(self):
        node = self
        _, pred_updrs, _, _, _, _ = node.choice
        pred_updrs = np.flip(pred_updrs, axis=1)
        pred_updrs = np.tanh(pred_updrs)
        pred_updrs = (pred_updrs + 1) * 53 / 2
        delt_pred_updrs = np.mean(pred_updrs, axis=0)[-1] - np.mean(pred_updrs, axis=0)[0]
        while node.parent != None:
            node.numVisits += 1
            node.totalReward.append(delt_pred_updrs)
            node = node.parent

            
    def getBestChild(self, best_child_num = 4):
        child_numerical_rew = []
        all_children = list(self.children.values())
        all_children_idx = list(self.children.keys())
        for children in all_children:
            child_numerical_rew.append(getRewardRepresentation(children.totalReward))
        fin_child_idx = np.argsort(np.array(child_numerical_rew))[:best_child_num]
        self.best_children = [all_children[i] for i in fin_child_idx]
        self.best_children_idx = [all_children_idx[i] for i in fin_child_idx]
        fin_child_idx = np.flip(np.argsort(np.array(child_numerical_rew)))[:best_child_num]
        self.bad_children = [all_children[i] for i in fin_child_idx]
        self.bad_children_idx = [all_children_idx[i] for i in fin_child_idx]
        return self.best_children
        
    def getRAGtext(self) -> str:  
        _, _, rag_str, _, _, bad_examples = self.choice
        gen_examples = None
        rag_str = rag_str.replace('<good_examples>', gen_str(self.best_children)).replace('<bad_examples>', gen_str(self.bad_children))
        rag_str = rag_str.replace('<medication_history>',format_print_list(self.med_history))
        rag_str = rag_str.replace('<nldopa_history>',format_print_list(self.med_history_nonlev,nonlev=True))
        return rag_str

    def __repr__(self):
        if self.choice == None:
            return 'parent'
        _, _, _, lev_input, _, _ = self.choice
        return lev_input.__repr__()
    
    def get_med_history(self):
        if self.med_history == None:
            assert self.parent.med_history != None, "The parent node must have the medication history."
            _, _, _, lev_input, _, _ = self.choice
            self.med_history = (self.parent.med_history + [lev_input])[-5:]
            self.med_history_nonlev = (self.parent.med_history_nonlev  + [round(self.nonlev,0)])[-5:]

