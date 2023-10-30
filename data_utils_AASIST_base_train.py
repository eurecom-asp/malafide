import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
import librosa
from random import randrange
import random
import os

___author__ = "Hemlata Tak, Jee-weon Jung, Antani"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com, antani@eurecom.fr"


def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    utt_IDs=[]
    label_list=[]
    
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            
            # key,label = line.strip().split(" ")
            #_,key =path.split("/")
            _, key, _, _, label = line.strip().split()  # ugly crap to make it fit to our protocols
        
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            # key, label = line.strip().split(" ")
            _, key, _, _, label = line.strip().split()  # ugly crap to make it fit to our protocols
            # _,utt_id =key.split("/")
            utt_id = key
            utt_IDs.append(utt_id)
            file_list.append(key)
            label_list.append(label)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta,file_list,utt_IDs,label_list
        
    
    else:
        for line in l_meta:
            # key,label = line.strip().split(" ")
            _, key, _, _, label = line.strip().split()  # ugly crap to make it fit to our protocols
            #_,utt_id =key.split("/")
            #utt_ID.append(utt_id)
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_train(Dataset):
    def __init__(self,list_IDs, labels, base_dir):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600 # take ~4 sec audio (64600 samples)
        self.fs  = 16000
     

    def __len__(self):
        return len(self.list_IDs)


    def __getitem__(self, index):
        key = self.list_IDs[index]
        # X, _ = librosa.load(self.base_dir+key+'.flac', sr=16000)
        X, _ = librosa.load(os.path.join(self.base_dir, f'{key}.flac'), sr=16000)
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[key] 
        return x_inp, target

class Dataset_dev(Dataset):
    def __init__(self,list_IDs, labels, base_dir):
        self.list_IDs = list_IDs
        self.labels = labels
        #self.utt_id = UTT_ID
        self.base_dir = base_dir
        self.cut = 64600 # take ~4 sec audio (64600 samples)
        self.fs  = 16000
        
        

    def __len__(self):
        return len(self.list_IDs)


    def __getitem__(self, index):
        key = self.list_IDs[index]
        #id_utt=self.utt_id[index]
        # X, _ = librosa.load(self.base_dir+key+'.flac', sr=16000)
        X, _ = librosa.load(os.path.join(self.base_dir, f'{key}.flac'), sr=16000)
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[key] 
        return x_inp, target
        
        
class Dataset_eval(Dataset):
    def __init__(self,list_IDs,labels, UTT_ID, base_dir, att_list):
        self.list_IDs = list_IDs
        self.utt_id = UTT_ID
        self.labels = labels
        self.base_dir = base_dir
        self.att_list = att_list
        self.cut = 64600 # take ~4 sec audio (64600 samples)
        self.fs  = 16000
        
        

    def __len__(self):
        return len(self.list_IDs)


    def __getitem__(self, index):
        key = self.list_IDs[index]
        id_utt = self.utt_id[index]
        att = self.att_list[index]
        # X, _ = librosa.load(self.base_dir+key+'.flac', sr=16000)  # for debug # wait, you didn't say "dubug" here?
        #X, _ = librosa.load(self.base_dir+key, sr=16000)
        X, _ = librosa.load(os.path.join(self.base_dir, f'{key}.flac'), sr=16000)
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[key]
        return x_inp,target, id_utt,att        


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = librosa.load(self.base_dir+'flac/'+key+'.flac', sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key
        

class Dataset_ASVspoof2021_eval(Dataset):
	def __init__(self, list_IDs, base_dir):
            '''self.list_IDs	: list of strings (each string: utt key),
               '''
               
            self.list_IDs = list_IDs
            self.base_dir = base_dir
            

	def __len__(self):
            return len(self.list_IDs)


	def __getitem__(self, index):
            self.cut=64600 # take ~4 sec audio (64600 samples)
            key = self.list_IDs[index]
            X, fs = librosa.load(self.base_dir+'flac/'+key+'.flac', sr=16000)
            
            X_pad = pad(X,self.cut)
            x_inp = Tensor(X_pad)
           
            return x_inp,key
