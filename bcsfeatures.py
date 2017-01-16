import nltk
import time
import nltk
from nltk.corpus import stopwords
from nltk.corpus import brown
import wikipedia, string
import os
from nltk.stem.wordnet import WordNetLemmatizer
import operator
from gensim.models import Word2Vec
import numpy as np
import h5py

class ReadFeatures():
    def __init__(self,type):
        self.na_features = []
        self.pw_features = []
        self.num_mention = 0
        self.read_na(type)
        #self.read_pw(type)


    def read_na(self,type):
        with h5py.File(type+'_small-na-offsets.h5', 'r') as hf:
            for key in hf.items():
                print('List of arrays in this file: \n', str(key) )

            doc_starts = hf.get("doc_starts")
            np_doc_starts = np.array(doc_starts)
            print(len(np_doc_starts))

            ment_starts = hf.get("ment_starts")
            np_ment_starts = np.array(ment_starts)
            print(len(np_ment_starts))

        with h5py.File(type+'_small-na-feats.h5', 'r') as hf:
            for key in hf.keys():
                print('List of arrays in this file: \n', str(key) )
            feats = hf.get("feats")
            np_feats = np.array(feats)
            print(len(np_feats))
            print(np_ment_starts[-1])
            self.num_mention = len(np_ment_starts)
        for i in range(len(np_ment_starts)-1):
            self.na_features.append((i, np_feats[np_ment_starts[i]:np_ment_starts[i+1]]))

    def read_pw(self,type):
        with h5py.File( type+'_small-pw-offsets.h5', 'r') as hf:
            for key in hf.items():
                print('List of arrays in this file: \n', str(key) )

            doc_starts = hf.get("doc_starts")
            np_doc_starts = np.array(doc_starts)
            print(len(np_doc_starts))

            ment_starts = hf.get("ment_starts")
            np_ment_starts = np.array(ment_starts)
            print(len(np_ment_starts))

        with h5py.File( type + '_small-pw-feats.h5', 'r') as hf:
            for key in hf.keys():
                print('List of arrays in this file: \n', str(key) )
            feats = hf.get("feats")
            np_feats = np.array(feats)
            print(np_feats)
        num = 0
        # for i in range(self.num_mention):
        #     for j in range(i):
        #         self.pw_features.append((i,j, np_feats[np_ment_starts[num]:np_ment_starts[num+1]]))
        #         num += 1
        for i in range(len(np_ment_starts)-1):
            self.pw_features.append((i, np_feats[np_ment_starts[i]:np_ment_starts[i+1]])) # concat done as follows for mention pairs (1,0) (2,0) (2,1) (3,0) (3,1) (3,2)
            # Here (i,j) i refers to the index of single mentions used in na_features.


obj = ReadFeatures("dev") # change to "train" or "test" or "dev"

print(obj.na_features)
