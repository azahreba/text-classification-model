
# coding: utf-8

from __future__ import division

import re
import pandas as pd
import numpy as np
import scipy
import math
import json
import pickle
from collections import Counter
from bs4 import BeautifulSoup
import sklearn
import xgboost as xgb
import nltk

import sys
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='english', ignore_stopwords=True)
tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-z]{3,}')


with open(sys.argv[1]) as f:
    text = f.read()

# In[ ]:

model_path = '/home/tomcat8/ml/models/'
model_type = 'xgb'
model_name = '2classes_xgb_tfidf_v1.model'

bst = xgb.Booster(model_file=model_path + model_type +'/' + model_name)


vectorizer_path = '/home/tomcat8/ml/vectorizers/'
vectorizer_type = 'tfidf'
vectorizer_name = 'vectorizer2.pk'

with open(vectorizer_path + vectorizer_type + '/' + vectorizer_name, 'rb') as fin:
    vectorizer = pickle.load(fin)


lable_dict = {'Approved': 1, 'Other': 0}

inv_lable_dict = dict(zip(lable_dict.values(), lable_dict.keys()))


text = text.decode('utf8', 'ignore')

texts = [text]

texts = map(lambda x: stemmer.stem(x), texts)
texts = map(lambda x: tokenizer.tokenize(x), texts)
texts = map(lambda x: re.sub(r'\s\d+\s', ' ', ' '.join(x)), texts)

X = np.array(texts)
X = vectorizer.transform(X)
X.shape

DX = xgb.DMatrix(X)
result = bst.predict(DX)

result = inv_lable_dict[result[0]]

f = open(sys.argv[2], 'w')

sys.stdout.write(result)
f.write(result)    

f.close()

