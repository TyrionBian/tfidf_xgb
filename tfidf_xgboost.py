import pickle
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


# prepare data, get data and label
def pre_data(filepath):
    # extract label
    sentences = pd.read_csv(filepath_or_buffer=filepath,header=None, index_col=False, quoting=3, sep='\t', encoding='utf-8')
    label_str = sentences.iloc[:,0].str.split(' , ').str.get(0).str.split('label__').str.get(1)
    label = pd.to_numeric(label_str)
    # extract data
    pure_sentences = sentences.iloc[:,0].str.split(' , ').str.get(1)
    return label, pure_sentences


label, pure_sentences = pre_data(filepath = r'data/sentences.txt')

# tfidf
train_set = pure_sentences.values.tolist()
vectorizer = TfidfVectorizer(max_features=1000)
freq_term_matrix = vectorizer.fit_transform(train_set)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(freq_term_matrix)

# xgboost data,label
X = tfidf
y = label.values

xgb_model = xgb.XGBClassifier()
params = {'nthread':[8],'objective':['binary:logistic'], 'learning_rate':[0.05],'max_depth':[4,5,6],
         'min_child_weight':[7],'silent':[1],'subsample':[0.7],'colsample_bytree':[0.8],'n_estimators':[320],'missing':[-999],'seed':[10]}
clf = GridSearchCV(xgb_model, params, n_jobs=5, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc', verbose=10, refit=True)

clf.fit(X, y)

best_parameters, score, _ = max(clf.grid_scores_, key = lambda x: x[1])
print('Raw AUC score: ', score)
print(best_parameters)