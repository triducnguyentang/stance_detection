import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats, load_features
from feature_engineering import word_overlap_features, gen_w2v
from utils.dataset import DataSet, Test_dataset
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version
from xgboost import XGBClassifier
from sklearn import svm
from lstm import build_model, build_ann

import gensim
from gensim.models.word2vec import Word2Vec
import pickle
from keras.models import load_model




def generate_features(stances,dataset):
    h, b, y = [],[],[]
    
    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_head_w2v, X_body_w2v = load_features(gen_w2v, h, b)
    X_overlap = load_features(word_overlap_features, h, b)
    X_refuting = load_features(refuting_features, h, b)
    X_polarity = load_features(polarity_features, h, b)
    X_hand = load_features(hand_features, h, b)

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap, X_head_w2v, X_body_w2v]
    return X, y


def run_model():
    
    d = DataSet()
    score = ['agree', 'disagree', 'discuss', 'unrelated']
    folds,hold_out = kfold_split(d,n_folds=1)
  
  
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)


    X_test, y_test = generate_features(hold_out_stances,d)
    # X_test = X_test
    # y_test = y_test
    # model = pickle.load(open("model.dat", "rb"))
    # predicted = model.predict(X_test)
    model = load_model('model_fnc.h5')
    predicted = model.predict_classes(X_test)
    for (y, p) in zip(y_test, predicted):
        print('Human: ' + score[y], '\tAI: ' + score[p])
    # print([print(('Human: ' + score[y], 'AI: ' + score[p])) for (y, p) in zip(y_test, predicted)])

if __name__ == '__main__':
    run_model()