import sys
import numpy as np
import csv

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_feats_for_test, load_features
from feature_engineering import word_overlap_features, gen_w2v
from utils.dataset import DataSet, TestDataset
from utils.generate_test_splits import kfold_split, get_stances_for_folds, get_stances_bodies
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version
from xgboost import XGBClassifier
from sklearn import svm
from lstm import build_model, build_ann

import gensim
from gensim.models.word2vec import Word2Vec
import pickle
from keras.models import load_model




def generate_features_for_test(stances,dataset):
    h, b= [],[]

    for stance in stances:
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_head_w2v, X_body_w2v = gen_feats_for_test(gen_w2v, h, b)
    X_overlap = gen_feats_for_test(word_overlap_features, h, b)
    X_refuting = gen_feats_for_test(refuting_features, h, b)
    X_polarity = gen_feats_for_test(polarity_features, h, b)
    X_hand = gen_feats_for_test(hand_features, h, b)

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap, X_head_w2v, X_body_w2v]
    return X, h, b


def run_model():
    
    data = TestDataset()
    # LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
    stances, _ = get_stances_bodies(data)
    X_test, headlines, bodies = generate_features_for_test(stances, data)
    # model = pickle.load(open("model.dat", "rb"))
    # predicted = model.predict(X_test)
    model = load_model('model_fnc.h5')
    predicted = np.zeros(len(stances))
    predicted = model.predict_classes(X_test)
    predicted_label = [LABELS[int(a)] for a in predicted]
    # predicted = np.reshape(predicted, (predicted.size,))
    with open('results.csv', 'wt') as f:
        writer = csv.writer(f)
        writer.writerows(zip(headlines, bodies, predicted_label))
    # np.savetxt("predict_stance_classes.csv", predicted, delimiter=",")
    # np.savetxt("predict_stance_labels.csv", predicted_label, fmt='%5s', delimiter=",")

if __name__ == '__main__':
    run_model()