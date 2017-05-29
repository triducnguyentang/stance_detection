import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features, gen_w2v
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version
from xgboost import XGBClassifier
from sklearn import svm
from lstm import build_model, build_ann

import gensim
from gensim.models.word2vec import Word2Vec
import pickle

#########################
SEQ = 8
FILENAME = "features_50/"

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_head_w2v, X_body_w2v = gen_or_load_feats(gen_w2v, h, b, FILENAME+"w2v."+name+".npy")
    X_overlap = gen_or_load_feats(word_overlap_features, h, b, FILENAME+"overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, FILENAME+"refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, FILENAME+"polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, FILENAME+"hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap, X_head_w2v, X_body_w2v]
    return X,y

if __name__ == "__main__":
    check_version()
    parse_params()

    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))


    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))
        # print("*********", X_train.shape)
        X_test = Xs[fold]
        y_test = ys[fold]

        # clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        # 79.53%
        
        # clf = XGBClassifier(learning_rate=0.05, max_depth=8, colsample_bytree=0.9, gamma=2.0, reg_lambda=1.0, n_estimators=200)
        # 86.068%
        # clf = XGBClassifier(learning_rate=0.05, max_depth=8, colsample_bytree=0.9, gamma=2.0, reg_lambda=1.0, n_estimators=1000)
        # 86.75%
        
        # clf = svm.SVC()
        # 77.87%
        
        # clf.fit(X_train, y_train)
        
        ###########   LSTM  #############################
        def convert_data(x, y, sequence_length, data_dim):
            feature = []
            for index in range(len(x) - sequence_length):
                feature.append(x[index: index + sequence_length])
            
            feature = np.array(feature)
            X_data = np.reshape(feature, (feature.shape[0], feature.shape[1], data_dim))
            return [X_data, y[SEQ:]]
        # X_train, y_train = convert_data(X_train, y_train, sequence_length=SEQ, data_dim=44)
        # X_test, y_test = convert_data(X_test, y_test, sequence_length=SEQ, data_dim=44)
        # clf = build_model(sequence_length=SEQ, data_dim=44, num_classes=4)
        clf = build_ann()
        clf.fit(X_train, y_train, batch_size=512, epochs=60, shuffle=True, validation_split=0)
        
        
        # predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        predicted = [LABELS[int(a)] for a in clf.predict_classes(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("\nScore for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf



    #Run on Holdout set and report the final score on the holdout set
    # predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    predicted = [LABELS[int(a)] for a in best_fold.predict_classes(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]
    report_score(actual,predicted)
    best_fold.save('model_fnc.h5')
    # pickle.dump(best_fold, open("model.dat", "wb"))
