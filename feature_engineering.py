import os
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from tqdm import tqdm


import gensim
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation

_wnl = nltk.WordNetLemmatizer()

stop = set(stopwords.words('english'))
NDIM = 300


def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def gen_or_load_feats(feat_fn, headlines, bodies, feature_file):
    # if not os.path.isfile(feature_file):
    feats = feat_fn(headlines, bodies)
    np.save(feature_file, feats)

    return np.load(feature_file)

def gen_feats_for_test(feat_fn, headlines, bodies):
    return feat_fn(headlines, bodies)

def load_features(feat_fn, headlines, bodies):
    feats = feat_fn(headlines, bodies)
    return feats


    
#####################################
def buildWordVector(tokens, size, model, tfidf):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

def tokenizer(text):
    try:
        tokens_ = [word_tokenize(sent) for sent in sent_tokenize(text)]
        
        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent

        tokens = list(filter(lambda t: t.lower() not in stop, tokens))
        tokens = list(filter(lambda t: t not in punctuation, tokens))
        tokens = list(filter(lambda t: t not in [u"'s", u"n't", u"...", u"''", u'``', 
                                            u'\u2014', u'\u2026', u'\u2013'], tokens))
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)

        filtered_tokens = list(map(lambda token: token.lower(), filtered_tokens))

        return filtered_tokens
    except Error as e:
        print(e)

def gen_w2v_global(headlines, bodies):
    print("***************begin***************")
    x_train = []
    X_headline = []
    X_body = []
    text = headlines + bodies
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        
        x_train.append(clean_headline+clean_body)
        X_headline.append(clean_headline)
        X_body.append(clean_body)
        

    ############### tf_idf ####################
    vectorizer = TfidfVectorizer(min_df=10, max_features=10000, tokenizer=tokenizer, ngram_range=(1, 3))
    print("************* traing tf_idf ***************")
    matrix = vectorizer.fit_transform(list(text))
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    
    ################ word2vec #################
    print("************* load word2vec model ***************")
    tweet_w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    print("************* create body feature ***************")
    train_body_vecs_w2v = np.concatenate([buildWordVector(z, NDIM, model=tweet_w2v, tfidf=tfidf) for z in tqdm(X_body)])
    print("************* create headline feature ***************")
    train_headline_vecs_w2v = np.concatenate([buildWordVector(z, NDIM, model=tweet_w2v, tfidf=tfidf) for z in tqdm(X_headline)])
    print("************* scale feature ***************")
    train_body_vecs_w2v = scale(train_body_vecs_w2v)
    train_headline_vecs_w2v = scale(train_headline_vecs_w2v)
    print("************* DONE ***************")
    
    return [train_headline_vecs_w2v, train_body_vecs_w2v]    
    
    
def gen_w2v(headlines, bodies):
    print("***************begin***************")
    x_train = []
    X_headline = []
    X_body = []
    text = headlines + bodies
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        
        x_train.append(clean_headline+clean_body)
        X_headline.append(clean_headline)
        X_body.append(clean_body)


    ############### tf_idf ####################
    vectorizer = TfidfVectorizer(min_df=10, max_features=10000, tokenizer=tokenizer, ngram_range=(1, 3))
    print("************* traing tf_idf ***************")
    matrix = vectorizer.fit_transform(list(text))
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

    
    ################ word2vec #################
    tweet_w2v = Word2Vec(size=NDIM, min_count=10)
    print("************* traing word2vec ***************")
    tweet_w2v.build_vocab([x for x in x_train])
    #tweet_w2v.train([x for x in x_train], total_examples=len(x_train), epochs=5)
    tweet_w2v.train([x for x in x_train], total_examples=len(x_train))
    tweet_w2v.save('tweet_w2v_300')
    
    ############################################
    print("************* load word2vec model ***************")
    tweet_w2v = gensim.models.Word2Vec.load('tweet_w2v_300')
    print("************* create body feature ***************")
    train_body_vecs_w2v = np.concatenate([buildWordVector(z, NDIM, model=tweet_w2v, tfidf=tfidf) for z in tqdm(X_body)])
    print("************* create headline feature ***************")
    train_headline_vecs_w2v = np.concatenate([buildWordVector(z, NDIM, model=tweet_w2v, tfidf=tfidf) for z in tqdm(X_headline)])
    print("************* scale feature ***************")
    train_body_vecs_w2v = scale(train_body_vecs_w2v)
    train_headline_vecs_w2v = scale(train_headline_vecs_w2v)
    print("************* DONE ***************")
    
    return [train_headline_vecs_w2v, train_body_vecs_w2v]


def word_overlap_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        features = [
            len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        X.append(features)
    return X


def refuting_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in _refuting_words]
        X.append(features)
    return X


def polarity_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = []
        features.append(calculate_polarity(clean_headline))
        features.append(calculate_polarity(clean_body))
        X.append(features)
    return np.array(X)


def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features


def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features


def hand_features(headlines, bodies):

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                bin_count += 1
            if headline_token in clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        ############# adding ###################################
        # features = append_chargrams(features, clean_headline, clean_body, 32)
        
        
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        
        ######################## adding ##########################
        # features = append_ngrams(features, clean_headline, clean_body, 7)
        # features = append_ngrams(features, clean_headline, clean_body, 8)
        # features = append_ngrams(features, clean_headline, clean_body, 9)
        # features = append_ngrams(features, clean_headline, clean_body, 10)
        # features = append_ngrams(features, clean_headline, clean_body, 11)
        # features = append_ngrams(features, clean_headline, clean_body, 12)
        # features = append_ngrams(features, clean_headline, clean_body, 13)
        # features = append_ngrams(features, clean_headline, clean_body, 14)
        # features = append_ngrams(features, clean_headline, clean_body, 15)
        # features = append_ngrams(features, clean_headline, clean_body, 16)
        return features

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        X.append(binary_co_occurence(headline, body)
                 + binary_co_occurence_stops(headline, body)
                 + count_grams(headline, body))


    return X
