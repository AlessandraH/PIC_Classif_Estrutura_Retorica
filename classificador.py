# -*- coding: utf-8 -*-

import json
import numpy as np
import time

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import cross_val_predict
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


def to_sentences(abstracts, senteces_max=None):
    sentences = []
    labels = []
    abstracts_sentences = []
    abstracts_labels = []
    ids = []

    for id, abstract in enumerate(abstracts):
        if senteces_max and len(abstract) > senteces_max:
            continue

        tmp_sentences = []
        tmp_labels = []

        for label, text in abstract:
            sentences.append(text)
            labels.append(label)

            tmp_sentences.append(text)
            tmp_labels.append(label)
            ids.append(id)

        abstracts_sentences.append(tmp_sentences)
        abstracts_labels.append(tmp_labels)

    assert (len(sentences) == len(labels))
    assert (len(abstracts_sentences) == len(abstracts_labels))

    return sentences, labels, abstracts_sentences, abstracts_labels, ids


def loadFromJson(file):
    data = []
    with open(file, 'r') as f:
        data = json.load(f, encoding='cp1252')
    return to_sentences(data)


def abstracts_to_sentences(abstracts, labels):
    ret = []
    ret_prev = []
    ret_next = []
    ret_labels = []
    ret_pos = []
    abstracts_idx = []

    for i, (sentences_labels, sentences) in enumerate(zip(labels, abstracts)):
        for j, (label, sentence) in enumerate(zip(sentences_labels, sentences)):
            ret.append(sentence)
            ret_pos.append(j)
            ret_labels.append(label)
            abstracts_idx.append(i)

            if j - 1 >= 0:
                ret_prev.append(sentences[j - 1])
            else:
                ret_prev.append("")

            if j + 1 < len(sentences):
                ret_next.append(sentences[j + 1])
            else:
                ret_next.append("")

    return ret, ret_prev, ret_next, ret_pos, ret_labels, abstracts_idx


def classificador(corpus, carac, ngrama, chi, k, algo, crossval):
    print(time.asctime(time.localtime(time.time())))

    print("Lendo arquivo...")
    _, _, data, labels, _ = loadFromJson('corpus/' + corpus + '.json')
    X_sentences, X_prev, X_next, X_pos, Y_sentences, _ = abstracts_to_sentences(data, labels)

    vectorizer = TfidfVectorizer(ngram_range=(1, ngrama))
    X_sentences = vectorizer.fit_transform(X_sentences)
    if carac:
        X_prev = vectorizer.fit(X_prev)
        X_next = vectorizer.fit(X_next)

    if chi:
        print("Aplicando chi")
        selector = SelectKBest(chi2, k=k)
        X_sentences = selector.fit_transform(X_sentences, Y_sentences)
        if carac:
            X_prev = selector.transform(X_prev)
            X_next = selector.transform(X_next)

    print("Adicionando ao treino prev e next")
    if carac:
        X_sentences = hstack([X_sentences, X_prev, X_next, np.expand_dims(np.array(X_pos), -1)])
    else:
        X_sentences = hstack([X_sentences, np.expand_dims(np.array(X_pos), -1)])


    print("Inicializando classificador...")
    if algo == 0:
        clf = LinearSVC(dual=False, tol=1e-3)
    elif algo == 1:
        clf = neighbors.KNeighborsClassifier(n_neighbors=15, weights='uniform')
    elif algo == 2:
        clf = MultinomialNB()
    else:
        clf = DecisionTreeClassifier(random_state=0)
    clf = clf.fit(X_sentences, Y_sentences)

    print("Predicao")
    pred = cross_val_predict(clf, X_sentences, Y_sentences, cv=crossval)

    print("Classification_report:")
    print(classification_report(Y_sentences, pred))
    print("Accuracy_score:")
    print(metrics.accuracy_score(Y_sentences, pred))
    print(confusion_matrix(Y_sentences, pred))

    print(time.asctime(time.localtime(time.time())))


def main():
    corpus = str(raw_input("Digite o nome do corpus (dev, data, output366, output466, output832): "))
    carac = bool(raw_input("Digite \n [0] - TfIdf da sentenca e posicao da sentenca; \n [1] - TfIdf da sentenca, TfIdf das sentencas anterior e posterior e posicao da sentenca. \n"))
    ngrama = int(raw_input("Digite o valor para n-grama: "))
    chi = bool(raw_input ("Realizar selecao com chi2 ([0] - Nao; [1] - Sim)? "))
    k = int(raw_input("Digite o valor de k: "))
    algo = int(raw_input("Digite o algoritmo:\n[0] SVM;\n[1] KNN;\n[2] NB;\n[3] DT.\n"))
    classificador(corpus, carac, ngrama, chi, k, algo, 10)


main()