import matplotlib.pyplot as plt
from IPython import display

import numpy as np
import pandas as pd
import os
import pickle

from scipy.sparse.linalg import svds
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

from word2vec_as_EMF import word2vec_as_EMF

def load_sentences(mode='debug'):
    """
    Load training corpus sentences/
    """
    
    if (mode == 'imdb'):
        sentences = pickle.load(open('data/sentences_all.txt', 'rb'))
    elif (mode == 'debug'):
        sentences = pickle.load(open('data/sentences1k.txt', 'rb'))
    elif (mode == 'enwik9'):
        sentences = pickle.load(open('data/enwik9_sentences.txt', 'rb'))
    return sentences

def plot_MF(MFs, x=None, xlabel='Iterations', ylabel='MF'):
    """
    Plot given MFs.
    """
    
    fig, ax = plt.subplots(figsize=(15, 5))
    if not x:
        ax.plot(MFs)
    else:
        ax.plot(x, MFs)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True)
    
    
################################## Projector splitting experiment ##################################
def ps_experiment(dataset = 'debug',
                  min_count = 1000,
                  k = 5,
                  d = 100,
                  eta = 5e-6,
                  MAX_ITER = 100,
                  init = False              
                 ):
    """
    Aggregator for projector splitting experiment.
    """
    
    
    # Create model
    model = word2vec_as_EMF()
    
    
    # Data to matrices
    filename = dataset+'-'+str(min_count)+'_matrices.npz'
    if not os.path.exists(filename):
        # Load sentences from file
        sentences = load_sentences(dataset)
        # Create dictionary, matrix D, matrix B and save them to file
        model.data_to_matrices(data=sentences, r=min_count, k=k, to_file=filename)
        print "Model was created"
    else:
        # Load dictionary, matrix D, matrix B from file
        model.load_matrices(from_file=filename)
        print "Model was loaded from disk"
        
    
    # Start projector splitting from svd of SPPMI or random initialization
    start_from='RAND'
    if (init):
        # Create SPPMI matrix and decompose it via SVD
        SPPMI = np.maximum(np.log(model.D) - np.log(model.B), 0)
        u, s, vt = svds(SPPMI, k=d)
        C_svd = u.dot(np.sqrt(np.diag(s))).T
        W_svd = np.sqrt(np.diag(s)).dot(vt)
        start_from = 'SVD'
        init_ = (True, C_svd, W_svd)
    else:
        init_ = (False, None, None)
    body = dataset+'-'+str(min_count)+'_PS'+str(MAX_ITER)+'iter_from'+start_from    
        
    save = (True, body+'_factors')
    model.projector_splitting(eta=eta, d=d, MAX_ITER=MAX_ITER, init=init_, save=save)
    
    
    # Calculate MF on each iteration
    model.factors_to_MF(from_folder=body+'_factors', to_file=body+'_MF.npz', MAX_ITER=MAX_ITER)
    
    return model


################################## Word similarity experiments ##################################
def corr_experiment(model, data, from_folder, MAX_ITER=100, plot_corrs=False):
    """
    Aggregator for word similarity correlation experiment.
    """
    
    # Load dataset and model dictionary

    #wordsim353 = pd.read_csv("data/wordsim353/combined.csv")
    dataset = data.values
    model_dict = model.dictionary

    # Choose only pairs of words which exist in model dictionary
    ind1 = []
    ind2 = []
    vec2 = []
    chosen_pairs = []
    for i in xrange(dataset.shape[0]):
        word1 = dataset[i, 0].lower()
        word2 = dataset[i, 1].lower()
        if (word1 in model_dict and word2 in model_dict):
            ind1.append(int(model_dict[word1]))
            ind2.append(int(model_dict[word2]))
            vec2.append(np.float64(dataset[i, 2]))
            chosen_pairs.append((word1, word2))
            
    # Calculate correlations
    corrs = []
    vecs = []
    for it in xrange(MAX_ITER):
        vec1 = []
        C, W = model.load_CW(from_folder, it)
        for i in xrange(len(vec2)):
            vec1.append(cosine(W[:,ind1[i]], W[:,ind2[i]]))
        corrs.append(spearmanr(vec1, vec2)[0])
        vecs.append(vec1)
    corrs = np.array(corrs)  
    vecs = np.array(vecs)
    
    # Plot correlations
    if (plot_corrs):
        plots = [-corrs, vecs.mean(axis=1), vecs.std(axis=1)]
        titles = ['Correlation', 'Mean', 'Standard deviation']

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for i in xrange(3):
            ax[i].plot(plots[i])
            ax[i].set_title(titles[i], fontsize=14)
            ax[i].set_xlabel('Iterations', fontsize=14)
            ax[i].grid()
    
    return corrs, vecs, vec2, chosen_pairs

def plot_dynamics(vecs, vec2, n=5, MAX_ITER=100):
    """
    Plot how the distances between pairs change with each n
    iterations of the optimization method.
    """
    
    for i in xrange(MAX_ITER):
        if (i%n==0):
            plt.clf()
            plt.xlim([-0.01, 1.2])
            plt.plot(vecs[i], vec2, 'ro', color='blue')
            plt.grid()
            display.clear_output(wait=True)
            display.display(plt.gcf())
    plt.clf()
    
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for i in xrange(2):
        ax[i].set_xlim([-0.01, 1.2])
        ax[i].plot(vecs[-i], vec2, 'ro', color='blue')
        ax[i].set_title(str(i*MAX_ITER)+' iterations')
        ax[i].set_xlabel('Cosine distance', fontsize=14)
        ax[i].set_ylabel('Assesor grade', fontsize=14)
        ax[i].grid()
    
    
def dist_change(vecs, vec2, chosen_pairs, n=5, dropped=True, from_iter=0, to_iter=-1):
    """
    Get top pairs which change distance between words the most.
    """
    
    vecs_diff = vecs[to_iter, :] - vecs[from_iter, :]
    args_sorted = np.argsort(vecs_diff)

    for i in xrange(n):
        if (dropped):
            idx = args_sorted[i]
        else:
            idx = args_sorted[-1-i]
        print "Words:", chosen_pairs[idx]
        print "Assesor score:", vec2[idx]
        print "Distance change:", vecs[from_iter, idx], '-->', vecs[to_iter, idx]
        print '\n'
    
    
def nearest_words_from_iter(model, word, from_folder, top=20, display=False, it=1):
    """
    Get top nearest words from some iteration of optimization method.
    """
    C, W = model.load_CW(from_folder=from_folder, iteration=it)

    model.W = W.copy()
    model.C = C.copy()

    nearest_sum = model.nearest_words(word, top, display)
    
    return nearest_sum

################################## Analogical reasoning experiments ##################################

def argmax_fun(W, indices, argmax_type='levi'):
    """
    cosine: b* = argmax cosine(b*, b - a + a*) 
    levi: b* = argmax cos(b*,a*)cos(b*,b)/(cos(b*,a)+eps)
    """
    
    if (argmax_type == 'levi'):
        W = W / np.linalg.norm(W, axis=0)
        words3 = W[:, indices]
        cosines = ((words3.T).dot(W) + 1) / 2
        obj = (cosines[1] * cosines[2]) / (cosines[0] + 1e-3)
        pred_idx = np.argmax(obj)
        
    elif (argmax_type == 'cosine'):
        words3_vec = W[:, indices].sum(axis=1) - 2*W[:, indices[0]]
        W = W / np.linalg.norm(W, axis=0)
        words3_vec = words3_vec / np.linalg.norm(words3_vec)
        cosines = (words3_vec.T).dot(W)
        pred_idx = np.argmax(cosines)
        
    return pred_idx

def analogical_reasoning(model, dataset, from_folder, it=0):
    """
    Calculate analogical reasoning accuracy for given dataset.
    """
    dic = model.dictionary
    
    _, W = model.load_CW(from_folder, iteration=it)
    W = W / np.linalg.norm(W, axis=0)

    good_sum = 0
    miss_sum = 0

    for words in dataset.values:

        a, b, a_, b_ = words

        if (a in dic and b in dic and a_ in dic and b_ in dic):

            indices = [dic[a], dic[b], dic[a_]]
            
            words3 = W[:, indices]
            cosines = ((words3.T).dot(W) + 1) / 2
            obj = (cosines[1] * cosines[2]) / (cosines[0] + 1e-3)
            pred_idx = np.argmax(obj)
            
            if (model.inv_dict[pred_idx] == b_):
                good_sum += 1
        else: 
            miss_sum += 1

    # calculate accuracy
    acc = (good_sum) / float(dataset.shape[0]-miss_sum)
    
    return acc, miss_sum

def AR_experiment(model, dataset, from_folder, MAX_ITER=100, step_size=5, plot_accs=False):
    """
    Aggregator for analogical reasoning accuracy experiment.
    """
    
    # Calculate accuracies 
    accs = []
    num_points = MAX_ITER/step_size + 1
    for i in xrange(num_points):
        acc, miss = analogical_reasoning(model, dataset, from_folder, i*step_size)
        accs.append(acc)
    accs = np.array(accs)
    
    # Plot accuracies
    if (plot_accs):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(np.arange(num_points)*step_size, accs)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_xlabel('Iterations', fontsize=14)
        ax.grid()    
        
    return accs, miss