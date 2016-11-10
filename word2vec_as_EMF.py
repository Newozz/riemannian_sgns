import matplotlib.pyplot as plt
import os
import csv
import pickle
import operator

import numpy as np
from numpy.linalg import svd, qr
from scipy.spatial.distance import cosine
from scipy.sparse.linalg import svds

class word2vec_as_EMF(object):
    
    def __init__(self):
        """
        Main class for working with word2vec as EMF.
        
        D -- word-context co-occurrence matrix;
        B -- such matrix that B_cw = k*(#c)*(#w)/|D|;
        C, W -- factors of matrix D decomposition;
        dictionary -- the dictionary of words from data;
        inv_dict -- inverse of dictionary.
        """
        
        self.D = None
        self.B = None
        self.C = None
        self.W = None
        self.dictionary = None
        self.inv_dict = None

    ################################# Create training corpus from raw sentences #################################
    
    def create_dictionary(self, data, r):
        """
        Create a dictionary from a list of sentences, 
        eliminating words which occur less than r times.
        """

        predictionary = {}
        for sentence in data:
            for word in sentence:
                if not predictionary.has_key(word):
                    predictionary[word] = 1
                else:
                    predictionary[word] += 1

        dictionary = {}
        idx = 0
        for word in predictionary:
            if (predictionary[word] >= r):
                dictionary[word] = idx
                idx += 1
        
        return dictionary

    def create_matrix_D(self, data, window_size=5):
        """
        Create a co-occurrence matrix D from training corpus.
        """

        dim = len(self.dictionary)
        D = np.zeros((dim, dim))
        s = window_size/2
            
        for sentence in data:
            l = len(sentence)
            for i in xrange(l):
                for j in xrange(max(0,i-s), min(i+s+1,l)):
                    if (i != j and self.dictionary.has_key(sentence[i]) 
                        and self.dictionary.has_key(sentence[j])):
                        c = self.dictionary[sentence[j]]
                        w = self.dictionary[sentence[i]]
                        D[c][w] += 1                  
        return D        
    
    def create_matrix_B(self, k):
        """
        Create matrix B (defined in init).
        """
        
        c_ = self.D.sum(axis=1)
        w_ = self.D.sum(axis=0)
        P = self.D.sum()

        w_v, c_v = np.meshgrid(w_, c_)
        B = k*(w_v*c_v)/float(P)
        return B
        
    ################################# Necessary functions #################################
    
    def sigmoid(self, X):
        """
        Sigmoid function sigma(x)=1/(1+e^{-x}) of matrix X.
        """
        Y = X.copy()
        
        Y[X>20] = 1-1e-6
        Y[X<-20] = 1e-6
        Y[(X<20)&(X>-20)] = 1 / (1 + np.exp(-X[(X<20)&(X>-20)]))
        
        return Y
    
    def MF(self, C, W):
        """
        Objective MF(D,C^TW) we want to minimize.
        """
        
        MF = 0
        X = C.T.dot(W)
        MF = self.D*np.log(self.sigmoid(X)) + self.B*np.log(self.sigmoid(-X))
        return -MF.mean()

    def grad_MF(self, C, W):
        """
        Gradient of the functional MF(D,C^TW) over C^TW.
        """
        
        X = C.T.dot(W)
        grad = self.D*self.sigmoid(-X) - self.B*self.sigmoid(X)
        return grad
    
    ################################# Alternating minimization algorithm #################################
    
    def alt_min(self, eta=1e-7, d=100, MAX_ITER=1, from_iter=0, display=0,
                init=(False, None, None), save=(False, None)):
        """
        Alternating mimimization algorithm for explicit matrix factorization.
        """
        
        # Initialization
        if (init[0]):
            self.C = init[1]
            self.W = init[2]
        else:
            self.C = np.random.rand(d, self.D.shape[0])
            self.W = np.random.rand(d, self.D.shape[1])  
            
        if (save[0] and from_iter==0):
                self.save_CW(save[1], 0)
                
        for it in xrange(from_iter, from_iter+MAX_ITER):    
            
            if (display):
                print "Iter #:", it+1
                
            gradW = (self.C).dot(self.grad_MF(self.C, self.W))
            self.W = self.W + eta*gradW
            gradC = self.W.dot(self.grad_MF(self.C, self.W).T)
            self.C = self.C + eta*gradC
                
            if (save[0]):
                self.save_CW(save[1], it+1)


    ################################# Projector splitting algorithm #################################
    
    def projector_splitting(self, eta=1e-1, d=100, MAX_ITER=1, from_iter=0, display=0, 
                            init=(False, None, None), save=(False, None)):
        """
        Projector splitting algorithm for explicit matrix factorization.
        """
        
        # Initialization
        if (init[0]):
            self.C = init[1]
            self.W = init[2]
        else:
            self.C = np.random.rand(d, self.D.shape[0])
            self.W = np.random.rand(d, self.D.shape[1]) 
            
        if (save[0] and from_iter==0):
                self.save_CW(save[1], 0)
        
        X = (self.C).T.dot(self.W)
        for it in xrange(from_iter, from_iter+MAX_ITER):
            
            if (display):
                print "Iter #:", it+1
            
            U, S, V = svds(X, d)
            S = np.diag(S)
            V = V.T
            
            self.C = U.dot(np.sqrt(S)).T
            self.W = np.sqrt(S).dot(V.T)
            
            if (save[0]):
                self.save_CW(save[1], it+1)
                     
            F = self.grad_MF(self.C, self.W)
            
            U, S_temp = qr((X + eta*F).dot(V))
            V, S = qr((X + eta*F).T.dot(U))
            V = V.T
            S = S.T
            
            X = U.dot(S).dot(V)                                  
   


    #######################################################################################
    ################################# Saving data to disk #################################
    #######################################################################################
    
    ################################# Data to Matrices #################################
    
    def data_to_matrices(self, data, r, k, to_file):
        """
        Process raw sentences, create word dictionary, matrix D and matrix B
        then save them to file.
        """
        
        self.dictionary = self.create_dictionary(data, r)
        self.D = self.create_matrix_D(data)
        self.B = self.create_matrix_B(k)
        
        sorted_dict = sorted(self.dictionary.items(), key=operator.itemgetter(1))
        dict_to_save = np.array([item[0] for item in sorted_dict])
        
        np.savez(open(to_file, 'wb'), dict=dict_to_save, D=self.D, B=self.B)
    
    ################################# Matrices to Factors #################################
 
    def load_matrices(self, from_file):
        """
        Load word dictionary, matrix D and matrix B from file.
        """
        
        matrices = np.load(open(from_file, 'rb'))
        self.D = matrices['D']
        self.B = matrices['B']
        
        dictionary = {}
        for i, word in enumerate(matrices['dict']):
            dictionary[word] = i
        self.dictionary = dictionary
        self.inv_dict = {v: k for k, v in self.dictionary.items()}
        
    def save_CW(self, to_folder, iteration):
        """
        Save C and W matrices (from iteration) to file in folder.
        """
        
        if not os.path.exists(to_folder):
            os.makedirs(to_folder)
        
        if (iteration < 10):
            pref = '00' + str(iteration)
        elif (iteration < 100):
            pref = '0' + str(iteration)
        else:
            pref = str(iteration)

        np.savez(open(to_folder+'/C'+pref+'.npz', 'wb'), C=self.C)
        np.savez(open(to_folder+'/W'+pref+'.npz', 'wb'), W=self.W) 
    
    ################################# Factors to Metrics #################################

    def load_CW(self, from_folder, iteration):
        """
        Load C and W matrices (from iteration) from folder.
        """        
           
        if not os.path.exists(from_folder):
            raise NameError('No such directory')
        
        if (iteration < 10):
            pref = '00' + str(iteration)
        elif (iteration < 100):
            pref = '0' + str(iteration)
        else:
            pref = str(iteration)
        
        C = np.load(open(from_folder+'/C'+pref+'.npz', 'rb'))['C']
        W = np.load(open(from_folder+'/W'+pref+'.npz', 'rb'))['W']
        
        return C, W
    
    def factors_to_MF(self, from_folder, to_file, MAX_ITER, from_iter=0):
        """
        Calculate MF for given sequence of matrices C and W
        and save result to file.
        """
        
        MFs = np.zeros(MAX_ITER)
        
        for it in xrange(from_iter, from_iter+MAX_ITER):
            C, W = self.load_CW(from_folder, it)
            MFs[it-from_iter] = self.MF(C, W)
        
        np.savez(open(to_file, 'wb'), MF=MFs) 
        
    ################################# Metrics to Figures #################################
    
    def load_MF(self, from_file):
        """
        Load MFs from file.
        """
        
        MFs = np.load(open(from_file), 'rb')['MF']
        
        return MFs
    
    ################################# Linquistic metrics #################################

    def word_vector(self, word, W):
        """
        Get vector representation of a word.
        """
        
        if word in self.dictionary:
            vec = W[:,int(self.dictionary[word])]
        else:
            print "No such word in dictionary."
            vec = None
            
        return vec
    
    def nearest_words(self, word, top=20, display=False):
        """
        Find the nearest words to word according to cosine similarity.
        """

        W = self.W / np.linalg.norm(self.W, axis=0)   
        if (type(word)==str):
            vec = self.word_vector(word, W)
        else:
            vec = word / np.linalg.norm(word)
 
        cosines = (vec.T).dot(W)
        args = np.argsort(cosines)[::-1]       
        
        # Display or not top nearest words
        if (display):
            for i in xrange(1, top+1):
                print self.inv_dict[args[i]], cosines[args[i]]
                
        nearest_sum = cosines[args[:top]].sum()
        
        return nearest_sum   
    