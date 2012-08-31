# -*- coding: utf-8 -*-

'''
:Module: digits_example
:Synopsis: Classification of hand-written digits usng SVM with mismatch string kernel
:Author: DOHMATOB Elvis Dopgima
'''

import sys
from trie import *
from sklearn import datasets
from numpy.random import shuffle
from numpy import savetxt, nonzero
from sklearn import svm

if __name__ == '__main__':
    print "\r\n\t\t --[ %s (C)  DOHMATOB Elvis Dopgima ]--\r\n"%sys.argv[0]

    # some useful settings (algorithmic complexity is O(nsamples*(dk)^m))
    d = 16 # alphabel size (branching degree)
    k = 6  # trie depth
    m = 1  # mismatch tolerance
    nsamples = 500

    # load hand-written digits dataset
    digits = datasets.load_digits()
    
    # prepare training samples
    X = digits.data[:nsamples]
    Y = digits.target[:nsamples]
    
    # instantial Trie object
    trie = Trie()

    # use Trie object to compute mismatch string kernel for classifier
    print
    print "Computing mismatch string kernel (this may take a while) .." 
    kernel =  trie.compute_kernel(k, d, training_data=X, m=m)
    print
    print "Done."
    print "Kernel:\r\n", kernel
    print

    # construct SVM classifier with our mismatch string kernel
    print "Constructing SVM classifier with our mismatch string kernel .."
    svc = svm.SVC(kernel='precomputed')
    print 'Done.'
    print 

    # fit
    print "Fitting against training data .."
    svc.fit(kernel, Y)
    print "Done (fitting accuracy: %.2f"%(len(nonzero(svc.predict(kernel) == Y)[0])*100.00/nsamples) + "%)."
    print 

