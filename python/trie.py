# -*- coding: utf-8 -*-

"""
:Module: trie
:Synopsis: A quick-and-dirty implementation of the calculus of "mistmatch string kernels"
:Author: DOHMATOB Elvis Dopgima
"""

import unittest
from numpy import array, zeros, nonzero, all, meshgrid, outer

_alphabet = [chr(x) for x in xrange(ord('A'), ord('Z'))]

class Trie:
    def __init__(self, label=-1, parent=0):
        self._label = label
        self._meta = {}
        self._rootpath = ""
        self._children = dict()
        self.set_parent(parent)
        self._nodecount = 0

        if not self.is_root():
            self._rootpath += _alphabet[label]
    
    def is_root(self):
        return (self._parent == 0)

    def inspect(self, training_data, m):
        if self.is_root():
            # populate meta data structures
            for index in xrange(len(training_data)):
                training_data[index] = array(training_data[index])
                starts = xrange(len(training_data[index]) - self.get_nchildren())
                self._meta[index] = array([(start, start, 0) for start in starts])
        else:
            meta = dict(self._meta)
            for index, lmers in meta.iteritems():
                training_data[index] = training_data[index]
                # check against length overflow
                valid = nonzero(lmers[:,1] < len(training_data[index]))[0]
                if len(valid) == 0:
                    del self._meta[index]
                    continue
                lmers = lmers[valid,:]

                # update the mismatches of the lmers in this source string
                mistmatched_lmers = nonzero(training_data[index][lmers[:,1]] != self._label)[0]
                lmers[mistmatched_lmers,2] += 1

                # increment of length of all the lmers
                lmers[:,1] += 1

                # kill all lmers which have mismatches above the threshold, m
                valid = nonzero(lmers[:,2] <= m)[0]
                if len(valid) == 0:
                    del self._meta[index]
                    continue
                lmers = lmers[valid,:]

                # update node meta data
                self._meta[index] = lmers

        return len(self._meta) > 0 
        
    def expand(self, k, d, training_data, m, kernel, padding=" "):
        rootpathcount = 0 # number of survived feature-kmers

        # inspect node and update its meta data
        go_ahead = self.inspect(training_data, m)

        if self.is_root():
            print "root\r\n \\"
        else:
            print padding[:-1] + '+-' + self._rootpath + '/'
        padding += ' '

        # does this node survive ?
        if go_ahead:
            # is this a surving feature k-mer ?
            if k == 0:
                # print "".join([_alphabet[x] for x in self._rootpath])   

                # compute the source weights of the training sequences
                source_weights = array([len(lmers) for lmers in self._meta.values()])
                
                # compute the contributions of this feature k-mer to the kernel
                contributions = outer(source_weights, source_weights)

                # update the kernel
                kernel[meshgrid(self._meta.keys(), self._meta.keys())] += contributions
                rootpathcount += 1
            else:
                # recursively expand all children of current node
                for j in xrange(k):
                    # span a new child
                    _ = Trie(j, self) 

                    # expand child
                    child_padding = padding
                    if j + 1 == k:
                        child_padding += ' '
                    else:
                        child_padding += '|'
                    ga, rc = self._children[j].expand(k-1, d, training_data, m, kernel, padding=child_padding)                        

                    # update the counts for this node
                    rootpathcount += rc
                        
        return go_ahead, rootpathcount

    def is_leaf(self):
        return self.get_nchildren() == 0

    def do_leafs(self, padding=' '):
        if self.is_root():
            print "root\r\n \\"            
        else:
            print padding[:-1] + '+-' + self._rootpath + '/'
        padding += ' '

        count = 0
        for _, child in self.get_children().iteritems():
            count += 1
            print padding + '|'
            if not child.is_leaf():
                if count == self.get_nchildren():
                    child.do_leafs(padding=padding + ' ')
                else:
                    child.do_leafs(padding=padding + '|')                    
            else:
                print padding + '+-' + child.get_rootpath()

    def compute_kernel(self, k, d, training_data, m=0):
        n = len(training_data)
        # intialize kernel
        kernel = zeros((n, n))
        
        # expand trie, constrainted by the training data, and update kernel along the way
        self.expand(k, d, training_data=training_data, m=m, kernel=kernel)
        # self.do_leafs()

        return kernel

    def set_parent(self, parent):
        self._parent = parent

        if parent:
            self._rootpath = parent.get_rootpath()
            self._meta = parent.get_meta()
            parent.add_child(self)

    def add_child(self, child):
        self._children[child.get_label()] = child
        self._nodecount += 1
    
    def get_nodecount(self):
        return self._nodecount

    def get_label(self):
        return self._label

    def get_rootpath(self):
        return self._rootpath

    def get_meta(self):
        return dict(self._meta)

    def get_children(self):
        return dict(self._children)

    def get_nchildren(self):
        return len(self._children)

    def get_parent(self):
        return self._parent


class TestTrie(unittest.TestCase):
    def test_constructors(self):
        trie = Trie()
        self.assertEqual(trie.get_label(), -1)

    def test_compute_kernel(self):
        trie = Trie()
        k = 2
        d = 7
        training_data = []
        training_data.append([0,1,1,1,0,1,0,1,1,0,0,0,1])
        training_data.append([1,1,0,1,0,1,0,0,1,0,0,0,0])
        training_data.append([0,1,1,1,1,0,0,0,1,0,0,0,1])
        training_data.append([1,1,1,1,0,1,0,0,1,1,1,1,0])
        training_data.append([1,1,0,1,0,1,0,0,1,0,0,0,0])
        training_data.append([0,1,0,1,0,1,0,0,1,0,0,0,1])
        training_data.append([1,1,0,1,0,1,0,0,1,0,0,0,1])
        

        print trie.compute_kernel(d, k, training_data=training_data, m=1)

        # for seq in training_data:
        #     print "".join([str(x) for x in seq])

if __name__ == '__main__':
    unittest.main()
