# -*- coding: utf-8 -*-

"""
:Module: trie.py
:Synopsis: A quick-and-dirty mplementations of the calculus of "mistmatch string kernels"
:Author: DOHMATOB Elvis Dopgima
"""

import unittest
from numpy import array, zeros, nonzero, all, meshgrid, outer

class Trie:
    def __init__(self, label=-1, parent=0):
        self._label = label
        self._meta = {}
        self._rootpath = []
        self._children = dict()
        self.set_parent(parent)

        if not self.is_root():
            self._rootpath.append(label);        
    
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
            for index, jmers in meta.iteritems():
                training_data[index] = training_data[index]
                # check against length overflow
                valid = nonzero(jmers[...,1] < len(training_data[index]))[0]
                if len(valid) == 0:
                    del self._meta[index]
                    continue
                jmers = jmers[valid,...]

                # update the mismatches of the jmers in this source string
                mistmatched_jmers = nonzero(training_data[index][jmers[...,1]] != self._label)[0]
                jmers[mistmatched_jmers,2] += 1

                # increment of length of all the jmers
                jmers[...,1] += 1

                # kill all jmers which have mismatches above the threshold, m
                valid = nonzero(jmers[...,2] <= m)[0]
                if len(valid) == 0:
                    del self._meta[index]
                    continue
                jmers = jmers[valid,...]

                # update node meta data
                self._meta[index] = jmers

        return len(self._meta) > 0 
        
    def expand(self, k, d, training_data, m, kernel):
        nodecount = 1 # number of nodes in the trie
        rootpathcount = 0 # number of survived feature-kmers

        # inspect node and update its meta data
        go_ahead = self.inspect(training_data, m)

        # does this node survive ?
        if go_ahead:
            # is this a surving feature k-mer ?
            if d == 0:
                print "".join([str(x) for x in self._rootpath])   

                # compute the source weights of the training sequences
                source_weights = array([len(jmers) for jmers in self._meta.values()])
                
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
                    ga, nc, rc = self._children[j].expand(k, d - 1, training_data, m, kernel)
                    if not ga:
                        del self._children[j]
                    else:
                        # update the counts for this node
                        nodecount += nc
                        rootpathcount += rc

        return go_ahead, nodecount, rootpathcount

    def is_leaf(self):
        return self.get_nchildren() == 0

    def do_leafs(self, k, callback=None):
        if  k == 0:
            if self.is_leaf():
                if not callback is None:
                    callback(self)
                else:
                    print "".join([str(x) for x in self._rootpath])                
        else:
            for child in self._children.values():
                child.do_leafs(k - 1, callback)

    def compute_kernel(self, k, d, training_data, m=0):
        n = len(training_data)
        # intialize kernel
        kernel = zeros((n, n))
        
        # expand trie, constrainted by the training data, and update kernel along the way
        self.expand(k, d, training_data=training_data, m=m, kernel=kernel)
        
        return kernel

    def set_parent(self, parent):
        self._parent = parent

        if parent:
            self._rootpath = parent.get_rootpath()
            self._meta = parent.get_meta()
            parent.add_child(self)

    def add_child(self, child):
        self._children[child.get_label()] = child
        
    def get_label(self):
        return self._label

    def get_rootpath(self):
        return list(self._rootpath)

    def get_meta(self):
        return dict(self._meta)

    def get_children(self):
        return list(self._children)

    def get_nchildren(self):
        return len(self._children)

    def get_parent(self):
        return self._parent


class TestTrie(unittest.TestCase):
    def test_constructors(self):
        trie = Trie()
        self.assertEqual(trie.get_label(), -1)

    def test_expand_tree(self):
        trie = Trie()
        k = 6
        d = 2
        # nodecount, rootpathcount = trie.expand(d, k);
        # self.assertEqual(nodecount, (d**(k+1)-1)/(d-1))
        # self.assertEqual(rootpathcount, d**k)
        training_data = []
        training_data.append([0,1,1,1,0,1,0,1,1,0,0,0,1])
        training_data.append([1,1,0,1,0,1,0,0,1,0,0,0,0])
        training_data.append([0,1,1,1,1,0,0,0,1,0,0,0,1])
        training_data.append([1,1,1,1,0,1,0,0,1,1,1,1,0])
        training_data.append([1,1,0,1,0,1,0,0,1,0,0,0,0])
        training_data.append([0,1,0,1,0,1,0,0,1,0,0,0,1])
        training_data.append([1,1,0,1,0,1,0,0,1,0,0,0,1])

        print trie.compute_kernel(d, k, training_data=training_data, m=1)
        for seq in training_data:
            print "".join([str(x) for x in seq])

if __name__ == '__main__':
    unittest.main()
