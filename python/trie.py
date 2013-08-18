"""
:Module: trie
:Synopsis: A quick-and-dirty implementation of the calculus of
"mistmatch string kernels"
:Author: DOHMATOB Elvis Dopgima

"""

import unittest
import numpy as np
from numpy import (array, zeros, nonzero, meshgrid,
                   outer, sqrt, ndindex)
from sklearn import svm

ALPHABET = ["\\x%02x" % x
             for x in xrange(256)]


class Trie:
    def __init__(self, label=-1, parent=0, alphabet=ALPHABET):
        self._label = label
        self._meta = {}
        self._rootpath = ""
        self._children = dict()
        self.set_parent(parent)
        self._nodecount = 0
        self._alphabet = alphabet

        if not self.is_root():
            self._rootpath += self._alphabet[label]

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
                # check against length overflow
                valid = nonzero(lmers[:,1] < len(training_data[index]))[0]
                if len(valid) == 0:
                    del self._meta[index]
                    continue
                lmers = lmers[valid,:]

                # update the mismatches of the lmers in this source string
                mistmatched_lmers = nonzero(training_data[index][lmers[:,1]] != self._label)[0]
                lmers[mistmatched_lmers,2] += 1

                # increment length of all the lmers
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
        '''
        Recursive method to grow a Trie object.

        Parameters
        ----------
        k : int
            depth limit of Trie growth
        d : int
            branching degree of Trie (noumber of children nodes to grow)
        training_data : numpy_like 2D array
            training data 
        m : int
            mismatch tolerance (so it won't be a problem if two words differ by at most m letters)
        kernel : numpy_like 2D array
            shared reference to kernel being computed
        padding : string
            marker for trie layout display 

        '''
                   
        rootpathcount = 0 # number of survived feature-kmers

        # inspect node and update its meta data
        go_ahead = self.inspect(training_data, m)

        if self.is_root():
            print "//\r\n \\"
        else:
            print padding[:-1] + '+-' + self._rootpath + ',' + str(len(self._meta)) + '/'
        padding += ' '

        # does this node survive ?
        if go_ahead:
            # is this a surving feature k-mer ?
            if k == 0:
                # compute the source weights, this far,  of the training sequences
                source_weights = array([len(lmers) for lmers in self._meta.values()])
                
                # compute the contributions of this feature k-mer to the kernel
                contributions = outer(source_weights, source_weights)
                
                # update the kernel
                kernel[meshgrid(self._meta.keys(), self._meta.keys())] += contributions
                rootpathcount += 1
            else:
                # recursively expand all children 
                for j in xrange(d):
                    # span a new child
                    _ = Trie(j, self, alphabet=self._alphabet) 

                    # expand child
                    child_padding = padding
                    if j + 1 == d:
                        child_padding += ' '
                    else:
                        child_padding += '|'
                    ga, rc = self._children[j].expand(k-1, d, training_data, m, kernel, padding=child_padding)                        

                    # update the counts 
                    if ga:
                        rootpathcount += rc
                        
        return go_ahead, rootpathcount

    def is_leaf(self):
        return self.get_nchildren() == 0

    def do_leafs(self, padding=' '):
        if self.is_root():
            print "root/\r\n \\"            
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
        '''
        Method to compute mistmatch string kernel.

        Parameters
        ----------
        k : int
            depth limit of Trie growth
        d : int
            branching degree of Trie (noumber of children nodes to grow)
        training_data : numpy_like 2D array
            training data 
        m : int (default 0)
            mismatch tolerance (so it won't be a problem if two words differ by at most m letters)
        kernel : numpy_like 2D array
            shared reference to kernel being computed
        padding : string
            marker for trie layout display 

        '''

        n = len(training_data)
        # intialize kernel
        kernel = zeros((n, n))
        
        # expand trie, constrainted by the training data, and update kernel along the way
        _, rc = self.expand(k, d, training_data=training_data, m=m, kernel=kernel)

        # normalize kernel to remove 'length bias'
        N = len(training_data)
        for x, y in ndindex((N,N)):
            if x == y:
                continue
            q = kernel[x,x]*kernel[y,y]
            if q:
                kernel[x,y] /= sqrt(q)

        import numpy as np
        np.fill_diagonal(kernel, 1.)
        print "%d out of %d (%d,%d)-mers survived"%(rc, d**k, k, m)

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


import binascii
def byte2bin(byte):
    y = bin(int(binascii.hexlify(byte), 16))[2:]
    return "0" * (8 - len(y)) + y

    return y


def bytes2bin(buf):
    return "".join([byte2bin(byte) for byte in buf])


class TestTrie(unittest.TestCase):
    def test_constructors(self):
        trie = Trie()
        self.assertEqual(trie.get_label(), -1)

    def test_compute_kernel(self):
        trie = Trie()
        k = 4
        d = 256
        m = 1

        X = np.zeros((1000, 12))
        X[:len(X) / 3, :6] = 1
        X[len(X) / 3:2 * len(X) / 3:, 3:9] = 1
        X[2 * len(X) / 3:, 6:] = 1

        Y = np.zeros(len(X))
        Y[:len(Y) / 3] = 0
        Y[len(Y) / 3:2 * len(Y) / 3:] = 1
        Y[2 * len(Y) / 3:] = 2

        X = ['GET /elvis.jpg HTTP/1.1\r\n\r\n',
             'POST /gael.jpg HTTP/1.1\r\n\r\n',
             'HEAD /michael.jpg HTTP/1.1\r\n\r\n',
             'SIP/2.0 480 Offline\r\n\r\n',
             'SIP/2.0 200 OK\r\n\r\n',
             'SIP/2.0 401 Not Authorized\r\n\r\n',
             'ekiga.net.sip > is148601.local.sip-tls: '
             '[udp sum ok] SIP, length: 404\r\n'
             'SIP/2.0 404 Not Here\r\n'
             'Via: SIP/2.0/UDP 127.0.0.1:5061;branch'
             '=z9hG4bK-4277144682;rport=5061;received'
             '=78.251.251.209\r\n'
             'From: "758198212831088"<sip:758198212831088'
             '@86.64.162.35>;tag=858043271111089947445360\r\n'
             'To: "758198212831088"<sip:758198212831088@86.64'
             '.162.35>;tag=c64e1f832a41ec1c1f4e5673ac5b80f6'
             '.c635\r\n'
             'CSeq: 1 OPTIONS\r\n'
             'Call-ID: 4085092783\r\n'
             'Server: Kamailio (1.5.3-notls (i386/linux))\r\n'
             'Content-Length: 0\r\n\r\n',
             'SIP/2.0 404 Not Here\r\n'
             'Via: SIP/2.0/UDP 127.0.0.1:5061;branch'
             '=z9hG4bK-4277144682;rport=5061;received'
             '=78.251.251.209\r\n'
             'From: "758198212831088"<sip:758198212831088'
             '@86.64.162.35>;tag=858043271111089947445360\r\n'
             'To: "758198212831088"<sip:758198212831088@86.64'
             '.162.35>;tag=c64e1f832a41ec1c1f4e5673ac5b80f6'
             '.c635\r\n'
             'CSeq: 1 PING\r\n'
             'Call-ID: 4085092783\r\n'
             'Server: Kamailio (1.5.3-notls (i386/linux))\r\n'
             'Content-Length: 0\r\n\r\n'
             ]
        X = [[int(bytes2bin(x.lower()[:12])[i:i + 4], 2)
              for i in xrange(0, 96, 4)] for x in X]
        d = 16
        Y = [0, 0, 0, 1, 1, 1, 1, 1]

        np.savetxt("/tmp/pkts.dat", X, fmt="%i")

        # kernel = trie.compute_kernel(k, d, training_data=X, m=m)
        import wrappers
        kernel = wrappers.load_boost_array("data/kernel.dat")

        print kernel
        print

        # construct SVM classifier with our mismatch string kernel
        print "Constructing SVM classifier with our mismatch string kernel .."
        svc = svm.SVC(kernel='precomputed')
        print 'Done.'
        print

        # kfd = open("kernel.txt", "a")
        # kfd.write("\n".join(["%i 0:%i %s" % (
        #                 Y[i], i + 1,
        #                 " ".join(["%i:%s" % (j + 1, kernel[i, j])
        #                           for j in xrange(kernel.shape[1])]))
        #                      for i in xrange(kernel.shape[0])]))
        # kfd.close()

        # xfd = open("test.txt", "a")
        # xfd.write("\n".join(["%i %s" % (
        #                 Y[i],
        #                 " ".join(["%i:%s" % (j + 1, X[i][j])
        #                           for j in xrange(X.shape[1])]))
        #                      for i in xrange(X.shape[0])]))
        # xfd.close()

        # d = np.ndarray(kernel.shape[0])
        # i = 8
        # for j in xrange(len(d)):
        #     d[j] = np.sqrt(kernel[i, i] - 2 * kernel[i, j] + kernel[j ,j])
        # print d

        # fit
        print "Fitting against training data .."
        svc.fit(kernel, Y)
        print "Done (fitting accuracy: %.2f" % (len(nonzero(
                    svc.predict(kernel) == Y)[0]) * 100.00 / len(Y)) + "%)."
        print

if __name__ == '__main__':
    unittest.main()
