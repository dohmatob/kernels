"""
:Module: mismatch_string_kernel
:Author: dohmatob elvis dopgima <gmdopp@gmail.com>
:Synopsis: Implementation of "Mismatch String Kernels"
(Reference http://bioinformatics.oxfordjournals.org/content/20/4/467.short)

"""

from trie import MismatchTrie, normalize_kernel
import wrappers

import numpy as np
from sklearn.svm import SVC
import re


def unique_kmers(x, k):
    x = list(x)

    ukmers = []
    offset = 0
    seen_kmers = []
    for offset in xrange(len(x) - k + 1):
        kmer = x[offset:offset + k]

        if kmer in seen_kmers:
            continue
        else:
            seen_kmers.append(kmer)

        count = 1
        for _offset in xrange(offset + 1, len(x) - k + 1):
            if np.all(x[_offset:_offset + k] == kmer):
                count += 1

        ukmers.append((kmer, count))

    return ukmers


class MismatchStringKernel(MismatchTrie):
    """
    Python implementation of Mismatch String Kernels. See reference above.

    Parameters
    ----------
    l: intt, optional (default None)
        size of alphabet. Example of values with a natural interpretation:
        2: for binary data
        256 for data encoded as strings of bytes
        20: for protein data (bioinformatics)
        If l is not provided, then the model should be specified explicitly
        when calling the `fit` method
    k: int, optional (default None)
        the k in 'k-mer' and 'k-gram'.
        If k is not provided, then the model should be specified explicitly
        when calling the `fit` method
    m: intt, optional (default None)
        maximum number of mismatches for 2 k-grams/-mers to be considered
        'similar'. Normally small values of m should work well, plus the
        complexity the algorithm is exponential in m.
        For example, if 'ELVIS' and '3LVIS' are dissimilar
        if m = 0, but similary if m = 1.
        If m is not provided, then the model should be specified explicitly
        when calling the `fit` method
    **kwargs: dict, optional (default empy)
        optional parameters to pass to `tree.MismatchTrie` instantiation

    Attributes
    ----------
    `kernel_`: 2D array of shape (n_sampled, n_samples)
        estimated kernel
    `n_survived_kmers`:
        number of leafs/k-mers that survived trie traversal
     `svc_`: fitted `SVC` object
        in case train labels `Y` were are provided during fitting, then
        and SVC (with precomputed kernel) will be fitted against the kernel

     Examples
    --------
    >>> mmsk = MismatchStringKernel(model=wrappers.read_mmsk_model(
    ... '/tmp/leafs.txt', '/tmp/kernel.txt'))

    Notes
    -----
    XXX Such code should be written exclusively in C/C++ STL

    """

    def __init__(self, l=None, k=None, m=None, verbose=0, **kwargs):

        kwargs['verbose'] = verbose
        self.verbose = verbose

        if not None in [l, k, m]:
            # don't be too chatty
            if not "display_summerized_kgrams" in kwargs:
                kwargs["display_summerized_kgrams"] = True

            # invoke trie.MismatchTrie constructor
            MismatchTrie.__init__(self, **kwargs)

            # sanitize alphabet size
            if l < 2:
                raise ValueError(
                    "Alphabet too small. l must be at least 2; got %i" % l)

            # sanitize kernel parameters (k, m)
            if 2 * m > k:
                raise ValueError(
                    ("You provided k = %i and m = %i. m is too big (must"
                     "be at ""must k / 2). This doesn't make sense.") % (k, m))

            self.l = l
            self.k = k
            self.m = m

    def fit(self, X, Y=None, **kwargs):
        """
        Fit Mismatch String Kernel on data.

        Parameters
        ----------
        X: 2D array of shape (n_samples, n_features), or tuple (
        l, k, m, leafs, kernel), where:
            `l`, `k`, and `m`: the parameters of the (precomputed) trie;
            `leafs`: a dictionary of leafs of the trie, of the form
                {'[0][14][10][15][10][0]': {146: 1},
                '[0][8][0][0][0][1]': {36: 3, 40: 2},
                '[11][0][0][2][0][0]': {48: 1},
                '[12][0][0][0][5][15]': {128: 1},
                '[13][14][7][7][6][1]': {56: 1},
                '[3][0][2][12][15][6]': {63: 1, 74: 1},
                '[4][0][6][0][0][7]': {88: 1},
                '[6][15][0][0][6][0]': {44: 1},
                '[6][5][6][12][6][4]': {32: 1},
                '[8][5][15][6][4][6]': {36: 1, 40: 1}},
                where each key corresponds to the full_label/rootpath of a leaf
                node (beta), and each value is a dict whose keys are the
                indices of the train sample point that are beta-similar, the
                values of this subdicts being weight of the leaf beta in the
                corresponding training sample point
            `kernel`: a 2D array of shape (n_samples, n_samples), representing
            precomputed kernel
            training data/model for the kernel
        Y: 1D array of length n_samples, optional (default None)
            labels of the training samples `X`. If Y is specified, then an
            SVC will be fitted for predicting test values

        Returns
        -------
        self: `MismatchStringKernel` object
            fitted kernel instance

        Notes
        -----
        The complexity of the algorithm used to compute the kernel is
        \[\texit{O}((length of 1 sample)(number of samples)^2l^mk^m)\].

        """

        if isinstance(X, tuple):
            assert len(X) == 5, "Invalid model."
            self.l, self.k, self.m, self.leaf_kgrams_, self.kernel_ = X
            # XXX sanitize the types and shapes of self.l, self.j, self.m,
            # self.leaf_kgrams, and self.kernel_
        else:
            # traverse/build trie proper
            for x in ['l', 'k', 'm']:
                if not hasattr(self, x):
                    raise RuntimeError(
                        ("'%s' not specified during object initialization."
                         "You must now specify complete model (tuple of l, "
                         "k, m, leafs, and, kernel).") % x)
            self.kernel_, _, _ = self.traverse(
                X, self.l, self.k, self.m, **kwargs)

            # normalize kernel
            self.kernel_ = normalize_kernel(self.kernel_)

            # gatther up the leafs
            self.leaf_kgrams_ = dict((leaf.full_label,
                                      dict((index, len(kgs)) for index, kgs
                                           in leaf.kgrams.iteritems()))
                                     for leaf in self.leafs())

        # fit SVC
        if not Y is None:
            self.log('Fitting SVC...')
            self.svc_ = SVC(kernel='precomputed').fit(self.kernel_, Y)
            self.log("...done; %s\r\n" % self.svc_)

        return self

    def _k_value(self, x):
        """
        Computes the inner product between the support vectors and
        the test vector x.

        """

        kv = np.zeros(len(self.svc_.support_))

        # scan through all (unique) k-grams/mers of x
        for kmer, count1 in unique_kmers(x, self.k):
            # vec2str
            kmer = ('[%s]' * self.k) % tuple(kmer)
            if kmer in self.leaf_kgrams_:
                for j in xrange(len(self.svc_.support_)):
                    if self.svc_.support_[j] in self.leaf_kgrams_[kmer]:

                        # retrieve the number of times this kmer appears
                        # in jth support vector
                        kgrams = self.leaf_kgrams_[kmer][self.svc_.support_[j]]
                        count2 = len(kgrams) if isinstance(kgrams,
                                                           list) else kgrams

                        # update kv entry for jth support vector
                        kv[j] += np.exp(-(count1 + count2))

        # scaling: correct for variance
        kv = kv / kv.var()

        # done
        return kv

    def _predict(self, x):
        """
        Predicts the label of a test string x. The multi-class method used
        is one-against-one, in which n_class * (n_class - 1) / 2 binary
        classifies are created.

        """

        # fitted ?
        if not hasattr(self, 'kernel_'):
            raise RuntimeError("fit(...) method not yet invoked.")

        if not hasattr(self, 'svc_'):
            raise RuntimeError(
                ("Parameter `Y` (train labels) not specfied during call "
                 "to fit(...) method. We can't do prediction"))

        n_class = len(self.svc_.classes_)
        start = np.cumsum([0] + [self.svc_.n_support_[j]
                                 for j in xrange(n_class - 1)])

        # compute inner product between support vectors and test vector x
        kvalue = self._k_value(x)

        # do one-against-one multi-class prediction
        p = 0
        vote = np.zeros(n_class)  # votes for each class
        for i in xrange(n_class):
            for j in xrange(i + 1, n_class):
                s = 0  # score

                # pointers
                si = start[i]
                sj = start[j]
                ci = self.svc_.n_support_[i]
                cj = self.svc_.n_support_[j]

                # dual coefficients
                coeff1 = self.svc_.dual_coef_[j - 1]
                coeff2 = self.svc_.dual_coef_[i]

                # do voting
                for k in xrange(ci):
                    s += coeff1[si + k] * kvalue[si + k]

                for k in xrange(cj):
                    s += coeff2[sj + k] * kvalue[sj + k]

                s -= self.svc_.intercept_[p]

                if s > 0:
                    vote[i] += 1
                else:
                    vote[j] += 1

                p += 1

        # and the winner is
        return self.svc_.classes_[np.argmax(vote)]

    def predict(self, x):
        """
        Predicts the label of the test vector or list of test vectors, x

        """

        x = np.array(x, dtype=np.int)
        if x.ndim == 1:
            return self._predict(x)
        else:
            assert x.ndim == 2
            return [self.predict(_x) for _x in x]


# demo
if __name__ == '__main__':
    import os
    import sys
    import matplotlib.pyplot as plt

    print "%s++[%s (c) d0hm4t06 3LV15 d0p91m4]++%s" % (
        '\t' * 5, os.path.basename(sys.argv[0]), "\r\n" * 2)

    # load data
    X = np.loadtxt(os.path.abspath(os.path.join(
                os.path.dirname(os.path.dirname(sys.argv[0])),
                'data/hack_data.txt')), dtype=np.int)
    Y = np.loadtxt(os.path.abspath(os.path.join(
                os.path.dirname(os.path.dirname(sys.argv[0])),
                'data/hack_data_labels.txt')), dtype=np.int)

    # prepare for visualization
    plt.gray()  # only gray scale
    plt.ion()  # fall into interative mode so that we don' block on show(...)

    # plot covariance matrix of train data
    plt.imshow(normalize_kernel(np.dot(X, X.T)))
    plt.axis('off')  # we don't need axes
    plt.savefig("normalized_covariance.png", bbox_inches="tight", dpi=200,
                facecolor="k", edgecolor="k")

    # initialize plot for kernel (will be updated iteratively)
    kern_fig = plt.figure()
    plt.axis('off')  # we don't need axes
    plt.imshow(np.zeros((X.shape[0], X.shape[0])))
    plt.show()

    def _update_kernel_plot(kern):
        """
        Callback function to update kernel plot.

        """

        # clear figure()
        plt.clf()

        # re-draw the figure
        plt.axis('off')
        plt.imshow(normalize_kernel(kern))
        kern_fig.canvas.draw()

    print "Computing Mismatch String Kernel..."
    l = 16  # alphabet size
    k = 4  # trie depth
    m = 0   # maximum allowable mismatch for 'similar' k-mers
    mmsk = MismatchStringKernel(l, k, m, verbose=1).fit(
        X, Y=Y, kernel_update_callback=_update_kernel_plot)
    print "...done."
    print "\r\nKernel:\r\n%s\r\n" % mmsk.kernel_

    # prediction on train data
    print "Prediction on train data..."
    print "...done; fitting accuracy: %.2f%s.\r\n" % (
        (mmsk.predict(X) == Y).sum() * 100. / len(Y), "%")

    # save kernel plot
    plt.ioff()  # exit interactive mode
    _update_kernel_plot(mmsk.kernel_)
    plt.savefig("kernel.png", bbox_inches="tight", dpi=200, facecolor="k",
                edgecolor="k")

    # one last time
    plt.show()
