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
            `leafs`: a dictionary of leafs of the trie
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
            self.kernel_, _, _ = self.traverse(
                X, self.l, self.k, self.m, **kwargs)

            # normalize kernel
            self.kernel_ = normalize_kernel(self.kernel_)

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

        xstr = ''.join(['[%s]' % a for a in x])
        kv = np.zeros(len(self.svc_.support_))

        # scan through all k-grams/mers of test string
        seen_kmers = []  # k-mers we've already seen in this test string
        for i in xrange(len(x) - self.k + 1):
            kmer = ('[%s]' * self.k) % tuple(x[i:i + self.k])
            if kmer in seen_kmers:
                continue

            seen_kmers.append(kmer)
            if kmer in self.leaf_kgrams_:
                for j in xrange(len(self.svc_.support_)):
                    if self.svc_.support_[j] in self.leaf_kgrams_[kmer]:
                        kgrams = self.leaf_kgrams_[kmer][self.svc_.support_[j]]
                        count = len(kgrams) if isinstance(
                            kgrams, list) else kgrams
                        kv[j] += np.exp(-(count + xstr.count(kmer)))

        # scaling: correct for variance
        kv = kv / kv.std()

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

                # voting
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
        Predicts the label of the test vector x.

        """

        x = np.array(x)
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
                '/tmp/pkts.txt')), dtype=np.int)
    Y = np.loadtxt(os.path.abspath(os.path.join(
                os.path.dirname(os.path.dirname(sys.argv[0])),
                '/tmp/labels.txt')), dtype=np.int)

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
    import wrappers
    # kernel = wrappers.load_boost_array("/tmp/k.txt")
    k = 4  # trie depth
    d = 16  # alphabet size
    m = 0   # maximum allowable mismatch for 'similar' k-mers
    mmsk = MismatchStringKernel(d, k, m)
    kernel = mmsk.fit(X, kernel_update_callback=_update_kernel_plot).kernel_
    print "...done."
    print "\r\nKernel:\r\n%s\r\n" % kernel

    # construct SVM classifier with our mismatch string kernel
    from sklearn.svm import SVC

    print ("Constructing SVM classifier with our precomputed "
           "mismatch string kernel...")
    svc = SVC(kernel='precomputed')
    print '...done; %s\r\n' % svc

    # fit
    print "Fitting SVM against training data..."
    svc.fit(kernel, Y)
    print "...done; fitting accuracy: %.2f%s.\r\n" % (
        (svc.predict(kernel) == Y).sum() * 100. / len(Y), "%")

    # save kernel plot
    plt.ioff()  # exit interactive mode
    _update_kernel_plot(kernel)
    plt.savefig("kernel.png", bbox_inches="tight", dpi=200, facecolor="k",
                edgecolor="k")

    # one last time
    plt.show()
