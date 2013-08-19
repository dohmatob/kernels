"""
:Module: mismatch_string_kernel
:Author: dohmatob elvis dopgima <gmdopp@gmail.com>
:Synopsis: Implementation of "Mismatch String Kernels"
(Reference http://bioinformatics.oxfordjournals.org/content/20/4/467.short)

"""

import trie


class MismatchStringKernel(object):
    """
    Python implementation of Mismatch String Kernels. See r

    """

    def __init__(self, l, k, m, verbose=1):
        """
        Parameters
        ----------
        l: int
            size of alphabet. Example of values with a natural interpretation:
            2: for binary data
            256 for data encoded as strings of bytes
            20: for protein data (bioinformatics)
        k: int
           the k in 'k-mer' and 'k-gram'.
        m: int
           maximum number of mismatches for 2 k-grams/-mers to be considered
           'similar'. Normally small values of m should work well, plus the
           complexity the algorithm is exponential in m.
           For example, if 'ELVIS' and '3LVIS' are dissimilar
           if m = 0, but similary if m = 1.
        verbose: int, optional (default 1)
            controls amount of verbosity (0 for no verbosity)

        """

        # sanitize alphabet size
        if l < 2:
            raise ValueError(
                "Alphabet too small. l must be at least 2; got %i" % l)

        # sanitize kernel parameters (k, m)
        if 2 * m > k:
            raise ValueError(
                ("You provided k = %i and m = %i. m is too big (must be at "
                 "must k / 2). This doesn't make sense.") % (k, m))

        self.l = l
        self.k = k
        self.m = m
        self.verbose = verbose

    def fit(self, data):
        """
        Fit Mismatch String Kernel on data.

        Parameters
        ----------
        data: 2D array of shape (n_samples, n_features)
            training data for the kernel

        Returns
        -------
        self: `MismatchStringKernel` object
            fitted kernel instance

        """

        # intantiate Trie object
        t = trie.Trie(verbose=self.verbose,
                      display_summerized_kgrams=True)

        # copute kernel
        self.kernel_  = t.traverse(data, self.l, self.k, self.m)[0]

        # normalize kernel
        trie.normalize_kernel(self.kernel_)

        return self
