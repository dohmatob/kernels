"""
:Module: mismatch_string_kernel
:Author: dohmatob elvis dopgima <gmdopp@gmail.com>
:Synopsis: Implementation of "Mismatch String Kernels"
(Reference http://bioinformatics.oxfordjournals.org/content/20/4/467.short)

"""

from trie import MismatchTrie, normalize_kernel


class MismatchStringKernel(MismatchTrie):
    """
    Python implementation of Mismatch String Kernels. See reference above.

    """

    def __init__(self, l, k, m, **kwargs):
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
        **kwargs: dict, optional (default empy)
            optional parameters to pass to `tree.MismatchTrie` instantiation

        Attributes
        ----------
        `kernel_`: 2D array of shape (n_sampled, n_samples)
            estimated kernel
        `n_survived_kmers`:
            number of leafs/k-mers that survived trie traversal

        """

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
                ("You provided k = %i and m = %i. m is too big (must be at "
                 "must k / 2). This doesn't make sense.") % (k, m))

        self.l = l
        self.k = k
        self.m = m

    def fit(self, X, **kwargs):
        """
        Fit Mismatch String Kernel on data.

        Parameters
        ----------
        X: 2D array of shape (n_samples, n_features)
            training data for the kernel

        Returns
        -------
        self: `MismatchStringKernel` object
            fitted kernel instance

        """

        # compute kernel
        self.kernel_, self.n_survived_kmers_, _ = self.traverse(
            X, self.l, self.k, self.m, **kwargs)

        # normalize kernel
        self.kernel_ = normalize_kernel(self.kernel_)

        return self
