"""
:Module: Trie
:Synopsis: Python implementation of Trie (shor for "Retrieval Tree")
data structure
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com>

"""

import nose
import nose.tools
import numpy.testing
import numpy as np


def normalize_kernel(kernel):
    """
    Normalizes a kernel: kernel[x, y] by doing:

        kernel[x, y] / sqrt(kernel[x, x] * kernel[y, y])

    """

    if not isinstance(kernel, np.ndarray):
        kernel = np.ndarray(kernel)

    assert kernel.ndim == 2

    for i in xrange(kernel.shape[0]):
        for j in xrange(kernel.shape[1]):
            if i < j:
                q = np.sqrt(kernel[i, i] * kernel[j, j])
                if q > 0:
                    kernel[i, j] /= q
                    kernel[j, i] = kernel[i, j]

    # finally, set diagonal elements to 1
    np.fill_diagonal(kernel, 1.)

    return kernel


class MismatchTrie(object):
    """
    Trie (short for "Retrieval Tree") implementation, specific to
    'Mismatch String Kernels'.

    """

    def __init__(self, label=None, parent=None, verbose=1,
                 display_summerized_kgrams=False):
        """
        label: int, optional (default None)
            node label
        parent: `Trie` instance, optional (default None)
            node's parent
        verbose: int, optional (default 1)
            controls amount of verbosity (0 for no verbosity)
        display_summerize_kgrams: boolean, optional (default False)
            only display summerized version of kgrams
        """

        self.label = label  # label on edge connecting this node to its parent
        self.level = 0  # level of this node beyond the root node
        self.verbose = verbose
        self.display_summerized_kgrams = display_summerized_kgrams
        self.children = {}  # children of this node

        # concatenation of all labels of nodes from root node to this node
        self.full_label = ""
        # for each sample string, this dict holds pointers to it's k-mer/k-gram
        # substrings
        self.kgrams = {}

        self.parent = parent

        if not parent is None:
            parent.add_child(self)

    def is_root(self):
        """
        Checks whether this node is the root.

        """

        return self.parent is None

    def is_leaf(self):
        """
        Checks whether this node is a leaf.

        """

        return len(self.children) == 0

    def is_empty(self):
        """
        checks whether a node has 'died'.

        """

        return len(self.kgrams) == 0

    def copy_kgrams(self):
        """
        Copies the kgram data for this node (not the reference pointer,
        as this would have unpredictable consequences).

        """

        return {index: np.array(substring_pointers)
                for index, substring_pointers in self.kgrams.iteritems()}

    def add_child(self, child):
        """
        Adds a new child to this node.

        """

        assert not child.label in self.children

        # initialize ngram data to that of parent
        child.kgrams = self.copy_kgrams()

        # child is one level beyond parent
        child.level = self.level + 1

        # parent's full label (concatenation of labels on edges leading
        # from root node) is a prefix to child's the remainder is one
        # symbol, the child's label
        child.full_label = '%s[%s]' % (self.full_label, child.label)

        # let parent adopt child: commit child to parent's booklist
        self.children[child.label] = child

        # let child adopt parent
        child.parent = self

    def delete_child(self, child):
        """
        Deletes a child.

        """

        # get child label
        label = child.label if isinstance(child, MismatchTrie) else child

        # check that child really exists
        assert label in self.children, "No child with label %s exists." % label

        # delete the child
        del self.children[label]

    def __str__(self):
        if self.is_empty():
            kgrams_str = '{DEADEND}'
        elif self.display_summerized_kgrams:
            kgrams_str = '{%i}' % len(self.kgrams)
        else:
            kgrams_str = str(dict((k, v.tolist())
                                  for k, v in self.kgrams.iteritems()))

        return self.full_label + kgrams_str

    def log(self, msg):
        """
        Logs a msg (according to verbosity level).

        """

        if self.verbose:
            print msg

    def compute_kgrams(self, training_data, k):
        """
        Computes the meta-data for this node: i.e, for each input string
        training_data[index], computes the list of offsets of it's k-grams
        together with the mismatch counts (intialially zero) for this
        k-grams with the k-mer represented by this node `self`.

        Parameters
        ----------
        training_data: 2D array of shape (n_samples, n_features)
            training data for the kernel
        k: int:
            we will use k-mers for computing the kernel

        """

        # sanity checks
        if not isinstance(training_data, np.ndarray):
            training_data = np.array(training_data)
        if training_data.ndim == 1:
            training_data = np.array([training_data])

        assert training_data.ndim == 2

        # compute the len(training_data[index]) - k + 1 kgrams of each
        # input training string
        for index in xrange(len(training_data)):
            self.kgrams[index] = np.array([(offset,
                                            0  # no mismatch yet
                                            )
                                           for offset in xrange(
                        len(training_data[index]) - k + 1)])

    def process_node(self, training_data, k, m):
        """
        Processes this node, re-computing its supported k-grams. Finally,
        determines if node survives or not.

        Parameters
        ----------
        training_data: 2D array of shape (n_samples, n_features)
            training data for the kernel
        k: int:
            we will use k-mers for computing the kernel
        m: int
           maximum number of mismatches for 2 k-grams/-mers to be considered
           'similar'. Normally small values of m should work well, plus the
           complexity the algorithm is exponential in m.
           For example, if 'ELVIS' and '3LVIS' are dissimilar
           if m = 0, but similary if m = 1.

        Returns
        -------
        True if node survives, False else

        """

        # sanity checks
        if not isinstance(training_data, np.ndarray):
            training_data = np.array(training_data)
        if training_data.ndim == 1:
            training_data = np.array([training_data])

        assert training_data.ndim == 2

        if self.is_root():
            # compute meta-data
            self.compute_kgrams(training_data, k)
        else:
            # loop on all k-kgrams of input string training_data[index]
            for index, substring_pointers in self.kgrams.iteritems():
                # update mismatch counts
                substring_pointers[..., 1] += (training_data[index][
                        substring_pointers[..., 0] + self.level - 1
                        ] != self.label)

                # delete substring_pointers that present more than m mismatches
                self.kgrams[index] = np.delete(substring_pointers,
                                               np.nonzero(
                        substring_pointers[..., 1] > m),
                                               axis=0)

            # delete entries with empty substring_pointer list
            self.kgrams = {index: substring_pointers for (
                    index, substring_pointers) in self.kgrams.iteritems(
                    ) if len(substring_pointers)}

        return not self.is_empty()

    def update_kernel(self, kernel, m, weighting=True):
        """
        Updates the kernel in-memory.

        Parameters
        ----------
        kernel: 2D array of shape (n_samples, n_samples)
            kernel to be updated
        m: int
            the m in '(k, m)-mismatch kernel' terminology
        weighting: boolean, optional (default True)
            if set, the kernel will be weighted (exponential damping)

        """

        for i in self.kgrams:
            for j in self.kgrams:
                if weighting:
                    kernel[i, j] += np.exp(-(len(self.kgrams[i]
                                                 ) + len(self.kgrams[j])))
                else:
                    kernel[i, j] += len(self.kgrams[i]) * len(self.kgrams[j])

    def traverse(self, training_data, l, k, m, kernel=None, indentation=""):
        """
        Traverses a node, expanding it to plausible descendants.

        Parameters
        ----------
        training_data: 2D array of shape (n_samples, n_features)
            training data for the kernel
        l: int
            size of alphabet. Example of values with a natural interpretation:
            2: for binary data
            256 for data encoded as strings of bytes
            20: for protein data (bioinformatics)
        k: int:
            we will use k-mers for computing the kernel
        m: int
           maximum number of mismatches for 2 k-grams/-mers to be considered
           'similar'. Normally small values of m should work well, plus the
           complexity the algorithm is exponential in m.
           For example, if 'ELVIS' and '3LVIS' are dissimilar
           if m = 0, but similary if m = 1.
        kernel: 2D array of shape (n_samples, n_samples), optional (
        default None)
            kernel to be, or being, estimated
        indentation: string, optional (default "")
            controls indentation controlling pretty-printing

        Returns
        -------
        kernel: 2D array of shape (n_samples, n_samples)
            estimated kernel
        n_survived_kmers: int
            number of leaf nodes that survived the traversal
        go_ahead: boolean
            a flag indicating whether the node got abotted (False) or not

        """

        # initialize kernel if None
        if kernel is None:
            kernel = np.zeros((len(training_data), len(training_data)))

        # counts the number of leafs which are decendants of this node
        n_surviving_kmers = 0

        # process the node
        go_ahead = self.process_node(training_data, k, m)

        # display the node
        if self.is_root():
            self.log("//\r\n \\")
        else:
            self.log(indentation[:-1] + "+-" + str(self))

        # is node dead ?
        if go_ahead:
            # we've hit a leaf
            if k == 0:
                # yes, this is one more leaf/kmer
                n_surviving_kmers += 1

                # update the kernel
                self.update_kernel(kernel, m)
            else:
                # recursively bear and traverse child nodes
                indentation += " "
                for j in xrange(l):
                    # indentation for child display
                    self.log(indentation + "|")
                    child_indentation = indentation + (" " if (
                        j + 1) == l else "|")

                    # bear child
                    dskg = self.display_summerized_kgrams
                    child = MismatchTrie(label=j, parent=self,
                                         verbose=self.verbose,
                                         display_summerized_kgrams=dskg)

                    # traverse child
                    kernel, child_n_surviving_kmers, \
                        child_go_ahead = child.traverse(
                        training_data, l, k - 1, m, kernel=kernel,
                        indentation=child_indentation
                        )

                    # delete child if dead
                    if child.is_empty():
                        self.delete_child(child)

                    # update leaf counts
                    n_surviving_kmers += child_n_surviving_kmers if \
                        child_go_ahead else 0

        if self.is_root():
            self.log("%i out of %i k-mers survived.\r\n" % (
                    n_surviving_kmers, l ** k))

        return kernel, n_surviving_kmers, go_ahead


def test_trie_constructor():
    t = MismatchTrie()
    nose.tools.assert_true(t.is_root())
    nose.tools.assert_true(t.is_leaf())
    nose.tools.assert_equal(t.label, None)

    c = MismatchTrie(2, parent=t)
    nose.tools.assert_false(c.is_root())
    nose.tools.assert_true(c.is_leaf())
    nose.tools.assert_false(t.is_leaf())
    nose.tools.assert_equal(c.label, 2)


def test_compute_kgrams():
    t = MismatchTrie()
    x = [0, 1, 0, 0, 1]
    t.compute_kgrams(x, 1)
    numpy.testing.assert_array_equal(t.kgrams[0], [[0, 0],
                                                   [1, 0],
                                                   [2, 0],
                                                   [3, 0],
                                                   [4, 0]
                                                   ]
                                     )


def test_process_node():
    t = MismatchTrie()
    x = [0, 1, 0, 0, 1]

    # root node
    t.process_node(x, 1, 0)
    numpy.testing.assert_array_equal(t.kgrams[0], [[0, 0],
                                                   [1, 0],
                                                   [2, 0],
                                                   [3, 0],
                                                   [4, 0]
                                                   ]
                                     )

    # left child
    c = MismatchTrie(label=0, parent=t)
    c.process_node(x, 1, 0)
    numpy.testing.assert_array_equal(c.kgrams[0], [[0, 0],
                                                   [2, 0],
                                                   [3, 0],
                                                   ]
                                     )

    # right child
    c = MismatchTrie(label=1, parent=t)
    c.process_node(x, 1, 0)
    numpy.testing.assert_array_equal(c.kgrams[0], [[1, 0],
                                                   [4, 0],
                                                   ]
                                     )


def test_traverse():
    trie = MismatchTrie(verbose=False)

    X = np.zeros((90, 18))
    X[:30, :6] = 1
    X[30:60, 6:12] = 1
    X[60:, 12:] = 1

    _, n_surviving_kmers, _ = trie.traverse(X, 2, 4, 0)

    nose.tools.assert_equal(n_surviving_kmers, 8)

# demo
if __name__ == '__main__':
    trie = MismatchTrie(display_summerized_kgrams=True)

    data = np.zeros((90, 18))
    data[:30, :6] = 1
    data[30:60, 6:12] = 1
    data[60:, 12:] = 1

    kern = trie.traverse(data, 2, 4, 0)[0]
    normalize_kernel(kern)

    print "Kernel:\r\n%s" % kern
