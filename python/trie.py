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

        np.fill_diagonal(kernel, 1.)

    return kernel


class Trie(object):
    """
    Trie (short for "Retrieval Tree") implementation.

    """

    def __init__(self, label=None, level=0, parent=None):
        self.label = label
        self.level = level
        self.full_label = ""
        self.children = {}
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

    def copy_kgrams(self):
        """
        Copies the kgram data for this node (not the reference pointer,
        as this would have unpredictable consequences).

        """

        return {index: np.array(chunks)
                for index, chunks in self.kgrams.iteritems()}

    def add_child(self, child):
        """
        Adds a new child to this node.

        """

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

    def compute_kgrams(self, training_data, k):
        """
        Computes the meta-data for this node: i.e, for each input string
        training_data[index], computes the list of offsets of it's k-grams
        together with the mismatch counts (intialially zero) for this
        k-grams with the k-mer represented by this node `self`.

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
        Processes this node.

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
            for index, chunks in self.kgrams.iteritems():
                # update mismatch counts
                chunks[..., 1] += (training_data[index][
                        chunks[..., 0] + self.level - 1] != self.label)

                # delete chunks that present more than m mismatches
                self.kgrams[index] = np.delete(chunks,
                                                np.nonzero(chunks[..., 1] > m),
                                                axis=0)

            # delete entries with empty chunk list
            self.kgrams = {index: chunks for (
                    index, chunks) in self.kgrams.iteritems() if len(chunks)}

        return len(self.kgrams)

    def update_kernel(self, kernel, m, weighting=True):
        for i in self.kgrams:
            for j in self.kgrams:
                if weighting:
                    kernel[i, j] += np.exp(-(len(self.kgrams[i]
                                                 ) + len(self.kgrams[j])))
                else:
                    kernel[i, j] += len(self.kgrams[i]) * len(self.kgrams[j])

    def __str__(self):
        return self.full_label + str(dict(
                (k, v.tolist()) for k, v in self.kgrams.iteritems()))

    def traverse(self, training_data, l, k, m, kernel=None, indentation=""):
        # initialize kernel if None
        if kernel is None:
            kernel = np.zeros((len(training_data), len(training_data)))

        # counts the number of leafs which are decendants of this node
        nkmers = 0

        # process the node
        go_ahead = self.process_node(training_data, k, m)

        # display the node
        if self.is_root():
            print "//\r\n \\"
        else:
            print indentation[:-1] + "+-" + str(self)

        indentation += " "

        # is node dead ?
        if go_ahead:
            # we've hit a leaf
            if k == 0:
                # yes, this is one more leaf/kmer
                nkmers += 1

                # update the kernel
                self.update_kernel(kernel, m)
            else:
                # recursively bear and traverse child nodes
                for j in xrange(l):
                    # indentation for child display
                    print indentation + "|"
                    child_indentation = indentation + (" " if (
                        j + 1) == l else "|")

                    # bear child
                    Trie(label=j, parent=self)

                    # traverse child
                    kernel, child_nkmers, child_go_ahead = self.children[
                        j].traverse(training_data, l, k - 1, m, kernel=kernel,
                        indentation=child_indentation
                        )

                    nkmers += child_nkmers if child_go_ahead else 0

        return kernel, nkmers, go_ahead


def test_trie_constructor():
    t = Trie()
    nose.tools.assert_true(t.is_root())
    nose.tools.assert_true(t.is_leaf())
    nose.tools.assert_equal(t.label, None)

    c = Trie(2, parent=t)
    nose.tools.assert_false(c.is_root())
    nose.tools.assert_true(c.is_leaf())
    nose.tools.assert_false(t.is_leaf())
    nose.tools.assert_equal(c.label, 2)


def test_compute_kgrams():
    t = Trie()
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
    t = Trie()
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
    c = Trie(label=0, parent=t)
    c.process_node(x, 1, 0)
    numpy.testing.assert_array_equal(c.kgrams[0], [[0, 0],
                                                   [2, 0],
                                                   [3, 0],
                                                   ]
                                     )

    # right child
    c = Trie(label=1, parent=t)
    c.process_node(x, 1, 0)
    numpy.testing.assert_array_equal(c.kgrams[0], [[1, 0],
                                                   [4, 0],
                                                   ]
                                     )


if __name__ == '__main__':
    trie = Trie()

    X = np.zeros((90, 18))
    X[:30, :6] = 1
    X[30:60, 6:12] = 1
    X[60:, 12:] = 1

    kernel, _, _ = trie.traverse(X, 2, 4, 0)
    print normalize_kernel(kernel)
