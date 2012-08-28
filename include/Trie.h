#ifndef TRIE_H
#define TRIE_H

/*!
  \file Trie.h
  \brief Header file for Trie-related structures.
  \author DOHMATOB Elvis Dopgima
*/

#include <vector>
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace ublas = boost::numeric::ublas;

/*!
  \namespace Combinatorics
  \brief Namespace for stuff related to combinarial structures (trees, etc.)
*/
namespace Combinatorics
{
  /*!
    \typedef Sequence
    \brief Internal representation of sequences.
  */						
  typedef std::vector<unsigned int > Sequence;

  /*!
    \struct trie_node_struct
    \brief Internal reprensentation nodes for a trie (prefix tree).
  */
  struct trie_node_struct
  {
    /*!
      Label of node.
    */
    int label;

    /*!
      k-mer of node (i.e, concatenation of all labels from root node down to this node).
    */
    Sequence kmer;

    /*!
      Parent of this node.
    */
    struct trie_node_struct *parent;

    /*!
      Children of this node.
    */
    std::vector<struct trie_node_struct *> children;
  };

  /*!
    \typedef TrieNode
    \brief A smarter name for "struct trie_node_struct"
  */
  typedef struct trie_node_struct TrieNode;

  /*!
    \typedef Trie
    \brief Internal reprensentation of trie object.
  */
  typedef TrieNode *Trie;

  /*!
    Function to check whether trie is root.

    \param trie a reference to the trie
  */
  unsigned short is_root(const Trie& trie);

  /*!
    Default function to construct Trie object.

    \return reference to constructed Trie object
  */
  Trie create_trie();

  /*!
    Function to create Trie object with given node label.

    \param label label of the node to be constructed

    \return reference to constructed Trie object
  */
  Trie create_trie(int label);

  /*!
    Function to create Trie object with given node label and parent.

    \param label label of the node to be constructed
    \param parent parent of the node to be constructed

    \return reference to constructed Trie object
  */
  Trie create_trie(int label, Trie& parent);

  /*!
    Overloading of operator<< for Sequence
    
    \param cout output stream to receive flux
    \param seq sequence to be displayed
  */
  std::ostream& operator<<(std::ostream& cout, const Sequence& seq);

  /*!
    \typedef Kernel
    \brief Internal representation of inner-product kernel.
  */
  typedef ublas::matrix<double > Kernel;

  /*!
    \class MismatchTrie
    \brief http://bioinformatics.oxfordjournals.org/content/20/4/467.full.pdf+html
  */
  class MismatchTrie
  {
  public:
    /*!
      Default constructor.
    */
    MismatchTrie();

    /*
      Method to return _trie member.
    */
    Trie get_trie() const;

    /*!
      Default method to grow trie from training set.

      \param trie reference to Trie object to expand
      \param depth depth limit for trie growth
    */
    void expand_trie(Trie& trie, unsigned int depth);

    /*!
      Overloading of get_trie() method.

      \param trie reference to Trie object to expand
      \param depth depth limit for tree growth
      \param nchildren number of children of per non-leaf node
    */
    void expand_trie(Trie& trie, unsigned int depth, unsigned nchildren);

    /*!
      Callback method invoked to process each created node, and decide whether to develope node's children.

      \param node reference to node under inspection
      
      \return boolean decision on whether to expand node's children or not.
    */
    unsigned short node_callback(const Trie& node) const;

    /*!
      Method to compute mismatch string kernel matrix, given training set.

      \param d depth limit for trie growth
      \param k alphabet size

      \return computed kernel matrix
    */
    Kernel compute_kernel(unsigned int d, unsigned k, const std::vector<Sequence >& training_seqs);

  private:
    /*!
      Trie object used in computing the kernel..
    */
    Trie _trie;
  };
};

#endif // TRIE_H   
