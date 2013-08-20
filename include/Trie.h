#ifndef TRIE_H
#define TRIE_H

/*!
  \file Trie.h
  \author DOHMATOB Elvis Dopgima
  \brief Specification of trie-related structures.
*/

#include <vector>
#include <map>
#include <iostream>
#include <string>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/multi_array.hpp>

using namespace boost;
namespace ublas = boost::numeric::ublas;

/*!
  \namespace Combinatorics
  \brief Namespace for combinatorial stuff like Tries (Suffix Trees), etc.
*/
namespace Combinatorics
{
  /*!
    \struct kgram_struct
    \brief Encapsulation of continguous subsequences.
  */
  struct kgram_struct
  {
    /*!
      offset where the kgram starts
    */
    int offset;

    /*!
      number of mismatches encountered yet by the kgram
    */
    int mismatches;
  };

  /*!
    \typedef Kgram
    \brief A shorter name for "struct kgram_struct"
  */
  typedef struct kgram_struct Kgram;

  /*!
    Handy function for creating a Kgram.

    \param offset offset where the kgram starts
    \param mismatches number of mismatches encountered yet by the kgram

    \return created kgram
  */
  Kgram create_kgram(int offset, int mismatches);

  /*!
    \typedef Kgrams
    \brief Encapsulation of the list of surviving kgrams at node.
  */
  typedef std::vector<Kgram > Kgrams;

  /*!
    \typedef TrieMetadata
    \brief Encapsulation of trie node's meta-data.
  */
  typedef std::map<int, Kgrams > TrieMetadata;

  /*!
    \struct trie_struct
    \brief Encapsulation of trie node.
  */
  struct trie_struct
  {
    /*!
      label of the edge connecting this node to its parent
    */
    int label;

    /*!
      the level of the node beyond the root node
    */
    int level;

    /*!
      parent of node
    */
    struct trie_struct *parent;

    /*!
      concatenation of all edge labels from root to this node
    */
    std::stringstream rootpath;

    /*!
      children of this node
    */
    std::map<int, struct trie_struct* > children;

    /*!
      meta-data of this node
    */
    TrieMetadata metadata;
  };

  /*!
    \typedef TrieNode
    \brief A shorter name for "struc trie_struct".
  */
  typedef struct trie_struct TrieNode;

  /*!
    \typedef TrieNodeChildren
    \brief A shorter name for "std::map<int, struct trie_struct* >"
  */
  typedef std::map<int, struct trie_struct* > TrieNodeChildren;

  /*!
    \typedef Trie
    \brief A trie is simply a pointer to its root node structure.
  */
  typedef TrieNode *Trie;

  /*!
    \typedef TrainingDataset
    \brief A container for holding sequences of training data sequences.
  */
  typedef ublas::matrix<int > TrainingDataset;

  /*!
    Default function to create a trie  node (root).

    \return pointer to root node of created node
  */
  Trie create_trienode();

  /*!
    Function to create a trie node with a given node label.
    
    \param label label of that will connect node to its parent

    \return pointer to root node of created node
  */
  Trie create_trienode(int label);

  /*!
    Function to create a trie node with a given node label and parent.

    \param label label of that will connect node to its parent
    \param parent pointer to parent node to to-be-created node

    \return pointer to root node of created node    
  */
  Trie create_trienode(int label, Trie& parent);

  /*!
    Function to free a trie node and all its descendants.

    \param trie pointer to the node.
  */
  void destroy_trie(Trie& trie);

  /*!
    Function to add a child node to a parent node.

  */
  void add_child(Trie& parent, Trie& child);

  /*!
    Function to determine whether a given node is root.

    \param trie pointer to node
    \return true if node is root, false otherwise
  */
  bool is_root(const Trie& trie);

  /*!
    Function to compute meta-data of trie node.

    \param trie pointer to node
    \param k depth of the target trie
    \param training_dataset bail of training sequences
  */
  void compute_metadata(Trie& trie, int k, TrainingDataset& training_dataset);

  /*!
    Function to trim-off all kgrams of a training sequence that have have hit the mismatch tolerance.

    \param trie pointer to node under inspection
    \param kgrams reference to Kgrams object under inspection
    \param index of the training sequence as a member in the training pool
    \param m mismatch tolerance (i.e, maximum number number of differences between two j-mers for which the j-mers are still considered 'similar')
    \param training_dataset bail of training sequences
  */
  void trim_bad_kgrams(Trie& trie, int index, int m, TrainingDataset& training_dataset);

  /*!
    Function to recompute the meta-data of a node, and determine whether it's worth exploring further down.

    \param trie pointer to node under inspection
    \param k depth of the target trie
    \param m mismatch tolerance (i.e, maximum number number of differences between two j-mers for which the j-mers are still considered 'similar')
    \param training_dataset bail of training sequences
  */    
  bool process_node(Trie& trie, int k, int m, TrainingDataset& training_dataset);

  /*!
    Function for updating the mismatch kernel, once a k-mer is reached.

    \param trie pointer to k-mer node under inspection
    \param m mismatch tolerance (i.e, maximum number number of differences between two j-mers for which the j-mers are still considered 'similar')
    \param kernel a reference to the mismatch kernel
  */
  void update_kernel(Trie& trie, int m, ublas::matrix<double >& kernel);

  /*!
    Function to normalize mismatch kernel so as to remove

    \param kernel a reference to the mismatch kernel to be normalized    
  */
  void normalize_kernel(ublas::matrix<double >& kernel);

  /*!
    Function to expand a node a given number of levels down and with a given braching factor,
    constrained by a training dataset and a mismatch tolerance.

    \param trie pointer to node to be expanded
    \param l alphabet size
    \param k depth of expansion
    \param m mismatch tolerance (i.e, maximum number number of differences between two j-mers
    for which the j-mers are still considered 'similar')
    \param training_dataset bail of training sequences
    \param kernel a reference to the mismatch kernel
    \param indentation a control string used in displaying the node

    \return number of surviving k-mers
  */
  int traverse(Trie& trie, int l, int k, int m, TrainingDataset& training_dataset, ublas::matrix<double >& kernel, std::string& indentation);

  /*!
    An overloading of traverse(..).

    \param trie pointer to node to be expanded
    \param k depth of expansion
    \param k depth of expansion
    \param m mismatch tolerance (i.e, maximum number number of differences between two j-mers for
    which the j-mers are still considered 'similar')
    \param training_dataset bail of training sequences
    \param kernel a reference to the mismatch kernel

    \return number of surviving k-mers
  */
  int traverse(Trie& trie, int l, int k, int m, TrainingDataset& training_dataset,
	       ublas::matrix<double >& kernel);
  
  /*!
    Overloading of operator<< for Kgram.

    \param cout output stream receiving results
    \param kgram reference to Kgram structure to be printed

    \return resulting ostream
  */
  std::ostream& operator<<(std::ostream& cout, const Kgram& kgram);

  /*!
    Overloading of operator<< for Kgrams.

    \param cout output stream receiving results
    \param kgrams reference to Kgrams object to be printed

    \return resulting ostream
  */  
  std::ostream& operator<<(std::ostream& cout, const Kgrams& kgrams);
  
  /*!
    Overloading of operator<< for TrieMetadata.
    
    \param cout output stream receiving results
    \param metadata reference to TrieMedata object to be printed

    \return resulting ostream
  */
  std::ostream& operator<<(std::ostream& cout, const TrieMetadata& metadata);

  /*!
    Overloading of operator<< for Trie.
    
    \param cout output stream receiving results
    \param trie pointer to the trie node to be printed

    \return resulting ostream
  */
  std::ostream& operator<<(std::ostream& cout, const Trie& trie);

  /*!
    Function to load a number of lines of (sample points) of training dataset from disk.
    Each line of the file must contain the same number of integers.

    \param filename filename containing dataset
    \parem nrows number of rows (samples) to read from file
    \return loaded dataset
  */
  TrainingDataset load_training_dataset(const std::string& filename, unsigned int nrows);

  /*!
    Function to load training dataset from disk. Each line of the file must contain
    the same number of integers.


    \param filename filename containing dataset

    \return loaded dataset
  */
  TrainingDataset load_training_dataset(const std::string& filename);
};

#endif // TRIE_H


