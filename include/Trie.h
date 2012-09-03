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
    \struct chunk_struct
    \brief Encapsulation of continguous subsequences.
  */
  struct chunk_struct
  {
    /*!
      offset where the chunk starts
    */
    int offset;

    /*!
      length of chunk
    */
    int length;
    
    /*!
      number of mismatches encountered yet by the chunk
    */
    int mismatches;
  };

  /*!
    \typedef Chunk
    \brief A shorter name for "struct chunk_struct"
  */
  typedef struct chunk_struct Chunk;

  /*!
    Handy function for creating a Chunk.

    \param offset offset where the chunk starts
    \param length length of chunk
    \param mismatches number of mismatches encountered yet by the chunk

    \return created chunk
  */
  Chunk create_chunk(int offset, int length, int mismatches);

  /*!
    \typedef Chunks
    \brief Encapsulation of the list of surviving chunks at node.
  */
  typedef std::vector<Chunk > Chunks;

  /*!
    \typedef TrieMetadata
    \brief Encapsulation of trie node's meta-data.
  */
  typedef std::map<int, Chunks > TrieMetadata;

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
    std::vector<struct trie_struct* > children;

    /*!
      number of nodes in trie rooted at this node
    */
    int nodecount;

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
    \typedef Trie
    \brief A trie is simply a pointer to its root node structure.
  */
  typedef TrieNode *Trie;

  /*!
    \typedef TrainingDataset
    \brief A container for holding sequences of training data sequences.
  */
  typedef std::vector<std::vector<int> > TrainingDataset;

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
    \param d branching degree of node
    \param training_dataset bail of training sequences
  */
  void compute_metadata(Trie& trie, int d, TrainingDataset& training_dataset);

  /*!
    Function to trim-off all chunks of a training sequence that have have hit the mismatch tolerance.

    \param trie pointer to node under inspection
    \param chunks reference to Chunks object under inspection
    \param index of the training sequence as a member in the training pool
    \param m mismatch tolerance (i.e, maximum number number of differences between two j-mers for which the j-mers are still considered 'similar')
    \param training_dataset bail of training sequences
  */
  void trim_bad_chunks(Trie& trie, int index, Chunks& chunks, int m, TrainingDataset& training_dataset);

  /*!
    Function to recompute the meta-data of a node, and determine whether it's worth exploring further down.

    \param trie pointer to node under inspection
    \param d branching degree of node
    \param m mismatch tolerance (i.e, maximum number number of differences between two j-mers for which the j-mers are still considered 'similar')
    \param training_dataset bail of training sequences
  */    
  bool inspect(Trie& trie, int d, int m, TrainingDataset& training_dataset);

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
    Function to expand a node a given number of levels down and with a given braching factor, constrained by a training dataset and a mismatch tolerance.

    \param trie pointer to node to be expanded
    \param k depth of expansion
    \param d branching factor
    \param m mismatch tolerance (i.e, maximum number number of differences between two j-mers for which the j-mers are still considered 'similar')
    \param training_dataset bail of training sequences
    \param kernel a reference to the mismatch kernel
    \param padding a control string used in displaying the node
  */
  void expand(Trie& trie, int k, int d, int m, TrainingDataset& training_dataset, ublas::matrix<double >& kernel, std::string& padding);

  /*!
    An overloading of expand(..).

    \param trie pointer to node to be expanded
    \param k depth of expansion
    \param d branching factor
    \param m mismatch tolerance (i.e, maximum number number of differences between two j-mers for which the j-mers are still considered 'similar')
    \param training_dataset bail of training sequences
    \param kernel a reference to the mismatch kernel
  */
  void expand(Trie& trie, int k, int d, int m, TrainingDataset& training_dataset, ublas::matrix<double >& kernel);
  
  /*!
    Overloading of operator<< for Chunk.

    \param cout output stream receiving results
    \param chunk reference to Chunk structure to be printed

    \return resulting ostream
  */
  std::ostream& operator<<(std::ostream& cout, const Chunk& chunk);

  /*!
    Overloading of operator<< for Chunks.

    \param cout output stream receiving results
    \param chunks reference to Chunks object to be printed

    \return resulting ostream
  */  
  std::ostream& operator<<(std::ostream& cout, const Chunks& chunks);
  
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
    Function to load training dataset from disk.

    \param filename filename containing dataset

    \return loaded dataset
  */
  TrainingDataset load_training_dataset(const std::string& filename);

  /*!
    Function for fancy-displaying trie node.

    \param trie pointer to node to be displayed
    \param d branching degree of trie node
    \param padding a control string used in displaying the node
  */
  void display_trienode(const Trie& trie, int d, const std::string& padding);
};

#endif // TRIE_H


