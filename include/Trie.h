#ifndef TRIE_H
#define TRIE_H

#include <vector>
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace ublas = boost::numeric::ublas;

namespace Combinatorics
{
  typedef std::vector<unsigned int > Sequence;
  typedef struct trie_node_struct
  {
    int label;
    Sequence kmer;
    struct trie_node_struct *parent;
    std::vector<struct trie_node_struct *> children;
  } TrieNode;

  typedef TrieNode *Trie;

  unsigned short is_root(const Trie& trie);

  Trie create_trie();

  Trie create_trie(int label);

  Trie create_trie(int label, Trie& parent);

  std::ostream& operator<<(std::ostream& cout, const Sequence& seq);

  typedef ublas::matrix<double > Kernel;

  class MismatchTrie
  {
  public:
    MismatchTrie();

    Trie get_trie() const;

    void expand_trie(Trie& trie, unsigned int depth);
  
    void expand_trie(Trie& trie, unsigned int depth, unsigned nchildren);

    unsigned short check_node(const Trie& node) const;

    Kernel compute_kernel(unsigned int d, unsigned k, const std::vector<Sequence >& training_seqs);

  private:
    Trie _trie;
  };
};

#endif // TRIE_H   
