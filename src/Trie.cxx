#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Trie test
#include <boost/test/unit_test.hpp>
#include "Trie.h"

unsigned short Combinatorics::is_root(const Combinatorics::Trie& trie)
{
  return trie ? !trie->parent : 0;
}

Combinatorics::Trie Combinatorics::create_trie()
{
  return create_trie(-1);
}

Combinatorics::Trie Combinatorics::create_trie(int label)
{
  Combinatorics::Trie trie = new Combinatorics::TrieNode;
  trie->label = label;
  trie->parent = 0;

  if(!is_root(trie))
    {
      trie->kmer.push_back(label);
    }
  
  return trie;
}

Combinatorics::Trie Combinatorics::create_trie(int label, Combinatorics::Trie& parent)
{
  Combinatorics::Trie trie = new TrieNode;
  trie->label = -1;
  trie->parent = parent;

  trie->kmer = parent->kmer;
  trie->kmer.push_back(label);

  parent->children.push_back(trie);

  return trie;
}

std::ostream& Combinatorics::operator<<(std::ostream& cout, const Combinatorics::Sequence& seq)
{
  for(Sequence::const_iterator it = seq.begin(); it < seq.end(); it++)
    {
      cout << *it;
    }

  return cout;
}

Combinatorics::MismatchTrie::MismatchTrie()
{
  _trie = create_trie();
}

Combinatorics::Trie Combinatorics::MismatchTrie::get_trie() const
{
  return _trie;
}

void Combinatorics::MismatchTrie::expand_trie(Combinatorics::Trie& trie, unsigned int depth)
{
  expand_trie(trie, depth, 2);
}

void Combinatorics::MismatchTrie::expand_trie(Combinatorics::Trie& trie, unsigned int depth, unsigned int nchildren)
{
  if(!check_node(trie))
    {
      goto end;
    }
  
  if(depth == 0)
    {
      goto end;
    }

  for(unsigned int j = 0; j < nchildren; j++)
    {
      create_trie(j, trie);
      expand_trie(trie->children[j], depth - 1, nchildren);
    }

 end:
  return;
}

Combinatorics::Kernel Combinatorics::MismatchTrie::compute_kernel(unsigned int d, unsigned int k, const std::vector<Combinatorics::Sequence >& training_seqs)
{
  Kernel kernel = ublas::zero_matrix<double>(training_seqs.size(), training_seqs.size());

  expand_trie(_trie, d, k);

  return kernel;
}


unsigned short  Combinatorics::MismatchTrie::check_node(const Combinatorics::Trie& node) const
{
  if(!is_root(node))
    {
      std::cout << node->kmer << std::endl;
    }
  
  return 1; // XXX fictitious
}

using namespace Combinatorics;

BOOST_AUTO_TEST_CASE(test_constructors)
{
  Trie trie = create_trie();
  BOOST_CHECK_EQUAL(trie->label, -1);
  BOOST_CHECK(!trie->parent);
}

BOOST_AUTO_TEST_CASE(test_expand_trie)
{
  MismatchTrie mmtrie;
  std::vector<Sequence > training_seqs;
  Kernel kernel = mmtrie.compute_kernel(100, 10, training_seqs);
  // BOOST_CHECK_EQUAL(trie->children.size(), 2);
}

