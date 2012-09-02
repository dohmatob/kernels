#ifndef TRIE_H
#define TRIE_H

#include <vector>
#include <map>
#include <iostream>
#include <string>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/multi_array.hpp>
#include <boost/assert.hpp>

using namespace boost;
namespace ublas = boost::numeric::ublas;

/*
  \file Trie.h
  \author DOHMATOB Elvis Dopgima
*/

/*!
  \namespace Combinatorics
*/
namespace Combinatorics
{
  /*!
    \struct chunk_struc
  */
  struct chunk_struct
  {
    int starting;
    int length;
    int mismatches;
  };

  /*!
    \typedef Chunk
  */
  typedef struct chunk_struct Chunk;

  Chunk create_chunk(int starting, int ending, int mismatches);

  /*!
    \typedef chunks
  */
  typedef std::vector<Chunk > Chunks;

  /*
    \typedef _TrieMetadata
  */
  typedef std::map<int, Chunks > _TrieMetadata;

  /*!
    \struct _trie_struct
  */
  struct _trie_struct
  {
    int label;
    struct _trie_struct *parent;
    std::stringstream rootpath;
    std::vector<struct _trie_struct* > children;
    int nodecount;
    _TrieMetadata metadata;
  };

  typedef struct _trie_struct _TrieNode;

  /*!
    \typedef _Trie
  */
  typedef _TrieNode *_Trie;

  _Trie create_trie();

  _Trie create_trie(int label);

  _Trie create_trie(int label, _Trie& parent);

  void add_child(_Trie& parent, _Trie& child);

  unsigned short is_root(const _Trie& trie);

  void compute_metadata(_Trie& trie, int d, std::vector<std::vector<int > >& training_data);
  void trim_bad_chunks(_Trie& trie, int index, Chunks& chunks, int m, std::vector<std::vector<int > >& training_data);

  unsigned short inspect(_Trie& trie, int d, int m, std::vector<std::vector<int > >& training_data);

  void update_kernel(_Trie& trie, ublas::matrix<double >& kernel);

  void normalize_kernel(ublas::matrix<double >& kernel);

  void expand(_Trie& trie, int k, int d, int m, std::vector<std::vector<int > >& training_data, ublas::matrix<double >& kernel, std::string& padding);

  std::ostream& operator<<(std::ostream& cout, const Chunk& chunk);

  std::ostream& operator<<(std::ostream& cout, const Chunks& chunks);

  std::ostream& operator<<(std::ostream& cout, const _TrieMetadata& metadata);

  /*!
    \class Trie
  */
  class Trie
  {
  public:
    /*!
      Default constructor.
    */
    Trie();

    _Trie get_trie() const;

  private:
    _Trie _trie;
  };
}

#endif // TRIE_H

