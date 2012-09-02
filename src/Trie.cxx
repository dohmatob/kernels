/*!
  \file Trie.cxx
  \author DOHMATOB Elvis Dopgima
  \todo Use numpy arrays !!!
*/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Trie test

#include <boost/test/unit_test.hpp>
#include <sstream>
#include <ctype.h>
#include <boost/assign/std/vector.hpp>
#include "Trie.h"

using namespace boost::assign;

Combinatorics::Trie::Trie()
{
  _trie = Combinatorics::create_trie();
}
  
void Combinatorics::add_child(_Trie& parent, _Trie& child)
{
  parent->children.push_back(child);
  parent->nodecount++;
} 

Combinatorics::_Trie Combinatorics::Trie::get_trie() const
{
  return _trie;
}

Combinatorics::_Trie Combinatorics::create_trie(int label)
{
  Combinatorics::_Trie trie = new Combinatorics::_TrieNode;
  trie->label = label;
  trie->parent = 0;
  trie->nodecount = 0;
}

Combinatorics::_Trie Combinatorics::create_trie()
{
  return create_trie(-1);
}

Combinatorics::_Trie Combinatorics::create_trie(int label, Combinatorics::_Trie& parent)
{
  Combinatorics::_Trie trie = new Combinatorics::_TrieNode;
  trie->label = label;
  trie->parent = parent;
  trie->nodecount = 0;
  trie->metadata = parent->metadata;
  trie->rootpath << parent->rootpath.str();
  trie->rootpath << char(__toascii('a') + label);
  add_child(parent, trie);
} 

unsigned short Combinatorics::is_root(const Combinatorics::_Trie& trie)
{
  return trie->parent == 0;
}

void Combinatorics::compute_metadata(Combinatorics::_Trie& trie, int d, std::vector<std::vector<int > >& training_data)
{
  for(unsigned index = 0; index < training_data.size(); index++)
    {
      Chunks chunks;
      BOOST_ASSERT(training_data[index].size() - d + 1 > 0);
      for(int starting = 0; starting < training_data[index].size() - d + 1; starting++)
	{
	  chunks.push_back(create_chunk(starting, 0, 0));
	}
      trie->metadata[index] = chunks;
    }
}

std::ostream& Combinatorics::operator<<(std::ostream& cout, const Combinatorics::Chunk& chunk)
{
  cout << "(" << chunk.starting << "," << chunk.length << "," << chunk.mismatches << ")";

  return cout;
}

std::ostream& Combinatorics::operator<<(std::ostream& cout, const Combinatorics::Chunks& chunks)
{
  Combinatorics::Chunks::const_iterator chunks_it = chunks.begin();
  cout << "[" << chunks.size() << "](" << (chunks_it == chunks.end() ? ")" : "");
  while(chunks_it != chunks.end())
    {
      cout << *chunks_it;
      chunks_it++;
      cout << (chunks_it == chunks.end() ? ")" : ",");
    }
  
  return cout;
}

std::ostream& Combinatorics::operator<<(std::ostream& cout, const Combinatorics::_TrieMetadata& metadata)
{
  Combinatorics::_TrieMetadata::const_iterator metadata_it = metadata.begin();
  cout << "{" << (metadata_it == metadata.end() ? "}" : "");
  while(metadata_it != metadata.end())
    {
      cout << metadata_it->first << ":" << metadata_it->second;
      metadata_it++;
      cout << (metadata_it == metadata.end() ? "}" : ",");
    }

  return cout;
}
  
void Combinatorics::trim_bad_chunks(Combinatorics::_Trie& trie, int index, Combinatorics::Chunks& chunks, int m, std::vector<std::vector<int > >& training_data)
{
  Combinatorics::Chunks::iterator chunks_it = chunks.begin();
  while(chunks_it != chunks.end())
    {
      Combinatorics::Chunk chunk = *chunks_it;
      chunk.mismatches += (training_data[index][chunk.starting + chunk.length] != trie->label) ? 1 : 0;
      chunk.length++;
      
      // delete this chunk if we have hit more than m mismatches with it
      if(chunk.mismatches > m)
	{
	  chunks_it = chunks.erase(chunks_it);
	  continue;
	}
      
      *chunks_it = chunk;
      chunks_it++;
    }
}
  
unsigned short Combinatorics::inspect(Combinatorics::_Trie& trie, int d, int m, std::vector<std::vector<int > >& training_data)
{
  if(is_root(trie))
    {
      // create meta data for root node (this will be copied to children nodes as they're created
      Combinatorics::compute_metadata(trie, d, training_data);
    }	  
  else
    {
      // house_keeping:
      Combinatorics::_TrieMetadata::iterator metadata_it = trie->metadata.begin();
      while(metadata_it != trie->metadata.end())
	{
	  int index = metadata_it->first;
	  Chunks chunks = metadata_it->second;

	  // trim-off all chunks that have hit the mismatch threshold (m)
	  Combinatorics::trim_bad_chunks(trie, index, chunks, m, training_data);

	  if(chunks.empty())
	    {
	      trie->metadata.erase(index);
	    }
	  else
	    {	
	      trie->metadata[index] = chunks;
	    }
	  metadata_it++;
	}
    }
    
  return !trie->metadata.empty(); 
}

void Combinatorics::normalize_kernel(ublas::matrix<double >& kernel)
{
  for(int i = 0; i < kernel.size1(); i++)
    {
      for(int j = 0; j < kernel.size2(); j++)
	{
	  double quotient = std::sqrt(kernel(i,i)*kernel(j,j));
	  kernel(i,j) /= (quotient > 0 ? quotient : 1);
	}
    }
}

void Combinatorics::update_kernel(Combinatorics::_Trie& trie, ublas::matrix<double >& kernel)
{
  // compute source weights for surving k-mers
  ublas::vector<int > source_weights = ublas::scalar_vector<int >(kernel.size1(), 0);  
  for(Combinatorics::_TrieMetadata::iterator metadata_it = trie->metadata.begin(); metadata_it != trie->metadata.end(); metadata_it++)
    {
      source_weights[metadata_it->first] =  metadata_it->second.size();
    }
  
  // update all kernel entries corresponding to surviving k-kmers
  kernel += outer_prod(source_weights, source_weights);

  // normalize kernel to remove the 'bias of length'
  Combinatorics::normalize_kernel(kernel);
}

void Combinatorics::expand(Combinatorics::_Trie& trie, int k, int d, int m, std::vector<std::vector<int > >& training_data, ublas::matrix<double >& kernel, std::string& padding)
{
  unsigned short go_ahead = inspect(trie, d, m, training_data);

  if(is_root(trie))
    {
      std::cout << "//\r\n" << (d > 0 ? " \\" : "") << std::endl;
    }
  else
    {
      std::cout << padding.substr(0, padding.length() - 1) + "+-" + trie->rootpath.str() + "," << trie->metadata.size() << "/" << std::endl;
    }
  
  padding += " ";

  if(go_ahead)
    {
      if(k == 0)
	{
	  // updata kernel
	  Combinatorics::update_kernel(trie, kernel);
	}
      else
	{
	  for(int j = 0; j < d; j++)
	    {
	      std::string child_padding(padding);
	      child_padding += (j + 1 == d) ? " " : "|";
	      _Trie tmp = create_trie(j, trie);
	      expand(trie->children[j], k - 1, d, m, training_data, kernel, child_padding);
	    }
	}
    }
}
      
Combinatorics::Chunk Combinatorics::create_chunk(int starting, int length, int mismatches)
{
  Combinatorics::Chunk *chunk = new Chunk;
  chunk->starting = starting;
  chunk->length = length;
  chunk->mismatches = mismatches;

  return *chunk;
}

using namespace Combinatorics;
      
BOOST_AUTO_TEST_CASE(test_Chunkconstructors)
{
  Chunk chunk = create_chunk(0, 3, 1);
  BOOST_CHECK_EQUAL(chunk.starting, 0);
  BOOST_CHECK_EQUAL(chunk.length, 3);
  BOOST_CHECK_EQUAL(chunk.mismatches, 1);
}

BOOST_AUTO_TEST_CASE(test_Chunks_constructors)
{
  Chunks chunks;
  Chunk chunk = create_chunk(2, 5, 0);
  chunks += chunk;

  for(Chunks::const_iterator chunks_it = chunks.begin(); chunks_it != chunks.end(); chunks_it++)
    {
      chunk = *chunks_it;
      BOOST_CHECK_EQUAL(chunk.starting, 2);
      BOOST_CHECK_EQUAL(chunk.length, 5);
      BOOST_CHECK_EQUAL(chunk.mismatches, 0);
    }

  BOOST_CHECK_EQUAL(chunks.size(), 1);
  Chunks::iterator chunks_it = chunks.begin();
  chunks_it = chunks.erase(chunks_it);
  BOOST_CHECK(chunks.empty());
}

BOOST_AUTO_TEST_CASE(test__TrieMetadata_constructors)
{
  _TrieMetadata metadata;
  Chunks chunks;
  Chunk chunk = create_chunk(2, 5, 0);
  chunks += chunk;
  chunk = create_chunk(1, 2, 1);
  chunks += chunk;
  metadata[13] = chunks;
  BOOST_CHECK_EQUAL(metadata.size(), 1);
}

BOOST_AUTO_TEST_CASE(test_compute_metadata)
{  
  _Trie trie = create_trie();

  ublas::matrix<double > kernel = ublas::zero_matrix<double >(3,3);

  std::vector<std::vector<int > > training_data;
  std::vector<int > seq;
  seq += 0,0,1,0;
  training_data += seq;
  seq.clear();
  seq += 1,0,1,0; 
  training_data += seq;
  seq.clear();
  seq += 1,1,1,0; 
  training_data += seq;
  seq.clear();

  compute_metadata(trie,
		   2, // branching degree (number of children per internal node
		   training_data
		   );

  for(_TrieMetadata::const_iterator metadata_it = trie->metadata.begin(); metadata_it != trie->metadata.end(); metadata_it++)
    {
      Chunks chunks = metadata_it->second;
      BOOST_CHECK_EQUAL(chunks.size(), 3);  
    }

  BOOST_CHECK_EQUAL(trie->metadata.size(), 3); 
}

BOOST_AUTO_TEST_CASE(test_misc)
{
  _Trie trie = create_trie();

  std::vector<std::vector<int > > training_data;
  std::vector<int > seq;
  seq += 0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1;
  training_data += seq;
  seq.clear();  
  seq += 0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1;
  training_data += seq;
  seq.clear();
  seq += 1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0;
  training_data += seq;
  seq.clear();
  seq += 1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1;
  training_data += seq;
  seq.clear();
  seq += 1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1;
  training_data += seq;
  seq.clear();
  seq += 1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,1,1;
  training_data += seq;
  seq.clear();
  seq += 1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1;
  training_data += seq;
  seq.clear();
  seq += 0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0;
  training_data += seq;
  seq.clear();
  seq += 0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0;
  training_data += seq;
  seq.clear();
  seq += 0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0;
  training_data += seq;
  seq.clear();
  seq += 0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0;
  training_data += seq;
  seq.clear();
  seq += 0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0;
  training_data += seq;
  seq.clear();
  seq += 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
  training_data += seq;
  seq.clear();
  seq += 0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0;
  training_data += seq;
  seq.clear();
  seq += 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1;
  training_data += seq;
  seq.clear();
  seq += 0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1;
  training_data += seq;
  seq.clear();
  seq += 0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1;
  training_data += seq;
  seq.clear();
  seq += 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0;
  training_data += seq;
  seq.clear();
  seq += 1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1;
  training_data += seq;
  seq.clear();
  seq += 1,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1;
  training_data += seq;
  seq.clear();
  seq += 0,0,0,0,0,0,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0;
  training_data += seq;
  seq.clear();
  seq += 0,0,1,0,0,0,0,1,0,0,0,1,1,1,1,1,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,1;
  training_data += seq;
  seq.clear();
  seq += 0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0;
  training_data += seq;
  seq.clear();
  seq += 1,1,1,1,1,1,1,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1;
  training_data += seq;
  seq.clear();
  seq += 1,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,1,1;
  training_data += seq;
  seq.clear();
  seq += 0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1;
  training_data += seq;
  seq.clear();
  seq += 0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0;
  training_data += seq;
  seq.clear();
  seq += 0,0,1,0,0,0,0,1,0,0,0,1,1,1,0,1,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,1;
  training_data += seq;
  seq.clear();
  seq += 0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0;
  training_data += seq;
  seq.clear();
  seq += 0,1,1,1,1,1,1,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0;
  training_data += seq;
  seq.clear();
  seq += 0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1;
  training_data += seq;
  seq.clear();
  seq += 0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1;
  training_data += seq;
  seq.clear();

  ublas::matrix<double > kernel = ublas::zero_matrix<double >(training_data.size(), training_data.size());

  std::string padding(" ");
  expand(trie, 7, 2, 2, training_data, kernel, padding);
  std::cout << std::endl << kernel << std::endl;
}
  










