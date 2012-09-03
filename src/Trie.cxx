/*!
  \file Trie.cxx
  \author DOHMATOB Elvis Dopgima
  \brief Implementation of Trie.h header file.
  \todo Use numpy arrays !!!
*/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Trie test

#include <boost/test/unit_test.hpp>
#include <boost/assert.hpp>
#include <boost/assign/std/vector.hpp>
#include <sstream>
#include <ctype.h>
#include <fstream>
#include "Trie.h"

using namespace boost::assign;

Combinatorics::Chunk Combinatorics::create_chunk(int offset, int length, int mismatches)
{
  Combinatorics::Chunk *chunk = new Chunk;
  chunk->offset = offset;
  chunk->length = length;
  chunk->mismatches = mismatches;

  return *chunk;
}
  
void Combinatorics::add_child(Combinatorics::Trie& parent, Trie& child)
{
  parent->children.push_back(child);
  parent->nodecount++;
} 

Combinatorics::Trie Combinatorics::create_trienode(int label)
{
  Combinatorics::Trie trie = new Combinatorics::TrieNode;
  trie->label = label;
  trie->parent = 0;
  trie->nodecount = 0;
}

Combinatorics::Trie Combinatorics::create_trienode()
{
  return create_trienode(-1);
}

Combinatorics::Trie Combinatorics::create_trienode(int label, Combinatorics::Trie& parent)
{
  Combinatorics::Trie trie = new Combinatorics::TrieNode;
  trie->label = label;
  trie->parent = parent;
  trie->nodecount = 0;
  trie->metadata = parent->metadata;
  trie->rootpath << parent->rootpath.str();
  trie->rootpath << char(__toascii('a') + label);
  add_child(parent, trie);
} 

bool Combinatorics::is_root(const Combinatorics::Trie& trie)
{
  return trie->parent == 0;
}

void Combinatorics::compute_metadata(Combinatorics::Trie& trie, int d, std::vector<std::vector<int > >& training_data)
{
  for(unsigned index = 0; index < training_data.size(); index++)
    {
      Chunks chunks;
      BOOST_ASSERT(training_data[index].size() - d + 1 > 0);
      for(int offset = 0; offset < training_data[index].size() - d + 1; offset++)
	{
	  chunks.push_back(create_chunk(offset, 0, 0));
	}
      trie->metadata[index] = chunks;
    }
}

std::ostream& Combinatorics::operator<<(std::ostream& cout, const Combinatorics::Chunk& chunk)
{
  cout << "(" << chunk.offset << "," << chunk.length << "," << chunk.mismatches << ")";

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

std::ostream& Combinatorics::operator<<(std::ostream& cout, const Combinatorics::TrieMetadata& metadata)
{
  Combinatorics::TrieMetadata::const_iterator metadata_it = metadata.begin();
  cout << "{" << (metadata_it == metadata.end() ? "}" : "");
  while(metadata_it != metadata.end())
    {
      cout << metadata_it->first << ":" << metadata_it->second;
      metadata_it++;
      cout << (metadata_it == metadata.end() ? "}" : ",");
    }

  return cout;
}
  
void Combinatorics::trim_bad_chunks(Combinatorics::Trie& trie, int index, Combinatorics::Chunks& chunks,
				    int m, std::vector<std::vector<int > >& training_data)
{
  Combinatorics::Chunks::iterator chunks_it = chunks.begin();
  while(chunks_it != chunks.end())
    {
      Combinatorics::Chunk chunk = *chunks_it;
      chunk.mismatches += (training_data[index][chunk.offset + chunk.length] != trie->label) ? 1 : 0;
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
  
bool Combinatorics::inspect(Combinatorics::Trie& trie, int d, int m, std::vector<std::vector<int > >& training_data)
{
  if(is_root(trie))
    {
      // create meta data for root node (this will be copied to children nodes as they're created
      Combinatorics::compute_metadata(trie, d, training_data);
    }	  
  else
    {
      // update metadata
      Combinatorics::TrieMetadata::iterator metadata_it = trie->metadata.begin();
      while(metadata_it != trie->metadata.end())
	{
	  int index = metadata_it->first;
	  Chunks chunks = metadata_it->second;

	  // trim-off all chunks that have hit the mismatch threshold (m)
	  Combinatorics::trim_bad_chunks(trie, index, chunks, m, training_data);

	  if(chunks.empty())
	    {
	      // no need keeping empty chunks
	      trie->metadata.erase(index);
	    }
	  else
	    {	
	      // update metadata entry
	      trie->metadata[index] = chunks;
	    }
	  
	  // proceed 
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

void Combinatorics::update_kernel(Combinatorics::Trie& trie, ublas::matrix<double >& kernel)
{
  // compute source weights for surving k-mers
  ublas::vector<int > source_weights = ublas::scalar_vector<int >(kernel.size1(), 0);  
  for(Combinatorics::TrieMetadata::iterator metadata_it = trie->metadata.begin(); metadata_it != trie->metadata.end(); metadata_it++)
    {
      source_weights[metadata_it->first] =  metadata_it->second.size();
    }
  
  // update all kernel entries corresponding to surviving k-kmers
  kernel += outer_prod(source_weights, source_weights);
}

std::ostream& Combinatorics::operator<<(std::ostream& cout, const Combinatorics::Trie& trie)
{
  if(!is_root(trie))
    {
      cout << trie->rootpath.str() + "," << trie->metadata.size() << "/";
    }
  
  return cout;
}

void Combinatorics::display_trienode(const Combinatorics::Trie& trie, int d, const std::string& padding)
{
  if(is_root(trie))
    {
      std::cout << "//\r\n" << (d > 0 ? " \\" : "") << std::endl;
    }
  else
    {
      std::cout << padding.substr(0, padding.length() - 1) + "+-" << trie << std::endl;
    }
}
  
void Combinatorics::expand(Combinatorics::Trie& trie, int k, int d, int m, std::vector<std::vector<int > >& training_data,
			   ublas::matrix<double >& kernel, std::string& padding)
{
  // recompute metadata of node
  bool go_ahead = inspect(trie, d, m, training_data);

  // display node info
  display_trienode(trie, d, padding);

  // update padding
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
	      Trie tmp = create_trienode(j, trie);
	      expand(trie->children[j], k - 1, d, m, training_data, kernel, child_padding);
	    }
	}
    }
}

void Combinatorics::expand(Combinatorics::Trie& trie, int k, int d, int m, std::vector<std::vector<int > >& training_data,
			   ublas::matrix<double >& kernel)
{
  // intantiate padding
  std::string padding(" ");

  // delegate to other version
  expand(trie, k, d, m, training_data, kernel, padding);
}
      
using namespace Combinatorics;
      
BOOST_AUTO_TEST_CASE(test_Chunkconstructors)
{
  Chunk chunk = create_chunk(0, 3, 1);
  BOOST_CHECK_EQUAL(chunk.offset, 0);
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
      BOOST_CHECK_EQUAL(chunk.offset, 2);
      BOOST_CHECK_EQUAL(chunk.length, 5);
      BOOST_CHECK_EQUAL(chunk.mismatches, 0);
    }

  BOOST_CHECK_EQUAL(chunks.size(), 1);
  Chunks::iterator chunks_it = chunks.begin();
  chunks_it = chunks.erase(chunks_it);
  BOOST_CHECK(chunks.empty());
}

BOOST_AUTO_TEST_CASE(test_TrieMetadata_constructors)
{
  TrieMetadata metadata;
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
  Trie trie = create_trienode();

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

  for(TrieMetadata::const_iterator metadata_it = trie->metadata.begin(); metadata_it != trie->metadata.end(); metadata_it++)
    {
      Chunks chunks = metadata_it->second;
      BOOST_CHECK_EQUAL(chunks.size(), 3);  
    }

  BOOST_CHECK_EQUAL(trie->metadata.size(), 3); 
}

BOOST_AUTO_TEST_CASE(test_misc)
{
  Trie trie = create_trienode();

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

  // expand
  expand(trie, 40, 2, 2, training_data, kernel);
  
  // normalize kernel to remove the 'bias of length'
  Combinatorics::normalize_kernel(kernel);
    
  // display kernel
  std::cout << std::endl << kernel << std::endl;

  // dump kernel disk
  std::ofstream kernelfile;
  kernelfile.open ("data/kernel.dat");
  kernelfile << kernel;
  kernelfile.close();
}
  










