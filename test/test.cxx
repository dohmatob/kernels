/*!
  \file test.cxx
  \author DOHMATOB Elvis Dopgima
  \brief Main test file.
*/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Trie test

#include <boost/assign/std/vector.hpp>
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctype.h>
#include <algorithm>

#include "Trie.h"

using namespace boost::unit_test_framework; 
using namespace boost::assign;
using namespace Combinatorics;
      
BOOST_AUTO_TEST_CASE(test_create_chunk)
{
  Chunk chunk = create_chunk(0, 3, 1);
  BOOST_CHECK_EQUAL(chunk.offset, 0);
  BOOST_CHECK_EQUAL(chunk.length, 3);
  BOOST_CHECK_EQUAL(chunk.mismatches, 1);
}

BOOST_AUTO_TEST_CASE(test_Chunks)
{
  Chunks chunks;
  chunks += create_chunk(2, 5, 0);  // add a chunk
  chunks += create_chunk(3, 20, 1);  // another chunk

  int count = 0;
  for(Chunks::const_iterator chunks_it = chunks.begin(); chunks_it != chunks.end(); 
      chunks_it++)
    {
      int offset = count == 0 ? 2 : 3;
      int length = count == 0 ? 5 : 20;
      int mismatches = count == 0 ? 0 : 1;
	  
      Chunk chunk = *chunks_it;
      BOOST_CHECK_EQUAL(chunk.offset, offset);
      BOOST_CHECK_EQUAL(chunk.length, length);
      BOOST_CHECK_EQUAL(chunk.mismatches, mismatches);

      count++;
    }

  BOOST_CHECK_EQUAL(chunks.size(), 2);
  Chunks::iterator chunks_it = chunks.begin();
  chunks_it = chunks.erase(chunks_it);
  BOOST_CHECK_EQUAL(chunks.size(), 1);
  chunks_it = chunks.erase(chunks_it);
  BOOST_CHECK(chunks.empty());
}

BOOST_AUTO_TEST_CASE(test_TrieMetadata)
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

  Combinatorics::TrainingDataset training_dataset;
  std::vector<int > seq;
  seq += 0,0,1,0;
  training_dataset += seq;
  seq.clear();
  seq += 1,0,1,0; 
  training_dataset += seq;
  seq.clear();
  seq += 1,1,1,0; 
  training_dataset += seq;
  seq.clear();

  compute_metadata(trie,
		   2, // branching degree (number of children per internal node
		   training_dataset
		   );

  for(TrieMetadata::const_iterator metadata_it = trie->metadata.begin(); metadata_it != trie->metadata.end(); 
      metadata_it++)
    {
      Chunks chunks = metadata_it->second;
      BOOST_CHECK_EQUAL(chunks.size(), 3);  
    }

  BOOST_CHECK_EQUAL(trie->metadata.size(), 3); 
}

BOOST_AUTO_TEST_CASE(test_digits_data)
{
  // instantiate trie object
  Trie trie = create_trienode();

  // load data
  TrainingDataset training_dataset;
  training_dataset = load_training_dataset("data/digits_data.dat");
  int n_samples = std::min((int)training_dataset.size(), 100);
  int k = 4;
  int d = 16;  // size of alphabet
  int m = 1;  // max number of allowable mismatches for 'similar' tokens
  training_dataset = TrainingDataset(training_dataset.begin(),
				     training_dataset.begin() + \
				     n_samples);

  // intialize kernel to zero
  ublas::matrix<double > kernel = ublas::zero_matrix<double >(training_dataset.size(), 
							      training_dataset.size());
				     
  // fit
  int nkmers = expand(trie, k, d, m, training_dataset, kernel);
  std::cout << nkmers << " " << k << "-mers out of " << std::pow(d, k) << " survived." 
	    << std::endl;
  
  // normalize kernel to remove the 'bias of length'
  Combinatorics::normalize_kernel(kernel);

  // dump kernel unto disk
  std::ofstream kernelfile;
  kernelfile.open("data/digits_kernel.dat");
  kernelfile << kernel;
  kernelfile.close();
}  
