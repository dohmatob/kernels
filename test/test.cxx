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
      
BOOST_AUTO_TEST_CASE(test_create_kgram)
{
  Kgram kgram = create_kgram(0, 1);
  BOOST_CHECK_EQUAL(kgram.offset, 0);
  BOOST_CHECK_EQUAL(kgram.mismatches, 1);
}

BOOST_AUTO_TEST_CASE(test_Kgrams)
{
  Kgrams kgrams;
  kgrams += create_kgram(2, 0);  // add a kgram
  kgrams += create_kgram(3, 1);  // another kgram

  int count = 0;
  for(Kgrams::const_iterator kgrams_it = kgrams.begin(); kgrams_it != kgrams.end(); 
      kgrams_it++)
    {
      int offset = count == 0 ? 2 : 3;
      int mismatches = count == 0 ? 0 : 1;
	  
      Kgram kgram = *kgrams_it;
      BOOST_CHECK_EQUAL(kgram.offset, offset);
      BOOST_CHECK_EQUAL(kgram.mismatches, mismatches);

      count++;
    }

  BOOST_CHECK_EQUAL(kgrams.size(), 2);
  Kgrams::iterator kgrams_it = kgrams.begin();
  kgrams_it = kgrams.erase(kgrams_it);
  BOOST_CHECK_EQUAL(kgrams.size(), 1);
  kgrams_it = kgrams.erase(kgrams_it);
  BOOST_CHECK(kgrams.empty());
}

BOOST_AUTO_TEST_CASE(test_TrieMetadata)
{
  TrieMetadata metadata;
  Kgrams kgrams;
  Kgram kgram = create_kgram(2, 0);
  kgrams += kgram;
  kgram = create_kgram(1, 1);
  kgrams += kgram;
  metadata[13] = kgrams;
  BOOST_CHECK_EQUAL(metadata.size(), 1);
}

BOOST_AUTO_TEST_CASE(test_compute_metadata)
{  
  Trie trie = create_trienode();

  ublas::matrix<double > kernel = ublas::zero_matrix<double >(3,3);

  ublas::matrix<int > training_dataset = load_training_dataset("data/unittest_dataset1.txt");

  compute_metadata(trie,
		   2,  // 2-mers
		   training_dataset
		   );

  for(TrieMetadata::const_iterator metadata_it = trie->metadata.begin();
      metadata_it != trie->metadata.end(); 
      metadata_it++)
    {
      Kgrams kgrams = metadata_it->second;
      BOOST_CHECK_EQUAL(kgrams.size(), 3);  
    }

  BOOST_CHECK_EQUAL(trie->metadata.size(), 3);

  // another case
  trie = create_trienode();
  training_dataset = load_training_dataset("data/unittest_dataset2.txt");

  compute_metadata(trie, 1, training_dataset);
 
  BOOST_CHECK_EQUAL(trie->metadata[0].size(), 5);

  for(unsigned int j = 0; j < trie->metadata[0].size(); j++)
    {
      BOOST_CHECK_EQUAL(trie->metadata[0][j].offset, j);
      BOOST_CHECK_EQUAL(trie->metadata[0][j].mismatches, 0);
    }

  // another case
  trie = create_trienode();

  process_node(trie, 1, 0, training_dataset);
  compute_metadata(trie, 1, training_dataset);

  // root node
  BOOST_CHECK_EQUAL(trie->metadata[0].size(), 5);

  for(unsigned int j = 0; j < trie->metadata[0].size(); j++)
    {
      BOOST_CHECK_EQUAL(trie->metadata[0][j].offset, j);
      BOOST_CHECK_EQUAL(trie->metadata[0][j].mismatches, 0);
    }

  // left child
  Trie lchild = create_trienode(0, trie);
  process_node(lchild, 1, 0, training_dataset);

  BOOST_CHECK_EQUAL(lchild->metadata[0].size(), 3);

  for(unsigned int j = 0; j < lchild->metadata[0].size(); j++)
    {
      BOOST_CHECK_EQUAL(lchild->metadata[0][j].offset, (j > 0 ? j + 1: 0));
      BOOST_CHECK_EQUAL(lchild->metadata[0][j].mismatches, 0);
    }


  // right child
  Trie rchild = create_trienode(1, trie);
  process_node(rchild, 1, 0, training_dataset);

  BOOST_CHECK_EQUAL(rchild->metadata[0].size(), 2);

  for(unsigned int j = 0; j < rchild->metadata[0].size(); j++)
    {
      BOOST_CHECK_EQUAL(rchild->metadata[0][j].offset, (j > 0 ? j + 3: 1));
      BOOST_CHECK_EQUAL(rchild->metadata[0][j].mismatches, 0);
    }
}


BOOST_AUTO_TEST_CASE(test_traverse)
{
  Trie trie = create_trienode();
  TrainingDataset training_dataset = load_training_dataset("data/unittest_dataset3.txt");

  // initialize kernel to zero
  ublas::matrix<double > kernel = ublas::zero_matrix<double >(training_dataset.size1(),
							      training_dataset.size1());

  int nkmers = traverse(trie, 2, 4, 0, training_dataset, kernel);

  BOOST_CHECK_EQUAL(nkmers, 8);
}


BOOST_AUTO_TEST_CASE(test_digits_data)
{
  // instantiate trie object
  Trie trie = create_trienode();

  // load data
  unsigned int n_samples = 100;
  TrainingDataset training_dataset;
  training_dataset = load_training_dataset("data/digits_data.dat", 100);
  int l = 16;  // size of alphabet
  int k = 4;
  int m = 1;  // max number of allowable mismatches for 'similar' tokens

  // intialize kernel to zero
  ublas::matrix<double > kernel = ublas::zero_matrix<double >(n_samples, n_samples);
				     
  // fit
  int nkmers = traverse(trie, l, k, m, training_dataset, kernel);
  
  // normalize kernel to remove the 'bias of length'
  Combinatorics::normalize_kernel(kernel);

  // dump kernel unto disk
  std::ofstream kernelfile;
  kernelfile.open("data/digits_kernel.dat");
  kernelfile << kernel;
  kernelfile.close();
}  
