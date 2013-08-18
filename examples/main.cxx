/*!
  \file main.cxx
  \author DOHMATOB Elvis Dopgima
  \brief Main demo file
*/

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "Trie.h"

using namespace Combinatorics;

int main(int argc, char *argv[])
{
  // sanitize command-line
  if(argc < 6)
    {
      printf("Usage: %s <trie_depth> <number_of_allowable_mismatches> "
	     "<size_of_alphabet> <input_file> <output_file>\r\n",
	     argv[0]);
      printf("Example:\r\n%s 6 1 16 data/digits_data.dat data/digits_kernel.dat\r\n",
	     argv[0]);

      return 1;
    }

  int k = atoi(argv[1]);
  int m = atoi(argv[2]);  // max number of allowable mismatches for 'similar' tokens
  int d = atoi(argv[3]);  // size of alphabet

  // instantiate trie object
  Trie trie = create_trienode();

  // load data
  TrainingDataset training_dataset;
  training_dataset = load_training_dataset(argv[4]);
  training_dataset = TrainingDataset(training_dataset.begin(),
				     training_dataset.begin() + training_dataset.size());

  // initialize kernel to zero
  ublas::matrix<double > kernel = ublas::zero_matrix<double >(training_dataset.size(), 
							      training_dataset.size());
				     
  // estimate kernel (fit)
  printf("k = %i, m = %i, d = %i\r\n\r\n", k, m, d);
  int nkmers = expand(trie, k, d, m, training_dataset, kernel);
  std::cout << nkmers << " " << k << "-mers out of " << std::pow(d, k) << " survived." 
	    << std::endl;
  
  // normalize kernel to remove the 'bias of length'
  Combinatorics::normalize_kernel(kernel);
    
  // dump kernel unto disk
  std::ofstream kernelfile;
  kernelfile.open(argv[5]);
  kernelfile << kernel;
  kernelfile.close();
  std::cout << std::endl << "Mismatch string kernel written to " << argv[5] << std::endl;

  return 0;
}  
