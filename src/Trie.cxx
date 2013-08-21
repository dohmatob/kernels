/*!
  \file TRie.Cxx
  \author DOHMATOB Elvis Dopgima
  \brief Implementation of Trie.h header file.
*/


#include <boost/assert.hpp>
#include <boost/assign/std/vector.hpp>
#include <sstream>
#include <ctype.h>
#include <fstream>
#include <iomanip>
#include "Trie.h"

using namespace boost::assign;


Combinatorics::Kgram Combinatorics::create_kgram(int offset, 
						 int mismatches)
{
  Combinatorics::Kgram *kgram = new Kgram;
  kgram->offset = offset;
  kgram->mismatches = mismatches;

  return *kgram;
}
  

void Combinatorics::add_child(Combinatorics::Trie& parent, 
			      Trie& child)
{
  // clone parent's data
  child->metadata = Combinatorics::TrieMetadata(parent->metadata);

  // child is one level beyond parent
  child->level = parent->level + 1;

  // parent's full label (concatenation of labels on edges leading
  // from root node) is a prefix to child's the remainder is one
  // symbol, the child's label
  child->rootpath << parent->rootpath.str();
  child->rootpath << "[";
  child->rootpath << child->label;
  child->rootpath << "]";

  // let parent adopt child
  parent->children[child->label] = child;

  // let child adopt parent
  child->parent = parent;
} 


Combinatorics::Trie Combinatorics::create_trienode(int label)
{
  Combinatorics::Trie trie = new Combinatorics::TrieNode;
  trie->label = label;
  trie->level = 0;
  trie->parent = 0;

  return trie;
}


Combinatorics::Trie Combinatorics::create_trienode()
{
  return create_trienode(-1);
}


Combinatorics::Trie Combinatorics::create_trienode(int label, 
						   Combinatorics::Trie& parent)
{
  Combinatorics::Trie trie = new Combinatorics::TrieNode;
  trie->label = label;
  trie->parent = parent;

  if(parent)
    {
      add_child(parent, trie);
    }

  return trie;
} 


bool Combinatorics::is_root(const Combinatorics::Trie& trie)
{
  return trie->parent == 0;
}


void Combinatorics::destroy_trie(Combinatorics::Trie& trie)
{
  if(trie)
    {
       // destroy all children nodes
      for(int j = 0; j < trie->children.size(); j++)
	{
	  Combinatorics::destroy_trie(trie->children[j]);
	}

      // free the node itself
      if(!is_root(trie))
    	{
    	  trie->parent->children.erase(trie->label);
    	}
      else
    	{
    	  delete trie;
    	}
    }
}

  
void Combinatorics::compute_metadata(Combinatorics::Trie& trie, 
				     int k, 
				     Combinatorics::TrainingDataset& training_dataset)
{
  BOOST_ASSERT(training_dataset.size2() - k + 1 > 0);
  for(unsigned index = 0; index < training_dataset.size1(); index++)
    {
      Combinatorics::Kgrams kgrams;
      for(int offset = 0; offset < training_dataset.size2() - k + 1; offset++)
	{
	  kgrams.push_back(create_kgram(offset, 0));
	}
      trie->metadata[index] = kgrams;
    }
}


std::ostream& Combinatorics::operator<<(std::ostream& cout, 
					const Combinatorics::Kgram& kgram)
{
  cout << "(" << kgram.offset << "," << kgram.mismatches << ")";

  return cout;
}


std::ostream& Combinatorics::operator<<(std::ostream& cout, 
					const Combinatorics::Kgrams& kgrams)
{
  Combinatorics::Kgrams::const_iterator kgrams_it = kgrams.begin();
  cout << "[" << kgrams.size() << "](" << (kgrams_it == kgrams.end() ? ")" : "");
  while(kgrams_it != kgrams.end())
    {
      cout << *kgrams_it;
      kgrams_it++;
      cout << (kgrams_it == kgrams.end() ? ")" : ",");
    }
  
  return cout;
}


std::ostream& Combinatorics::operator<<(std::ostream& cout, 
					const Combinatorics::TrieMetadata& metadata)
{
  if(metadata.empty())
    {
      cout << "{DEADEND}";
    }
  else
    {
      cout << "{" << metadata.size() << "}";
      // Combinatorics::TrieMetadata::const_iterator metadata_it = metadata.begin();
      // cout << "{" << (metadata_it == metadata.end() ? "}" : "");
      // while(metadata_it != metadata.end())
      // 	{
      // 	  cout << metadata_it->first << ":" << metadata_it->second;
      // 	  metadata_it++;
      // 	cout << (metadata_it == metadata.end() ? "}" : ",");
      // 	}
    }

  return cout;
}
  

void Combinatorics::trim_bad_kgrams(Combinatorics::Trie& trie, 
				    int index, 
				    int m, 
				    Combinatorics::TrainingDataset& training_dataset)
{
  Combinatorics::Kgrams kgrams = trie->metadata[index];
  Combinatorics::Kgrams::iterator kgrams_it = kgrams.begin();
  while(kgrams_it != kgrams.end())
    {
      // update mismatch count for kgram
      kgrams_it->mismatches += training_dataset(index,
						kgrams_it->offset + trie->level - 1
						) != trie->label ? 1 : 0;

      // delete this kgram if we have hit more than m mismatches with it
      if(kgrams_it->mismatches > m)
	{
	  kgrams_it = kgrams.erase(kgrams_it);
	  continue;
	}
      
      // proceed to next kgram
      kgrams_it++;
    }

  // update metadata entry
  trie->metadata[index] = kgrams;
}

  
bool Combinatorics::process_node(Combinatorics::Trie& trie, 
				 int k, 
				 int m, 
				 Combinatorics::TrainingDataset& training_dataset)
{
  if(is_root(trie))
    {
      // create meta data for root node (this will be copied to children
      // nodes as they're created along)
      Combinatorics::compute_metadata(trie, k, training_dataset);
    }	  
  else
    {
      // update metadata
      Combinatorics::TrieMetadata::iterator metadata_it = trie->metadata.begin();
      while(metadata_it != trie->metadata.end())
	{
	  int index = metadata_it->first;

	  // trim-off all kgrams that have exceeded the mismatch threshold (m)
	  Combinatorics::trim_bad_kgrams(trie, index, m, training_dataset);

	  Combinatorics::Kgrams kgrams = metadata_it->second;
	  if(kgrams.empty())
	    {
	      // no need keeping empty kgrams
	      trie->metadata.erase(index);
	    }
	  
	  // next
	  metadata_it++;
	}
    }

  return !trie->metadata.empty(); 
}


void Combinatorics::normalize_kernel(ublas::matrix<double >& kernel)
{
  // set k(x, y) = k(x, y) / sqrt(k(x, x)k(y, y)), for all non-diagonal cells (x, y)

  for(int i = 0; i < kernel.size1(); i++)
    {
      for(int j = i + 1; j < kernel.size2(); j++)
	{
	  double quotient = std::sqrt(kernel(i, i) * kernel(j, j));
	  kernel(i, j) /= (quotient > 0 ? quotient : 1);
	  kernel(j, i) = kernel(i, j);  // symmetry
	}
    }

  // set k = 1 for all diagonal cells (x, x)
  for(int i = 0; i < kernel.size1(); i++)
    {
      kernel(i, i) = 1;
    }
}


void Combinatorics::update_kernel(Combinatorics::Trie& trie, 
				  int m, 
				  ublas::matrix<double >& kernel)
{
  // i_it points to a pair (i, s_i), where s_i is the list of urviving k-grams
  // of the ith input string, those m-grams of the ith input string which lie
  // within m mismatches of the the k-mer represented by this leaf node
  for(Combinatorics::TrieMetadata::const_iterator i_it = trie->metadata.begin(); 
      i_it != trie->metadata.end(); i_it++)
    {
      // j_it points to some (j, s_j); similary definition as for i and s_i above
      for(Combinatorics::TrieMetadata::const_iterator j_it = trie->metadata.begin(); 
	  j_it != trie->metadata.end(); j_it++)
	{
	  // increment kernel[i, j] by z(n_i, n_j) := exp(-(n_i + n_j)), where n_i
	  // (resp. n_j) is the cardinality of s_i (resp. s_j). A reason for this
	  // damping factor is that: if n_i + n_j is "large", then WLOG n_i is "large"
	  // and so th k-mer represented by the current leaf is rather common (well,
	  // except for a few mismatches here and there...) in the ith input string.
	  // Thus we should prevent this banality from doping the similarity between
	  // the ith input string and others (such faking would happen if we updated
	  // the kernel value with somethig proportional to n_i, say n_i * n_j, for
	  // example)
	  kernel(i_it->first, j_it->first) += std::exp(-(double)(i_it->second.size() + \
								 j_it->second.size()));
	}
    }
}


std::ostream& Combinatorics::operator<<(std::ostream& cout, 
					const Combinatorics::Trie& trie)
{
  if(!is_root(trie))
    {
      cout << trie->rootpath.str() << trie->metadata;
    }
  
  return cout;
}


int Combinatorics::traverse(Combinatorics::Trie& trie, 
			    int l, 
			    int k, 
			    int m, 
			    Combinatorics::TrainingDataset& training_dataset,
			    ublas::matrix<double >& kernel, 
			    std::string& indentation)
{
  int nkmers = 0;

  // recompute metadata of node, and determine it survives
  bool go_ahead = process_node(trie, k, m, training_dataset);

  // display this node
  if(is_root(trie))
    {
      std::cout << "//\r\n \\" << std::endl;
    }
  else
    {
      std::cout << indentation.substr(0, indentation.length() - 1) + "+-" << trie << std::endl;
  }

  // explore node further
  if(go_ahead)
    {
      if(k == 0)  // are we at a leaf ?
	{
	  // increment number of kmers
	  nkmers++;

	  // update kernel
	  Combinatorics::update_kernel(trie, m, kernel);
	}
      else
	{
	  indentation += " ";
	  // recursively expand all children nodes
	  for(int j = 0; j < l; j++)
	    {
	      // compute indentation for child display
	      std::cout << indentation + "|" << std::endl;
	      std::string child_indentation(indentation);
	      child_indentation += (j + 1 == l) ? " " : "|";

	      // bear new child with label j and expand it
	      create_trienode(j, trie);
	      nkmers += traverse(trie->children[j], l, k - 1, m, training_dataset,
				 kernel, child_indentation);
	    }
	}
    }
  else
    {
      // liberate memory occupied by node and all its descendants
      Combinatorics::destroy_trie(trie);
    }

  if(is_root(trie))
    {
      std::cout << nkmers << " " << k << "-mers out of " << std::pow(l, k) << " survived." 
		<< std::endl;
    }

  // return number of surviving leafs (k-mers)
  return nkmers;
}


int Combinatorics::traverse(Combinatorics::Trie& trie, 
			    int l, 
			    int k, 
			    int m, 
			    Combinatorics::TrainingDataset& training_dataset,
			    ublas::matrix<double >& kernel)
{
  // intantiate indentation
  std::string indentation(" ");

  // delegate to other version
  return traverse(trie, l, k, m, training_dataset, kernel, indentation);
}
   

Combinatorics::TrainingDataset Combinatorics::load_training_dataset(const std::string& filename,
								    unsigned int nrows)
{
  // XXX check that filename exists

  std::vector<std::vector<int> > training_dataset;
  std::ifstream input(filename.c_str());
  std::string lineData;
  unsigned int n = 0;
  unsigned int m;

  while(n != nrows && std::getline(input, lineData))
    {
      int d;
      std::vector<int > row;
      std::stringstream lineStream(lineData);

      while (lineStream >> d)
	{
	  row.push_back(d);
	}
	  
      if (n == 0)
	{
          m = row.size();
	}
      
      if (row.size() > 0)
        {
          if (row.size() != m)
            {
              throw "mal-formed matrix line";
            }
	  
          training_dataset.push_back(row);
          n++;
        }
    }

  // convert list of lists into matrix
  // XXX it should be possible to do this with a one-liner
  Combinatorics::TrainingDataset X(n, m);
  for(unsigned int i = 0; i < n; i++)
    {
      for(unsigned int j = 0; j < m; j++)
	{
	  X(i, j) = training_dataset[i][j];
	}
    }

  return X;
}


Combinatorics::TrainingDataset Combinatorics::load_training_dataset(const std::string& filename)
{
  return load_training_dataset(filename, -1);
}






