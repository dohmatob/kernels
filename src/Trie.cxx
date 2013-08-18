/*!
  \file Trie.cxx
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


Combinatorics::Chunk Combinatorics::create_chunk(int offset, 
						 int length, 
						 int mismatches)
{
  Combinatorics::Chunk *chunk = new Chunk;
  chunk->offset = offset;
  chunk->length = length;
  chunk->mismatches = mismatches;

  return *chunk;
}
  
void Combinatorics::add_child(Combinatorics::Trie& parent, 
			      Trie& child)
{
  parent->children[child->label] = child;
} 

Combinatorics::Trie Combinatorics::create_trienode(int label)
{
  Combinatorics::Trie trie = new Combinatorics::TrieNode;
  trie->label = label;
  trie->parent = 0;
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
  trie->metadata = parent->metadata;
  trie->rootpath << parent->rootpath.str();
  trie->rootpath << "[";
  trie->rootpath << label;
  trie->rootpath << "]";

  add_child(parent, trie);
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

// int Combinatorics::display_trie(const Combinatorics::Trie& trie, std::string& padding)
// {
//   int nodecount = 0;

//   if(trie)
//     {
//       Combinatorics::display_trienode(trie, trie->children.size(), padding);
      
//       nodecount++;
//       padding += " ";
  
//       int count = 0;
//       for(Combinatorics::TrieNodeChildren::const_iterator children_it = trie->children.begin(); 
// 	  children_it != trie->children.end(); children_it++) 
// 	{
// 	  count++;
// 	  std::string child_padding(padding);
// 	  child_padding += (count == trie->children.size()) ? " " : "|";
// 	  nodecount += display_trie(children_it->second, child_padding);
// 	}
//     }

//   return nodecount;
// }

// int Combinatorics::display_trie(const Combinatorics::Trie& trie)
// {
//   std::string padding(" ");
  
//   return Combinatorics::display_trie(trie, padding);
// }
  
void Combinatorics::compute_metadata(Combinatorics::Trie& trie, 
				     int d, 
				     Combinatorics::TrainingDataset& training_dataset)
{
  for(unsigned index = 0; index < training_dataset.size(); index++)
    {
      Chunks chunks;
      BOOST_ASSERT(training_dataset[index].size() - d + 1 > 0);
      for(int offset = 0; offset < training_dataset[index].size() - d + 1; offset++)
	{
	  chunks.push_back(create_chunk(offset, 0, 0));
	}
      trie->metadata[index] = chunks;
    }
}

std::ostream& Combinatorics::operator<<(std::ostream& cout, 
					const Combinatorics::Chunk& chunk)
{
  cout << "(" << chunk.offset << "," << chunk.length << "," << chunk.mismatches << ")";

  return cout;
}

std::ostream& Combinatorics::operator<<(std::ostream& cout, 
					const Combinatorics::Chunks& chunks)
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

std::ostream& Combinatorics::operator<<(std::ostream& cout, 
					const Combinatorics::TrieMetadata& metadata)
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
  
void Combinatorics::trim_bad_chunks(Combinatorics::Trie& trie, 
				    int index, 
				    int m, 
				    Combinatorics::TrainingDataset& training_dataset)
{
  Combinatorics::Chunks chunks = trie->metadata[index];
  Combinatorics::Chunks::iterator chunks_it = chunks.begin();
  while(chunks_it != chunks.end())
    {
      // update mismatch count for chunk
      chunks_it->mismatches += training_dataset[index][chunks_it->offset + 
						       chunks_it->length] != trie->label ? 1 : 0;

      // update chunk length
      chunks_it->length++;
      
      // delete this chunk if we have hit more than m mismatches with it
      if(chunks_it->mismatches > m)
	{
	  chunks_it = chunks.erase(chunks_it);
	  continue;
	}
      
      // proceed to next chunk
      chunks_it++;
    }

  // update metadata entry
  trie->metadata[index] = chunks;
}
  
bool Combinatorics::inspect(Combinatorics::Trie& trie, 
			    int d, 
			    int m, 
			    Combinatorics::TrainingDataset& training_dataset)
{
  if(is_root(trie))
    {
      // create meta data for root node (this will be copied to children nodes as they're created
      Combinatorics::compute_metadata(trie, d, training_dataset);
    }	  
  else
    {
      // update metadata
      Combinatorics::TrieMetadata::iterator metadata_it = trie->metadata.begin();
      while(metadata_it != trie->metadata.end())
	{
	  int index = metadata_it->first;

	  // trim-off all chunks that have hit the mismatch threshold (m)
	  Combinatorics::trim_bad_chunks(trie, index, m, training_dataset);

	  Combinatorics::Chunks chunks = metadata_it->second;
	  if(chunks.empty())
	    {
	      // no need keeping empty chunks
	      trie->metadata.erase(index);
	    }
	  
	  // proceed 
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
      for(int j = 0; j < kernel.size2(); j++)
	{
	  if(j != i)
	    {
	      double quotient = std::sqrt(kernel(i,i)*kernel(j,j));
	      kernel(i,j) /= (quotient > 0 ? quotient : 1);
	    }
	}
    }

  // set k = 1 for all diagonal cells (x, x)
  for(int i = 0; i < kernel.size1(); i++)
    {
      kernel(i,i) = 1;
    }
}

void Combinatorics::update_kernel(Combinatorics::Trie& trie, 
				  int m, 
				  ublas::matrix<double >& kernel)
{
  // compute source weights for surving k-mers
  ublas::vector<double > source_counts = ublas::scalar_vector<double >(kernel.size1(), 0);  
  for(Combinatorics::TrieMetadata::const_iterator metadata_it = trie->metadata.begin(); 
      metadata_it != trie->metadata.end(); metadata_it++)
    {
      int index = metadata_it->first;
      Chunks chunks = metadata_it->second;

      // update source count
      source_counts[index] += chunks.size();
    }
  
  // compute kmer weighting factor
  double kmer_weight = std::pow(trie->metadata.size(),
				-2*std::log(trie->metadata.size()));

  // update all kernel entries corresponding to the k-kmer
  kernel += outer_prod(source_counts, source_counts)*kmer_weight;
}

std::ostream& Combinatorics::operator<<(std::ostream& cout, 
					const Combinatorics::Trie& trie)
{
  if(!is_root(trie))
    {
      cout << trie->rootpath.str() + "{" << trie->metadata.size() << "}";
    }
  
  return cout;
}

// void Combinatorics::display_trienode(const Combinatorics::Trie& trie, 
// 				     int d, 
// 				     const std::string& padding)
// {
//   if(trie)
//     {
//       if(is_root(trie))
// 	{
// 	  std::cout << "//\r\n" << (d > 0 ? " \\" : "") << std::endl;
// 	}
//       else
// 	{
// 	  std::cout << padding.substr(0, padding.length() - 1) + "+-" << trie << std::endl;
// 	}
//     }
// }
  
int Combinatorics::expand(Combinatorics::Trie& trie, 
			   int k, 
			   int d, 
			   int m, 
			   Combinatorics::TrainingDataset& training_dataset,
			   ublas::matrix<double >& kernel, 
			   std::string& padding)
{
  int nkmers = 0;

  // recompute metadata of node
  bool go_ahead = inspect(trie, d, m, training_dataset);

  if(is_root(trie))
    {
      std::cout << "//\r\n \\" << std::endl;
    }
  else
    {
      std::cout << padding.substr(0, padding.length() - 1) + "+-" << trie << std::endl;
  }

  padding += " ";

  if(go_ahead)
    {
      if(k == 0)
	{
	  // increment number of kmers
	  nkmers++;

	  // updata kernel
	  Combinatorics::update_kernel(trie, m, kernel);
	}
      else
	{
	  for(int j = 0; j < d; j++)
	    {
	      std::cout << padding + "|" << std::endl;
	      std::string child_padding(padding);
	      child_padding += (j + 1 == d) ? " " : "|";
	      create_trienode(j, trie);
	      nkmers += expand(trie->children[j], k - 1, d, m, training_dataset, kernel, child_padding);
	    }
	}
    }
  else
    {
      // liberate memory occupied by node and all its descendants
      Combinatorics::destroy_trie(trie);
    }

  return nkmers;
}

int Combinatorics::expand(Combinatorics::Trie& trie, 
			   int k, 
			   int d, 
			   int m, 
			   Combinatorics::TrainingDataset& training_dataset,
			   ublas::matrix<double >& kernel)
{
  // intantiate padding
  std::string padding(" ");

  // delegate to other version
  return expand(trie, k, d, m, training_dataset, kernel, padding);
}
   
Combinatorics::TrainingDataset Combinatorics::load_training_dataset(const std::string& filename)
{
  // XXX check that filename exists

  std::vector<std::vector<int> > training_dataset;
  std::ifstream input(filename.c_str());
  std::string lineData;
  int n = 0;
  int m;

  while(std::getline(input, lineData))
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
  
  return training_dataset;
}











