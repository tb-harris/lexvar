from gensim.models import KeyedVectors
from gensim.models.keyedvectors import BaseKeyedVectors
from nltk.probability import FreqDist
from nltk.probability import LaplaceProbDist
from collections import OrderedDict
from itertools import chain
import numpy as np
import nltk.probability
import os
import pickle

class AlignedVectors:
  def __init__(self, vecs_path, data_path, min_count):
    '''
    Initialize class to track aligned geographic vector spaces
    
    Arguments:
     * vecs_path - Path to directory containing word2vec-style text files
     * data_path - Path to directory of text files ofwhite-spaced separated
       tokens vectors are derived from, with matching geography file names
     * min_count - The probability of a word with this frequency in the geography
       with the least amount of training data becomes a lower bound on the
       a word's maximum p(word|geo) for inclusion. MUST be at least word2vec
       minimum frequency
    '''
    self.vector_spaces = []
    self.name2id = dict()
    self.id2name = []
    freq_dists = []
    prob_dists = []
    
    for f_name in os.listdir(vecs_path):
      self.name2id[f_name] = len(self.id2name)
      self.id2name.append(f_name)
      
      # Add vector space, frequency distribution for geography
      self.vector_spaces.append(KeyedVectors.load_word2vec_format(os.path.join(vecs_path, f_name)))
      print("Loaded " + f_name + " vectors")
      freq_dists.append(FreqDist(chain(*[l.split() for l in open(os.path.join(data_path, f_name))])))
      print("Built " + f_name + " frequency distribution")
      
    self.num_geos = len(self.id2name)
    
    # p(word|geography) distributions for each geography, with Laplace smoothing
    prob_dists = [LaplaceProbDist(fd) for fd in fr]
    print("Built probability distributions")
    
    # Find the probability of a word with frequency min_count in the geographic region
    # with the least data
    min_prob = min(prob_dists, key = lambda pd: pd.freqdist.N()).prob(None)*(min_count + 1)
    
    # Build vocab from items whose probs in most overrepresented vocabularies
    # exceed threshold
    # Allows exclusion of low-probability items that is not biased against geographies
    # with less associated data
    self.vocab = list({w for pd in prob_dists for w in pd.samples() if pd.prob(w) >= min_prob})
    print("Loaded vocabulary")
    
    self.geo_probs = BaseKeyedVectors(self.num_geos)
    geo_probs.add(self.vocab, [np.array([pd[w] for pd in prob_dists]) for w in self.vocab])
    print("Built probability vectors")
    
    # pmi = log(p(word|geo)/p(word)) = log(p(word|geo)) - log(p(word))
    # Let p(word) = avg p(word|geo) over all geographies
    # allows equal weighting of each geographic vector space regardless of token count
    pmi = np.log(geo_probs.vectors()) - np.log(geo_probs.vectors().mean(axis=1).reshape(-1, 1))
    self.geo_pmi = KeyedVectors(self.num_geos)
    self.geo_pmi.add(vocab, pmi)
    print("Built PMI vectors")
    
    # log p(word, geo)/(p(word)p(geo)) = log p(word|geo)/p(word) = log(p(word|geo)) - log(p(word)) = log(p(word|geo)) - log(c(word)) + log(c(tokens))
    # pmi = np.log(probs) - np.log(type_freqs_vec).reshape(-1, 1) + log_num_tokens
    
  def save(self, f_path):
    pickle.dump(self, open(f_path, "wb"))
    
  def __iter__(self):
    return iter(self.vector_spaces)
  
  def __getitem__(self, key):
    if isinstance(key, str):
      return self.vector_spaces[self.name2id[key]]
    return self.vector_spaces[key]
    
def test():
  print("Running test...")
  x = AlignedVectors("vecs_aligned_nomin", "corpus", 100)
  x.save("av.bin")
  
if __name__ == "__main__":
  test()
