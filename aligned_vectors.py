from gensim.models import KeyedVectors
from collections import Counter
from collections import OrderedDict
import numpy as np
import os
import pickle

SMOOTHING = 1

class AlignedVectors:
  def __init__(self, dir_path, models_path):
    self.vector_spaces = []
    self.counts = []
    self.name2id = dict()
    self.id2name = [] 
    self.SEP = '￨'
    
    self.type_freqs = None
    self.token_counts = None
    
    self.geo_counts = None
    self.geo_probs = None
    self.geo_pmi = None
    
    if dir_path != None:
      for f_name in os.listdir(dir_path):
        self.__add_space(os.path.join(dir_path, f_name))
        
    self.build_geo_vectors(models_path, self.name2id)
  
  def __add_space(self, f_path):
    name = f_path.split(os.path.sep)[-1]
    self.name2id[name] = len(self.vector_spaces)
    self.id2name.append(name)
    vector_space = KeyedVectors.load_word2vec_format(f_path)
    self.vector_spaces.append(vector_space)
      
    print("Added", name)
    
    combined_names = [name + self.SEP + word for word in vector_space.index2word]
      
    self.combined_space.add(combined_names, vector_space.vectors)
    
    print("Added", name, "to combined space")

  def build_geo_vectors(self, models_path, name2id):
    vocabs = []
    dim = 0
    for name in name2id:
      vector_space = KeyedVectors.load(os.path.join(models_path, name))
      dim = vector_space.vector_size
      vocabs.append(vector_space.vocab)
        
    types = set(word for vocab in vocabs for word in vocab.keys())
    type_freqs_unsorted = {word : sum(vocab[word].count if word in vocab else 0 for vocab in vocabs) for word in types}
    self.type_freqs = OrderedDict(sorted(type_freqs_unsorted.items(), key=lambda p: (p[1], p[0]), reverse=True))
    
    # Token counts by geography
    self.token_counts = np.array([sum(word_obj.count for word_obj in vocab.values()) for vocab in vocabs])
    
    
    # Matrix counting occurences of words in each geography
    counts = np.array([np.array([vocabs[i][word].count if word in vocabs[i] else 0 for i in range(len(vocabs))]) for word, freq in self.type_freqs.items()])
    
    # p (word | geo), with additive smoothing
    probs = np.array([np.array([(vocabs[i][word].count + SMOOTHING)/self.token_counts[i] if word in vocabs[i] else 0 for i in range(len(vocabs))]) for word, freq in self.type_freqs.items()])
    
    type_freqs_vec = np.array(list(self.type_freqs.values()))
    log_num_tokens = np.log(np.sum(self.token_counts))
    
    # log p(word, geo)/(p(word)p(geo)) = log p(word|geo)/p(word) = log(p(word|geo)) - log(p(word)) = log(p(word|geo)) - log(c(word)) + log(c(tokens))
    pmi = np.log(probs) - np.log(type_freqs_vec).reshape(-1, 1) + log_num_tokens
    
    self.geo_counts = KeyedVectors(len(vocabs))
    self.geo_probs = KeyedVectors(len(vocabs))
    self.geo_pmi = KeyedVectors(len(vocabs))
    
    self.geo_counts.add(list(self.type_freqs.keys()), counts)
    self.geo_probs.add(list(self.type_freqs.keys()), probs)
    self.geo_pmi.add(list(self.type_freqs.keys()), pmi)
    
  def __iter__(self):
    return iter(self.vector_spaces)
  
  def __getitem__(self, key):
    if isinstance(key, str):
      return self.vector_spaces[self.name2id[key]]
    return self.vector_spaces[key]
