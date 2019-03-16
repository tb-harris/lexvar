from aligned_vectors import AlignedVectors
import numpy as np
import sys
import pickle
from collections import OrderedDict
x = None

if len(sys.argv) == 1:
  x = AlignedVectors("vecs_aligned_nomin", "models_nomin")
else:
  x = pickle.load(open(sys.argv[1], "rb"))

CUTOFF_PMI_UNDERREP = np.log(.6)
CUTOFF_PMI_OVERREP = np.log(1.2)
SIM_THRESHOLD = .5
SIM_TOPN_THRESHOLD = 1000
MIN_FREQ = 10000

num_geos = len(x.id2name)

log_num_tokens = np.log(np.sum(x.token_counts))
def pmi(p_word_geo, count_word):
  return np.log(p_word_geo) - np.log(count_word) + log_num_tokens

poss_words = OrderedDict()

# Iterate over all words
for word, freq in x.type_freqs.items():
  if freq < MIN_FREQ:
    break
  v_pmi = x.geo_pmi[word] # Vector of pointwise mutual information for each word, geo
  geos_underrep = np.where(v_pmi <= CUTOFF_PMI_UNDERREP)[0]
  geos_overrep = np.where(v_pmi >= CUTOFF_PMI_OVERREP)[0]
  geos_0plus = np.where(v_pmi >= 0)[0]
  
  # Skip word if it does not appear to be a possible dominant lexical variation
  if not (len(geos_underrep) > 0 and len(geos_underrep) < 5):
    continue
    
  # Average of embeddings in spaces with above-average representation
  v_dominant = np.mean([x.vector_spaces[i][word] for i in np.where(v_pmi >= 0)[0]], axis=0)
  
  # List of sets of possible variants in each space where word is underrepresented
  complements = []
  
  print(freq) # progress output
  
  # Iterate over geography index, geographic vector spaces
  for g, V in enumerate(x.vector_spaces):
    # Skip spaces where vector is not underreepresented
    if v_pmi[g] > CUTOFF_PMI_UNDERREP:
     continue
     
    complements.append(set())
     
    for w_sim, sim in V.similar_by_vector(v_dominant, topn=SIM_TOPN_THRESHOLD):
      if sim < SIM_THRESHOLD:
        break
        
      # Prevents inclusion of word itself as variant
      if w_sim == word:
        continue
      
      # TODO: Attempts to exclude words with region-specific references by comparing geographic distribution of surrounding words

      
      if x.geo_pmi[w_sim][g] >= CUTOFF_PMI_OVERREP:
        pmi_w = x.geo_pmi[word]
        pmi_sim = x.geo_pmi[w_sim]
        pmi_combined = pmi(x.geo_probs[word] + x.geo_probs[w_sim], x.type_freqs[word] + x.type_freqs[w_sim])
        
        #if pmi_combined[g] > pmi_w[g] and pmi_combined[g] < pmi_sim[g]:
        if pmi_combined.std() < pmi_w.std() and pmi_combined.std() < pmi_sim.std():
          print("included", word, w_sim)
          complements[-1].add((w_sim, pmi_mod))
        else:
          print("excluded", word, w_sim)
  
  C = set.intersection(*complements)
  for w_sim in C:
    pmi_sim = x.geo_pmi[w_sim]
    pmi_combined = pmi(x.geo_probs[word] + x.geo_probs[w_sim], x.type_freqs[word] + x.type_freqs[w_sim])
    
    pmi_mod = (pmi_w.std() - pmi_combined.std()) + (pmi_sim.std() - pmi_combined.std())
    
    numpy.array([numpy.array(x.vector_spaces[i].similar_by_vector(x.vector_spaces[j]. for j in range(num_geos)]) for i in range(num_geos)])
    
    poss_words[word] = C
          
        
    
  #std = pmi.std() # Standard deviation of PMI vec
  #poss_words.append((std, word, freq))
    
# print((sorted(poss_words.items(), key = lambda w : len([s for s in poss_words[w][1] if len(s) > 0])/len(poss_words[w][))))

