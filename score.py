from aligned_vectors import AlignedVectors
import aligned_vectors
import numpy as np
import pickle
import sys
from scipy.stats import zscore
_n = 10

def n_most_similar_to_given(V, entity1, entities_arr, n):
  '''
  Adaption of gensim similar_to_given to allow for
  top n argument
  '''
  return entities_arr[np.argpartition([V.similarity(entity1, entity) if entity in V.vocab and entity != entity1 else 0 for entity in entities_arr], -n)[-n:]]
  
def get_pair(vocab, idx_flat):
  return (vocab[idx_flat//len(vocab)], vocab[idx_flat%len(vocab)])
  
def words_by_matrix(vocab, matrix, topn = 0, display = True, agg = False):
  idxs_topn_unsorted = np.argpartition(matrix, -topn, axis=None)[-topn:]
  idxs_topn_sorted = idxs_topn_unsorted[np.argsort(matrix.take(idxs_topn_unsorted))]
  
  if len(matrix.shape) == 1:
    words = [vocab[idxs_topn_sorted[-i]] for i in range(1, len(idxs_topn_unsorted) + 1)]
  else:
    words = [get_pair(vocab, idxs_topn_sorted[-i]) for i in range(1, len(idxs_topn_unsorted) + 1)]
  if display:
    if (len(matrix.shape) == 1):
      print("# Word", "Score", "Geo 1 +", "Geo 1 -", sep='\t')
    else:
      print("#", "Word 1", "Word 2", "Score", "Geo 1 +", "Geo 1 -", "Geo 2 +", "Geo 2 -", "Variability", "Specificity", "Similarity", "Complementarity", sep='\t')
    for i, w in enumerate(words):
      if len(w) == 1:
        print(str(i) + " " + w, round(matrix.take(idxs_topn_sorted[-(i + 1)]), 2), sep="\t", end="\t")
      else:
        print(str(i), w[0], w[1], round(matrix.take(idxs_topn_sorted[-(i + 1)]), 2), sep="\t", end = "\t")
        
      geos = [g.rsplit("_", 1)[0].split("-", 1)[0] for g in x.id2name]
      if len(w) == 1:
        print(geos[np.argmax(x.probs_norm[w])], geos[np.argmin(x.probs_norm[w])], sep='\t', end = '\t')
      else:
        print(geos[np.argmax(x.probs_norm[w[0]])], geos[np.argmin(x.probs_norm[w[0]])], geos[np.argmax(x.probs_norm[w[1]])], geos[np.argmin(x.probs_norm[w[1]])], sep='\t', end = '\t')
        
      if agg:
        w0 = word2id[w[0]]
        w1 = word2id[w[1]]
        print(round((variability[w0] + variability[w1])/2, 2), round(1 - (specificity[w0] + specificity[w1])/2, 2), round(similarity[w0, w1], 2), round(complementarity[w0, w1], 2), sep='\t')
      else:
        print()
      
  return words

if len(sys.argv) == 1:
  x = AlignedVectors("vecs_aligned_nomin", "models_nomin", 50)
else:
  x = pickle.load(open(sys.argv[1], "rb"))
  
vocab = np.array(x.vocab)
word2id = {w: i for i, w in enumerate(x.vocab)}

# 
even_dist = np.ones(x.num_geos) / np.linalg.norm(np.ones(x.num_geos))
probs_norm = x.probs.vectors / np.linalg.norm(x.probs.vectors, axis=1).reshape(-1, 1)

x.probs_norm = aligned_vectors.BaseKeyedVectors(x.num_geos)
x.probs_norm.add(x.vocab, probs_norm)

# Variability [word]
# Score of to what degree geography conditions probability of word
# variability = np.mean(np.abs(x.pmi.vectors), axis = 1)
variability = 1 - np.matmul(probs_norm, even_dist)
var_mul = 1/np.max(variability)
variability *= var_mul

# Specificity [word]
# Score that reflects the degree to which similar words condition the probability of words
# Reliant on previously computed variability scores
# Takes a while to compute
if not len(sys.argv) >= 3:
  specificity = np.zeros(len(vocab))
  for i, w in enumerate(vocab):
    neighbors = n_most_similar_to_given(x[np.argmax(x.pmi[w])], w, vocab, _n)
    variability_neighbors = variability[np.array([word2id[nbr] for nbr in neighbors])]
    score = np.mean(variability_neighbors)
    specificity[i] = score
else:
  specificity = pickle.load(open(sys.argv[2], "rb"))
  specificity *= var_mul

# Similarity [word pair]
# Score that reflects the similarity between 
# Build matrix of normalized vectors for each word in most dominant space
vectors_dominant = np.array([x[np.argmax(x.pmi[w])].word_vec(w)/np.linalg.norm(x[np.argmax(x.pmi[w])].word_vec(w)) for w in vocab])
similarity = np.matmul(vectors_dominant, vectors_dominant.transpose())

# Complementarity [word pair]
if not len(sys.argv) >= 4:
  complementarity = np.zeros((len(vocab), len(vocab)))
  for i, w in enumerate(vocab):
    probs_additive = x.probs.vectors + x.probs.vectors[i]
    probs_additive_normed = probs_additive / np.linalg.norm(probs_additive, axis=1).reshape(-1, 1)
    complementarity[i] = np.matmul(probs_additive_normed, even_dist)
else:
  complementarity = pickle.load(open(sys.argv[3], "rb"))
  
#variability[variability > .5] = .4
#variability *= 1/.4
 
# Feature weighting
def score(w_var = 1, w_gen = 1, w_sim = 1, w_comp = 1):
  return ((1 - np.tril(np.ones((len(vocab), len(vocab)))))) * (w_sim*similarity + w_comp*complementarity + (w_var/2)*variability + (w_var/2)*variability.reshape((-1, 1)) + (w_gen/2)*(1 - specificity) + (w_gen/2)*(1 - specificity).reshape((-1, 1)))

ws = words_by_matrix(vocab, score(1, 1, 1, 1), 100000, True, True)

