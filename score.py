from aligned_vectors import AlignedVectors
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
  
def words_by_matrix(vocab, matrix, topn, display = False):
  idxs_topn_unsorted = np.argpartition(matrix, -topn, axis=None)[-topn:]
  idxs_topn_sorted = idxs_topn_unsorted[np.argsort(matrix.take(idxs_topn_unsorted))]
  if len(matrix.shape) == 1:
    words = [vocab[idxs_topn_sorted[-i]] for i in range(1, topn + 1)]
  else:
    words = [get_pair(vocab, idxs_topn_sorted[-i]) for i in range(1, topn + 1)]
  if display:
    for i, w in enumerate(words):
      print(i, w, matrix.take(idxs_topn_sorted[-(i + 1)]))
      
  return words

if len(sys.argv) == 1:
  x = AlignedVectors("vecs_aligned_nomin", "models_nomin", 50)
else:
  x = pickle.load(open(sys.argv[1], "rb"))
  
vocab = np.array(x.vocab)
word2id = {w: i for i, w in enumerate(x.vocab)}

# 
even_dist = np.ones(x.num_geos) / np.linalg.norm(x.num_geos)
probs_norm = x.probs.vectors / np.linalg.norm(x.probs.vectors, axis=1).reshape(-1, 1)

# Variability [word]
# Score of to what degree geography conditions probability of word
# variability = np.mean(np.abs(x.pmi.vectors), axis = 1)
variability = 1 - np.matmul(probs_norm, even_dist)

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

# Similarity [word pair]
# Score that reflects the similarity between 
# Build matrix of normalized vectors for each word in most dominant space
vectors_dominant = np.array([x[np.argmax(x.pmi[w])].word_vec(w)/np.linalg.norm(x[np.argmax(x.pmi[w])].word_vec(w)) for w in vocab])
similarity = np.matmul(vectors_dominant, vectors_dominant.transpose())
# del vectors_dominant
# S = similarity
# del similarity

# Complementarity [word pair]
complementarity = np.zeros((len(vocab), len(vocab)))
for i, w in enumerate(vocab):
  probs_additive = x.probs.vectors + x.probs.vectors[i]
  probs_additive_normed = probs_additive / np.linalg.norm(probs_additive, axis=1).reshape(-1, 1)
  complementarity[i] = np.matmul(probs_additive_normed, even_dist)

'''  
# Simple feature intersection
n = .3
w_v = words_by_matrix(vocab, variability, int(np.size(variability)*n))
w_sp = words_by_matrix(vocab, 1 - specificity, int(np.size(specificity)*n))
wp_sm = words_by_matrix(vocab, similarity, int(np.size(similarity)*n))
wp_c = words_by_matrix(vocab, complementarity, int(np.size(complementarity)*n))

intr = [(w1, w2) for (w1, w2) in wp_c if w1 in w_sp and w2 in w_sp and w1 in w_v and w2 in w_v and (w1, w2) in wp_sm]
print(intr)
'''
  
# Feature weighting

w_var = 1
w_spec = 1
w_sim = 1
w_comp = 1

# variability[w1] + variability[w2] + specificity[w1] + specificity[w2] + similarity[w1, w2] + complementarity[w1, w2]
S = w_sim*similarity + w_comp*complementarity + (w_var/2)*variability + (w_var/2)*variability.reshape((-1, 1)) + (w_spec/2)*(1 - specificity) + (w_spec/2)*(1 - specificity).reshape((-1, 1))

# eliminate symmetrical 
S *= 1 - np.tril(np.ones((len(vocab), len(vocab))))
top500 = words_by_matrix(vocab, S, 500, True)
