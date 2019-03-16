import sys
import os
import time
from gensim.models import Word2Vec

corpus_path = sys.argv[1]
output_path = sys.argv[2]
model_path = sys.argv[3]
dimensions = int(sys.argv[4])
min_freq = int(sys.argv[5])

# Build models
for f_name in os.listdir(corpus_path):
  start = time.time()
  print("Building vectors for", f_name)
  m = Word2Vec(corpus_file=os.path.join(corpus_path, f_name), size=dimensions, window=5, min_count=min_freq, workers=4, sorted_vocab=1)
  print("Built in", round(time.time() - start, 2), "seconds")
  m.wv.save(os.path.join(model_path, f_name))
  m.wv.save_word2vec_format(os.path.join(output_path, f_name))
