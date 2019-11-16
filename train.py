import sys
import os
import time
import pickle
import numpy as np
from gensim.models import Word2Vec
from argparse import ArgumentParser

p = ArgumentParser("Train vector representations")

p.add_argument("corpus", help="Path to tokenized corpus")
p.add_argument("output", help="Path to vector output directory")
p.add_argument("fd", help="Path to file with pickled conditional freq dist")
p.add_argument("rel_min", type=int, help="Minimum frequency for the smallest geography")
p.add_argument("--dims", type=int, help="Number of dimensions for each vector", default=100)
p.add_argument("--window", type=int, help="Max distance between the current and predicted word within a sentence", default=5)
p.add_argument("--workers", type=int, help="Number of cores", default=1)

args = p.parse_args()

cfd = pickle.load(open(args.fd, "rb"))
geos = cfd.keys()

# Number of tokens per geography
num_toks = np.array([fd.N() for fd in cfd.values()])

# Minimum counts by geography
print("==Vec min counts==")
mins = np.ceil((num_toks/np.min(num_toks))*args.rel_min).astype(int)
for i, geo in enumerate(geos):
  print(str(geo) + "\t" + str(mins[i]))
print()

# Build models
for i, geo in enumerate(geos):
  start = time.time()
  print("Building vectors for", geo)
  m = Word2Vec(corpus_file=os.path.join(args.corpus, geo), size=args.dims, window=args.window, min_count=mins[i], workers=args.workers)
  print("Built in", round(time.time() - start, 2), "seconds")
  m.wv.save_word2vec_format(os.path.join(args.output, geo))
