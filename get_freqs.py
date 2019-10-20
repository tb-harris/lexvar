import os
from glob import glob
import pickle
import nltk
from nltk.tokenize.regexp import WhitespaceTokenizer, BlanklineTokenizer
from nltk.corpus import PlaintextCorpusReader
from argparse import ArgumentParser

p = ArgumentParser("Analyze tokenized geographic review files")

p.add_argument("corpus_dir", help="Path to directy of tokenized Yelp reviews with one file for each geography")
p.add_argument("fd_path", help="Path to save conditional freq dist pickle")

args = p.parse_args()

file_names = [os.path.basename(fn) for fn in glob(os.path.join(args.corpus_dir, "*"))]

corpus = PlaintextCorpusReader(
  args.corpus_dir,
  file_names,
  word_tokenizer=WhitespaceTokenizer(),
  sent_tokenizer=BlanklineTokenizer())
  
print("Read")

cfd = nltk.ConditionalFreqDist(
  (geography, word)
  for geography in corpus.fileids()
  for word in corpus.words(geography))
  
print("Built fd")
  
pickle.dump(cfd, open(args.fd_path, "wb"))
