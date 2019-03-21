'''
Align vectors into a single vector space

'''

import sys
import os
import subprocess
import random

# Random string
n = str(random.randrange(999999999))
print("Experiment Name:", n)

N = 100

def run_muse(dir_muse, src, tgt, output):
  '''
  Run monolingual Muse alignment on given vectors, move re-aligned source
  vectors to new location
  * dir_muse - Directory containing muse implementation
  * src - Text file with word2vec-style embeddings to be aligned
  * tgt - Embeddings to align with
  * output - Path to output aligned source embeddings in word2vec-style text file
  '''
  # Get file names without paths
  src_name = src.split(os.path.sep)[-1]
  tgt_name = tgt.split(os.path.sep)[-1]
  
  subprocess.run(["mkdir", "muse_data"])
  
  # Run muse
  subprocess.run(["python3", os.path.join(dir_muse, "supervised.py"), "--verbose", "1", "--cuda", "0", "--emb_dim", str(N), "--src_emb", src, "--tgt_emb", tgt, "--export", "txt", "--src_lang", src_name, "--tgt_lang", tgt_name, "--dico_train", "identical_char", "--dico_eval", "monolingual_dict.txt", "--exp_path", "muse_data", "--exp_name", n, "--exp_id", src_name])
  
  # Copy target vector to output directory
  subprocess.run(["cp", os.path.join("muse_data", n, src_name, "vectors-" + src_name + ".txt"), os.path.join(output)])
  
blacklist = ["Montreal_QC", "Champaign-Urbana_IL"]

dir_muse, dir_vecs, dir_output = sys.argv[1:]


files = [f_name for f_name in os.listdir(dir_vecs) if f_name not in blacklist]

target = files[0]

subprocess.run(["cp", os.path.join(dir_vecs, target), dir_output])

# Align vector spaces with first vector space
for f_vecs in files[1:]:
  print("Aligning " + f_vecs + " with " + target)
  run_muse(dir_muse, os.path.join(dir_vecs, f_vecs), os.path.join(dir_vecs, target), os.path.join(dir_output, f_vecs))
