'''
Align vectors into a single vector space using
supervised.py script from Facebook MUSE library
'''

import subprocess
import random
from glob import glob
from os import path
from argparse import ArgumentParser


p = ArgumentParser("Align multiple embedding spaces into a single space with Facebook Research's MUSE library")

p.add_argument("dir_vecs", help="Path to directory containing source embeddings spaces (target embeddings will be excluded)")
p.add_argument("dir_output", help="Path to directory for aligned vectors")
p.add_argument("--tgt", help="Path to embedding space to use as target (otherwise arbitrarily selected from sources)", default=None)
p.add_argument("--dir_muse", help="Path to directory containing MUSE implementation", default="./MUSE/")
p.add_argument("--dir_logs", help="Path to directory containing experiment logs and outputs", default="./muse_logs/")
p.add_argument("--dim", type=int, default=100, help="Dimension of source and target embedding spaces")
p.add_argument("--cuda", action="store_true", help="Run on GPU rather than CPU")

args = p.parse_args()

# Random string
n = str(random.randrange(999999999))
print("Experiment Name:", n)

N = 100

def run_muse(dir_muse, src, tgt, output):
  '''
  Run monolingual Muse alignment on given vectors, move re-aligned source
  vectors to new location
  * dir_muse - Directory containing FMUSE implementation
  * src - Text file with word2vec-style embeddings to be aligned
  * tgt - Embeddings to align with
  * output - Path to DIRECTORY to output aligned source and modified (less precise) target embeddings in word2vec-style text file
  '''
  # Get file names without paths
  src_name = src.split(path.sep)[-1]
  tgt_name = tgt.split(path.sep)[-1]
  
  subprocess.run(["mkdir", args.dir_logs])
  subprocess.run(["mkdir", args.dir_output])
  
  subprocess.run([
    "python3", path.join(dir_muse, "supervised.py"),
    "--verbose", "1",
    "--cuda", str(args.cuda),
    "--emb_dim", str(args.dim),
    "--src_emb", src, # Path to source space
    "--tgt_emb", tgt, # Path to target space
    "--export", "txt", # Aligned vectors as text file
    "--dico_train", "identical_char", # Anchor w identical word pairs
    # en-es eval dict with english words mapped to themselves, duplicates removed
    "--dico_eval", "monolingual_dict.txt",
    # Vectors will be output in dir:
    # EXP_PATH/EXP_NAME/EXP_ID
    "--exp_path", args.dir_logs,
    "--exp_name", n,
    "--exp_id", src_name,
    # Output file names
    "--src_lang", src_name,
    "--tgt_lang", tgt_name
  ])
  
  # Copy source, target vector to output directory
  # Target vectors should be effectively the same as original, but are rounded to 3 digits
  subprocess.run(["cp", path.join(args.dir_logs, n, src_name, "vectors-" + src_name + ".txt"), path.join(output, src_name)])
  subprocess.run(["cp", path.join(args.dir_logs, n, src_name, "vectors-" + tgt_name + ".txt"), path.join(output, tgt_name)])
  
# List of vector spaces  
vec_spaces = glob(path.join(args.dir_vecs, "*"))
  
# Get target from args, pick arbitrary embedding space if none provided
if args.tgt:
  tgt = args.tgt
else:
  tgt = vec_spaces[0]
  print("Selected", tgt, "as target embedding space")
  
# Build list of source embeddings, excluding target
srcs = [vec_space for vec_space in vec_spaces if path.normpath(vec_space) != path.normpath(tgt)]

# Align vector spaces with first vector space
for f_vecs in srcs:
  print("Aligning " + f_vecs + " with " + tgt)
  run_muse(args.dir_muse, f_vecs, tgt, args.dir_output)
