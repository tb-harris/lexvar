# ISSUES:
# case
# sentence vs review as sequence

# Separate reviews into separate metro area files,
# tokenize with nltk tokenizer

import sys
import os
import json
import math
import nltk
'''
try:
  nltk.find('punkt')
except LookupError:
  nltk.download('punkt')
'''
def dist(p1, p2):
  '''
  Finds distance between two ordered pairs
  '''
  return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
  
def find_metro(lat, lon, metro_centroids):
  '''
  Finds closest metro point to longitude
  Args:
   * lat - latitude ordered pair
   * lon - longitude ordered pair
   * metro_centroids - list of ordered pairs corresponding to
   *                    point for each metro
  Returns:
   * int - ID of closest metro based on position in centroid list
  '''
  closest = (-1, 1000) # ID, distance
  for i, p in enumerate(metro_centroids):
    d = dist((lat, lon), p)
    if d < closest[1]:
      closest = (i, d)
  return closest[0]

# 0 Las Vegas-Henderson-Paradise, NV
# 1 Phoenix-Mesa-Scottsdale, AZ
# 2 Charlotte-Concord-Gastonia, NC-SC
# 3 Pittsburgh, PA
# 4 Cleveland-Elyria, OH
# 5 Madison, WI
# 6 Champaign-Urbana, IL
# 7 Calgary, AB
# 8 Toronto, ON
# 9 Montreal, QC

metro_names = [
'Las_Vegas-Henderson-Paradise_NV',
'Phoenix-Mesa-Scottsdale_AZ',
'Charlotte-Concord-Gastonia_NC-SC',
'Pittsburgh_PA',
'Cleveland-Elyria_OH',
'Madison_WI',
'Champaign-Urbana_IL',
'Calgary_AB',
'Toronto_ON',
'Montreal_QC'
]

metro_centroids = [
  (36.214257,	-115.013812),
  (33.185765,	-112.067862),
  (35.187295,	-80.867491),
  (40.434338,	-79.828061),
  (41.760392,	-81.724217),
  (43.084288,	-89.597178),
  (40.234489,	-88.298623),
  (51.025327, -114.049868),
  (43.654000, -79.387200),
  (45.497216, -73.610364)
]

data_path = sys.argv[1]
output_path = sys.argv[2]

# Build business to metro cluster mappings
business2metro = dict()
for s_json in open(os.path.join(data_path, "business.json")):
  business = json.loads(s_json.strip())
  metro_id = find_metro(business['latitude'], business['longitude'], metro_centroids)
  business2metro[business['business_id']] = metro_id

print("Clustered businesses.")

# One file for each metro area
corpus = [open(os.path.join(output_path, metro_name), "w") for metro_name in metro_names]

# Build tokenized text files of reviews
for i, s_json in enumerate(open(os.path.join(data_path, "review.json"))):
  if (i + 1) % 10000 == 0:
    print("Processing review ", i + 1)
  review = json.loads(s_json.strip())
  text = review['text']
  tokens = nltk.word_tokenize(text.lower())
  corpus[business2metro[review['business_id']]].write(" ".join(tokens) + "\n")
