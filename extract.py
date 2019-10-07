# Extract reviews and geographies from
# dataset

import sys
import os
import json
import math
from argparse import ArgumentParser
from time import time

p = ArgumentParser("Extract reviews and features from Yelp corpus")

p.add_argument("dataset_path", help="Path to Yelp dataset directory")
p.add_argument("--review_structure", choices=["merged", "geo"], help="Merge reviews into single file or separate by geography", default="geo")
p.add_argument("--geo_output", choices=["name", "id"], help = "Identify geographies by name or numerical ID", default="name")
p.add_argument("--output_reviews", help="Path to reviews output file (merged mode only)", default="./reviews.untok")
p.add_argument("--output_geo", help="Path to geography feature output file (merged mode only)", default="./geos.feat")
p.add_argument("--output_dir", help="Directory for geographic review files (geo mode only)", default = "./reviews/")
p.add_argument("--limit", help="Only process the first n reviews, for testing", type=int, default=-1)
p.add_argument("--exclude", help="Space-separated list of geographies to exclude", choices=["lasvegas", "phoenix", "charlotte", "pittsburgh", "cleveland", "madison", "calgary", "toronto", "champaign-urbana", "montreal"], nargs="+", default=[])
# p.add_argument("--feat_files", help="Separate reviews into file for each feature value", action='store_true')

args = p.parse_args()

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
# 6 Calgary, AB
# 7 Toronto, ON
# 8 Champaign-Urbana, IL
# 9 Montreal, QC

# Correspondence between each metro name
# and ID
metro_names = [
'Las_Vegas-Henderson-Paradise_NV',
'Phoenix-Mesa-Scottsdale_AZ',
'Charlotte-Concord-Gastonia_NC-SC',
'Pittsburgh_PA',
'Cleveland-Elyria_OH',
'Madison_WI',
'Calgary_AB',
'Toronto_ON',
'Champaign-Urbana_IL',
'Montreal_QC'
]

# Businesses will be grouped by nearest
# centroid by euclidian distance
metro_centroids = [
  (36.214257,	-115.013812),
  (33.185765,	-112.067862),
  (35.187295,	-80.867491),
  (40.434338,	-79.828061),
  (41.760392,	-81.724217),
  (43.084288,	-89.597178),rev
  (51.025327, -114.049868),
  (43.654000, -79.387200),
  (45.497216, -73.610364),
  (40.234489,	-88.298623)
]

metros = metro_names if args.geo_output == "name" else list(range(0, len(metro_names))

if args.review_structure == "merged":
  rev = open(args.output_reviews, "w")
  geo = open(args.output_geo, "w")
else:
  corpus = [open(os.path.join(output_path, metro_name), "w") for metro_name in metros]

# Build business to metro cluster mappings
business2metro = dict()
for s_json in open(os.path.join(args.dataset_path, "business.json")):
  business = json.loads(s_json.strip())
  metro_id = find_metro(business['latitude'], business['longitude'], metro_centroids)
  business2metro[business['business_id']] = metro_id

print("Clustered businesses.")

start = time()
for i, s_json in enumerate(open(os.path.join(args.dataset_path, "review.json"))):
  if (i + 1) % 10000 == 0:
    print("Processing review ", i + 1)
  
  if args.limit >= 0 and i >= args.limit:
    print("Reached limit")
    break
    
  review = json.loads(s_json.strip())
  
  geo_id = business2metro[review['business_id']]
  if geo_id in args.exclude:
    continue
  
  text = review['text'].replace("\n", "\t") + "\n"
  geography = str(geos[geo_id]) + "\n"
  
  if args.review_structure == "merged":
    rev.write(text)
    geo.write(geography)
  else:
    corpus[geo_id].write(text)
  
print(i, "reviews processed in", round(time() - start, 4), "seconds")
