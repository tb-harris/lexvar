__PLT = True

import json
from collections import Counter
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import pyqtgraph as pg

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

CENTROIDS = [
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

def dist(p1, p2):
  return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
  
def find_metro(lat, lon):
  closest = (-1, 1000) # ID, distance
  for i, p in enumerate(CENTROIDS):
    d = dist((lat, lon), p)
    if d < closest[1]:
      closest = (i, d)
  return closest[0]

def display_counts(counter):
  for item, count in counter.most_common():
    print(str(item) + "\t" + str(count))
  print("\n\n")

metro2color = generate_colors(len(CENTROIDS))
lats = []
lons = []
cols = []

cities = Counter()
states = Counter()
metros = Counter()
metro_cities = defaultdict(Counter)
business2metro = dict()


for l in open("dataset/business.json"):
  business = json.loads(l.strip())
  cities[business['city'], business['state']] += 1
  states[business['state']] += 1
  metro_id = find_metro(business['latitude'], business['longitude'])
  metros[metro_id] += 1
  
  business2metro[business['business_id']] = metro_id
  
  lats.append(business['latitude'])
  lons.append(business['longitude'])
  cols.append(pg.mkBrush(*metro2color[metro_id]))
  metro_cities[metro_id][business['state']] += 1

display_counts(cities)
display_counts(states)
display_counts(metros)

'''
for i in range(len(CENTROIDS)):
  print(i)
  display_counts(metro_cities[i])
'''
