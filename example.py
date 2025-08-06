# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:20:24 2023

@author: vinic
"""

#Dimensions judgement 
#
# Required Libraries
import numpy as np

# Fuzzy AHP 
from pyDecision.algorithm import fuzzy_ahp_method
# Fuzzy AHP

# Dataset
dataset = list([
    [ (  1,   1,   1), (1/2, 1/3, 1/4), (1/2, 1/3, 1/4), (1/2, 1/3, 1/4), (  2,   3,   4), (  2,   3,   4), (1/2, 1/3, 1/4) ], #g1 Governance
    [ (  2,   3,   4), (  1,   1,   1), (1/2, 1/3, 1/4), (1/2, 1/3, 1/4), (  2,   3,   4), (  2,   3,   4), (  2,   3,   4) ], #g2 Security
    [ (  2,   3,   4), (  2,   3,   4), (  1,   1,   1), (  1,   2,   3), (  2,   3,   4), (  2,   3,   4), (1/2, 1/3, 1/4) ], #g3 Data Modernization
    [ (  2,   3,   4), (  2,   3,   4), (1/1, 1/2, 1/3), (  1,   1,   1), (  2,   3,   4), (  2,   3,   4), (1/1, 1/1, 1/2) ], #g4 Products and Services
    [ (1/2, 1/3, 1/4), (1/2, 1/3, 1/4), (1/2, 1/3, 1/4), (1/2, 1/3, 1/4), (  1,   1,   1), (  1,   2,   3), (1/2, 1/3, 1/4) ], #G5 Market Place
    [ (1/2, 1/3, 1/4), (1/2, 1/3, 1/4), (1/2, 1/3, 1/4), (1/2, 1/3, 1/4), (1/1, 1/2, 1/3), (  1,   1,   1), (1/2, 1/3, 1/4) ], #G6 Ecosystem
    [ (  2,   3,   4), (1/2, 1/3, 1/4), (  2,   3,   4), (  1,   1,   2), (  2,   3,   4), (  2,   3,   4), (  1,   1,   1) ], #G7 Customer Relationship Management
    ],)
# Call Fuzzy AHP Function        
fuzzy_weights, defuzzified_weights, normalized_weights, rc = fuzzy_ahp_method(dataset)

for i in range(0, len(fuzzy_weights)):
  print('g'+str(i+1)+': ', np.around(fuzzy_weights[i], 3))
  # Crisp Weigths
for i in range(0, len(defuzzified_weights)):
  print('g'+str(i+1)+': ', round(defuzzified_weights[i], 3))
  
for i in range(0, len(normalized_weights)):
  print('g'+str(i+1)+': ', round(normalized_weights[i], 3))
  # Consistency Ratio
print('RC: ' + str(round(rc, 2)))
if (rc > 0.10):
  print('The solution is inconsistent, the pairwise comparisons must be reviewed')
else:
  print('The solution is consistent')
  
  
  
  
  
 
  
 




  
  