from eval_data import data
from data_abc import abc
import pandas as pd

# read dvmp2.csv
#dataset = pd.read_csv('dvmp2.csv')['prompt'].tolist()

PROMPTS = ['A boy in a red shirt with a helmet and yellow bat']

SEEDS = [0]
# 'a checkered balloon and a sliced strawberry and a purple banana','a modern crown and a skewered apple and a gray dog', 'a teal tomato and a furry mouse and a brown wooden camera',\
#             'a purple baby horse and a black banana and a sliced tomato', 'a purple modern camera and a spotted baby dog and a sliced tomato',\
#                  'a red metal crown and a white bear and a wooden chair', 'a green banana and a spotted chair and a red baby furry bear', 'a baby monkey and a sliced tomato and a blue baby gorilla' 
# 'an orange suitcase and a sliced strawberry and a baby mouse', \
#            'a spotted turtle and a purple apple and a sliced blue tomato',  'a skewered strawberry and a checkered clock and a spotted sleepy penguin', \
#                   'a sliced apple and a purple camera and a teal lion', 