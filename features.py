#%%
import pandas as pd, numpy as np, os, csv
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams
import spacy
from collections import Counter
from glob import glob
from tqdm import tqdm



SPACY_MODEL = spacy.load("en_core_web_sm")
# suffix n-grams

def suffix_ngrams(text, n=3):
  return [words[-n:] for words in word_tokenize(text) if len(words) >= n]

def preffix_ngrams(text, n=3):
  return [words[:n] for words in word_tokenize(text) if len(words) >= n]

def char_ngrams(text, n=3):
  return [text[i:i+n] for i in range(len(text)-n+1)]

def words_ngrams(text, n=3):
  return list(ngrams(word_tokenize(text), n))

def get_postag(text):
  proc = SPACY_MODEL(text.strip()[:1000000])
  return [word.tag_ for word in proc]


function = { 'words_ngrams': words_ngrams, 'postag': get_postag, 'suffix_ngrams': suffix_ngrams, 'preffix_ngrams': preffix_ngrams, 'char_ngrams': char_ngrams}

#%%
# def Compute_IG(maleInstances, femaleInstances):

#   '''Compute Information Gain for each tokens in the corpus'''

#   positive_instances = []
#   negative_instances = []

#   list_of_words = []
#   n_male = 0
#   n_female = 0

#   for instance in maleInstances:
#     positive_instances += instance
#     n_male += 1
#     list_of_words += instance
    
#   for instance in femaleInstances:
#     negative_instances += instance
#     n_female += 1
#     list_of_words += instance


#   words = len(list_of_words)
#   list_of_words = Counter(list_of_words)

#   TC = [ Counter(negative_instances), Counter(positive_instances) ] # term probability by class
#   CC = [ len(negative_instances), len(positive_instances) ] #class cardinality
#   Pc = [ n_female/(n_female + n_male),  n_male/(n_female + n_male)] #class probability
#   IG = [dict(), dict()] #****** Information Gain

#   for c in range(2):
#     for t in TC[c].keys():

#       # term
#       TkC = TC[c][t]/CC[c]
#       Pt = list_of_words[t]/words

#       PT = TkC * np.log2( 1e-8 + TkC/(Pc[c]*Pt + 1e-8) )

#       # not term

#       TkC = (CC[c] - TC[c][t])/CC[c]
#       Pt = 1.0 - list_of_words[t]/words
#       PNT = TkC * np.log2( 1e-8 + TkC/(Pc[c]*Pt + 1e-8) )

#       IG[c][t] = PT + PNT 


#   return sorted(list(IG[0].keys()), key = lambda index : IG[0][index], reverse=True)[:500], sorted(list(IG[1].keys()), key = lambda index : IG[1][index], reverse=True)[:500]


# df = pd.read_csv('gutemberg.csv')

# csvfile_female = csv.writer(open('keys_female.csv', 'wt', newline='', encoding="utf-8"), delimiter=',', quoting=csv.QUOTE_MINIMAL)
# csvfile_male = csv.writer(open('keys_male.csv', 'wt', newline='', encoding="utf-8"), delimiter=',', quoting=csv.QUOTE_MINIMAL)
# csvfile_female.writerow(['keywords', 'source'])
# csvfile_male.writerow(['keywords', 'source'])
  
# #compute keys_ngrams_suffix with IG

# function = { 'words_ngrams': words_ngrams, 'postag': get_postag, 'suffix_ngrams': suffix_ngrams, 'preffix_ngrams': preffix_ngrams, 'char_ngrams': char_ngrams}
# for func in function.keys():
#   maleInstances = []
#   femaleInstances = []

#   for i in range(len(df)):
#     if df['label'].iloc[i] == 1:
#       maleInstances += [function[func](df['text'].iloc[i])]
#     else:
#       femaleInstances += [function[func](df['text'].iloc[i])]
#   female, male = Compute_IG(maleInstances, femaleInstances)
#   csvfile_female.writerows([[i, func] for i in female])
#   csvfile_male.writerows([[i, func] for i in male])

# #compute keys_ngrams_suffix with IG

## filter
#%%
def check(text, filters):
  
  transforms = []
  for func in function.keys():
    transforms += function[func](text)

  return len(set(transforms).intersection(filters)) > 0

df = pd.read_csv('gutemberg.csv')

filtersMale = set(pd.read_csv('keys_male.csv')['keywords'].to_list())
filtersFemale = set(pd.read_csv('keys_female.csv')['keywords'].to_list())

with open('gutemberg_filtered.csv', 'wt', newline='', encoding="utf-8") as csvfile:
  csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
  csvwriter.writerow(['text', 'label'])

  for i in range(len(df)):
    if df['label'].iloc[i] == 1 and not check(df['text'].iloc[i], filtersMale):
      continue
    elif df['label'].iloc[i] == 0 and not check(df['text'].iloc[i], filtersFemale):
      continue

    csvwriter.writerow([df['text'].iloc[i], df['label'].iloc[i]])
# %%
