from glob import glob
import csv, pandas as pd, re, string
from matplotlib.pyplot import annotate
import urllib.parse, requests, json
from utils.params import bcolors, PPLM as lmparams
import argparse, sys


PORT = 5501
URL = 'hddevp.no-ip.org'
STEP = 200

def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

def getBacthResonanceInfo(text):

  query = f"http://{URL}:{PORT}/api/v1/analyzer/?text={urllib.parse.quote(text)}"

  try:
    response = requests.request("POST", query)
    response.raise_for_status()
    result = json.loads(response.text)
      
  except requests.exceptions.RequestException as e: 
    return None

  if 'disambiguate' not in result.keys():
    return None

  annotation = []
  positivity, negativity = [0]*5, [0]*5

  for sentence in result['disambiguate']:
    for token in sentence['sentence']:


      if 'THxISISROBXERTOXTOXKEN' in token['entry']['word']:
        annotation += [positivity.copy()]
        positivity, negativity = [0]*5, [0]*5
        continue

      if 'negativeFacets' in token['entry'].keys():
        for i in token['entry']['negativeFacets']:
          negativity['OCEAN'.find(i)] |= 1

      if 'positiveFacets' in token['entry'].keys():
        for i in token['entry']['positiveFacets']:
          positivity['OCEAN'.find(i)] |= 1

    positivity = [positivity[i] if negativity[i] == 0 else -1 for i in range(5)] 
  return annotation

#! TODO parametrize in getBatchResonanceInfo
def getResonanceInfo(text):

  query = f"http://{URL}:{PORT}/api/v1/analyzer/?text={urllib.parse.quote(text)}"

  try:
    response = requests.request("POST", query, timeout=3.0)
    response.raise_for_status()
    result = json.loads(response.text)
      
  except requests.exceptions.RequestException as e: 
    return None

  if 'disambiguate' not in result.keys():
    return None

  positivity, negativity = [0]*5, [0]*5
  # tokens = 0
  for sentence in result['disambiguate']:
    for token in sentence['sentence']:

      relevant = 0
      if 'negativeFacets' in token['entry'].keys():
        for i in token['entry']['negativeFacets']:
          negativity['OCEAN'.find(i)] |= 1
        # relevant |= 1

      if 'positiveFacets' in token['entry'].keys():
        for i in token['entry']['positiveFacets']:
          positivity['OCEAN'.find(i)] |= 1
        # relevant |= 1
      # tokens += relevant 
  positivity = [positivity[i] if negativity[i] == 0 else -1 for i in range(5)] 
  return positivity


def load_keywords_tree(facetas):

  keywords = '' 
  for f in facetas:
    with open(f'data/meta_{f}.txt') as file:
      keywords = '|'.join([i[:-1] for i in file]) if keywords == '' else keywords + '|'.join([i[:-1] for i in file])

  return re.compile(keywords, re.IGNORECASE)

def check_params(args=None):
  parser = argparse.ArgumentParser(description='Language Model Encoder')

  parser.add_argument('-m', metavar='mode', help='Either simple or filter')
  parser.add_argument('-f', metavar='faceta', default=None, help='personalities to filter')
  parser.add_argument('-s', metavar='sourcefile', help='source file')
  parser.add_argument('-o', metavar='outputfile', help='output file')
  parser.add_argument('-e', metavar='errorfile', help='erorr loger file')
  return parser.parse_args(args)

if __name__ == '__main__':


  parameters = check_params(sys.argv[1:])


  data_path = parameters.s
  mode = parameters.m
  filters = parameters.f
  output = parameters.o
  errorlogs = parameters.e
  
  if filters is not None:
    TREE = load_keywords_tree([i for i in lmparams.DISCRIMINATOR_MODELS_PARAMS.keys() if i.lower()[0] in filters.lower()]) 

  addrs = [data_path]

  #compute amount of examples
  total = 0.0
  for file in addrs:
    total += len(pd.read_csv(file, usecols=['text']))

  perc = 0.0
  processed = 0.0

  wasted = 0
  total = 0

  with open(output, 'wt', newline='', encoding="utf-8") as csvfile:
    
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['text', 'O', 'C', 'E', 'A', 'N'])

    for file in addrs:
      dataframe = pd.read_csv(file, usecols=['text']).fillna('')
      wasted += len(dataframe)
      total += len(dataframe)

      dataframe = [ strip_all_entities(strip_links(text.replace('\\n', ' '))) for text in dataframe['text'].to_list()  if mode == 'simple' or (mode == 'filter' and len(TREE.search(text)))]
      wasted -= len(dataframe)

      for i in range(0, len(dataframe), STEP):
        
        if processed/total - perc >= 0.0001:
          perc = processed/total
          print(f"\r{bcolors.OKGREEN}{bcolors.BOLD}Analyzing Data{bcolors.ENDC}: {perc*100.0:.2f}%", end="") 

        processed += STEP
        text = ''
        for j in dataframe[i: i + STEP]:
          text += j.strip()
          if not len(text):
            text += ' Errorxt. '
          text += ' puntual. THxISISROBXERTOXTOXKEN. ' if text[-1] not in '.;?*' else ' THxISISROBXERTOXTOXKEN. '

        resonance = getBacthResonanceInfo(text)
        if resonance is None:
          with open(errorlogs, 'a') as logging: 
            for j in dataframe[i: i + STEP]:
               logging.write(j + '\n')
          continue

        for text, annotations in zip(dataframe[i: i + STEP], resonance):
          spamwriter.writerow([text] + annotations)

    print(f"\r{bcolors.OKGREEN}{bcolors.BOLD}Analyzing Data ok. Wasted {wasted} of {total}{bcolors.ENDC}") 

#%%
import pandas as pd
import csv, re


def load_keywords_tree(facetas):

  keywords = '' 
  for f in facetas:
    with open(f'data/meta_{f}.txt') as file:
      keywords = '|'.join([i[:-1] for i in file]) if keywords == '' else keywords + '|'.join([i[:-1] for i in file])

  return re.compile(keywords, re.IGNORECASE)

TREE = load_keywords_tree(['openness'])
file_w = open('data_filtered.csv', 'w')
spamwriter = csv.writer(file_w, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
spamwriter.writerow(['text'])

with open('data.csv', 'r') as file_r:
  for text in file_r:
    if TREE.search(text) is None:
      continue
    spamwriter.writerow([text])
    
# %%
