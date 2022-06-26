from glob import glob
import csv, pandas as pd, re, string
import urllib.parse, requests, json
from utils.params import bcolors


PORT = 5501
URL = 'hddevp.no-ip.org' #'hddevp.no-ip.org'
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

def getResonanceInfo(text):

  query = f"http://{URL}:{PORT}/api/v1/analyzer/?text={urllib.parse.quote(text)}"

  try:
    response = requests.request("POST", query)
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


def load_keywords_tree(faceta):

  with open(f'data/meta_{faceta}.txt') as file:
    keywords = '|'.join([i[:-1] for i in file])

  return re.compile(keywords, re.IGNORECASE)


if __name__ == '__main__':

  data_path = 'data'
  # addrs = sorted(glob(data_path + '/*.csv'))
  TREE = load_keywords_tree('agreeableness')

  addrs = [data_path + '/sentiment140-train.csv']

  #compute amount of examples
  total = 0.0
  for file in addrs:
    total += len(pd.read_csv(file, usecols=['text']))

  perc = 0.0
  processed = 0.0

  wasted = 0
  with open('data/resonance.torncal.csv', 'wt', newline='', encoding="utf-8") as csvfile:
    
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['text', 'O', 'C', 'E', 'A', 'N'])

    for file in addrs:
      dataframe = pd.read_csv(file, usecols=['text']).fillna('')

      for text in dataframe['text']:
        
        if processed/total - perc >= 0.0001:
          perc = processed/total
          print(f"\r{bcolors.OKGREEN}{bcolors.BOLD}Analyzing Data{bcolors.ENDC}: {perc*100.0:.2f}%", end="") 

        processed += 1
        if not len(TREE.findall(text)):
          wasted += 1
          continue

        cleaned = strip_all_entities(strip_links(text.replace('\\n', ' ')))
        resonance = getResonanceInfo(cleaned)
        
        if resonance is None:
          with open('data/error_1.log', 'a') as logging: logging.write(cleaned + '\n')
          continue
        spamwriter.writerow([cleaned] + resonance)

        


    print(f"\r{bcolors.OKGREEN}{bcolors.BOLD}Analyzing Data ok. Wasted {wasted} of {len(dataframe['text'])}{bcolors.ENDC}") 

