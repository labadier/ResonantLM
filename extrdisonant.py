
#%%
import pandas as pd
import urllib.parse, requests, json

PORT = 5501
URL = 'hddevp.no-ip.org'

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
  disonantwords = []
  resonantwords = []
  
  for sentence in result['disambiguate']:
    for token in sentence['sentence']:

      relevant = 0
      if 'negativeFacets' in token['entry'].keys():
        for i in token['entry']['negativeFacets']:
          negativity['OCEAN'.find(i)] |= 1
          if i == 'O':
            disonantwords += [token['entry']['word']]
        # relevant |= 1

      if 'positiveFacets' in token['entry'].keys():
        for i in token['entry']['positiveFacets']:
          positivity['OCEAN'.find(i)] |= 1
          if i == 'O':
            resonantwords += [token['entry']['word']]
        # relevant |= 1
      # tokens += relevant 
  positivity = [positivity[i] if negativity[i] == 0 else -1 for i in range(5)] 
  return positivity, disonantwords, resonantwords


R = []
D = []
data = pd.read_csv('stencencespliter_disonant.csv')

for i in range(len(data)):
  p, d, r = getResonanceInfo(data['X'].iloc[i])
  R += [','.join(r)]
  D += [','.join(d)]

data['ResonantW'] = R
data['DisonantW'] = D
data.to_csv('sentence_spliter_resonanceList.csv')

#%%
import pandas as pd
import urllib.parse, requests, json

PORT = 5501
URL = 'hddevp.no-ip.org'

sente = "Move freely while listening to your favorite music, with the new Soundtrack Play app on iOS and iPod touch. With Soundtrack Play, you can navigate songs or music tracks on the way to play on your iPhone from your computer."
query = f"http://{URL}:{PORT}/api/v1/analyzer/?text={urllib.parse.quote(sente)}"
response = requests.request("POST", query, timeout=3.0)
response.raise_for_status()
result = json.loads(response.text)
# %%

l = []
ex = []
for i in "Move freely while listening to your favorite music, with the new Soundtrack Play app on iOS and iPod touch. With Soundtrack Play, you can navigate songs or music tracks on the way to play on your iPhone from your computer.".split(' '):
  l += [i]
  ex += [' '.join(l)]

dictf = {'text':ex}
dictf = pd.DataFrame(dictf)
dictf.to_csv('checkp')