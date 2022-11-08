#%%

import json
import pandas as pd, csv

dirs = '../gpt-2-output-dataset/data/medium-345M.train.jsonl'

jsonObj = pd.read_json(path_or_buf=dirs, lines=True)
#%%

with open('data.csv', 'wt', newline='', encoding="utf-8") as csvfile:
  
  spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
  spamwriter.writerow(['text'])

  for i in range(len(jsonObj)):
    jsonObj['text'][i] = jsonObj['text'][i].replace('\n\n', '\n')
    if jsonObj.iloc[i]['length'] > 250:
      texts = jsonObj.iloc[i]['text'].split('\n')
    else: texts = [jsonObj.iloc[i]['text']]

    for text in texts:
          spamwriter.writerow([text])

# %%
