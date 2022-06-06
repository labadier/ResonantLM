#%%

import pandas as pd
import csv

with open('data.faceta.csv', 'w', newline='', encoding="utf-8") as csvfile:
  
  spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  spamwriter.writerow(['text', 'O', 'C', 'E', 'A', 'N'])

  addrs = ['resonance.csv', 'resonance_1.csv']

  for file in addrs:
    dataframe = pd.read_csv(file)
    for i in range(len(dataframe)):  
      if i != 0 and dataframe.iloc[i]['text'] == dataframe.iloc[i-1]['text']:
        continue
      
      spamwriter.writerow([dataframe.iloc[i]['text'], dataframe.iloc[i]['O'], dataframe.iloc[i]['C'], dataframe.iloc[i]['E'], dataframe.iloc[i]['A'], dataframe.iloc[i]['N']])

# %%
