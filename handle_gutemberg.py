import pandas as pd, numpy as np, os, csv
from nltk.tokenize import sent_tokenize
from glob import glob
from tqdm import tqdm
from features import check


male = pd.read_csv('metadata/metadata_enriched.csv', sep = '\t')
male = male[male['language'].str.contains('en')]

print(len(male))
female = male[male['gender'] == 'F']['id'].to_list()
male = np.array(male[male['gender'] == 'M']['id'].to_list())
male = male[np.random.permutation(len(male))[:len(female)]]


dirs = glob('data/raw/*.txt')
os.system('mkdir female')
os.system('mkdir male')

fline = []
mline = [] 

df = {'text':[], 'labels':[]}

def save_examples(file, user, gender):

  num_lines = sum(1 for line in open(file, errors='ignore'))
  if gender == 'male':
    mline.append(num_lines)
  else:
    fline.append(num_lines)

  texts = []
  with open(f"{gender}/{user}.csv", 'wt', newline='', encoding="utf-8", errors='ignore') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['text'])
    
    for i, line in enumerate(open(file, errors='ignore')):

      if i < num_lines*0.3:
        continue
      if i > num_lines*0.8:
        break
      
      texts.append(line)

    texts = sent_tokenize(' '.join(texts.strip().split('\n')))
    texts = [(text, check(text)) for text in texts ]

    texts.sort(key=lambda x: x[1], reverse=True)
    texts = [text[0] for i in range(len(texts)) if i < min(len(texts)*0.1, 200)]

    for text in texts:
      spamwriter.writerow([text])
  
  df['text'] += list(np.array(texts)[np.random.permutation(len(texts))[:100]])
  df['labels'] += [int(gender == 'male')]*min(len(texts), 100)


     
iter = tqdm(dirs)

for i in iter:

  if i.split('/')[-1].split('.')[0].split('_')[0] in male:
    save_examples(i, i.split('/')[-1].split('.')[0].split('_')[0], 'male')

  if i.split('/')[-1].split('.')[0].split('_')[0] in female:
    save_examples(i, i.split('/')[-1].split('.')[0].split('_')[0], 'female')


df = pd.DataFrame(df)
df.to_csv('data/gutemberg2.0.csv', index=False)

print(f'male\n mean: {np.mean(mline):.3f} max: {np.max(mline):.3f} min: {np.min(mline):.3f} std: {np.std(mline):.3f}')
print(f'female\n mean: {np.mean(fline):.3f} max: {np.max(fline):.3f} min: {np.min(fline):.3f} std: {np.std(fline):.3f}')