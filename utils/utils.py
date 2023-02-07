from transformers import BloomForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

from utils.params import params, bcolors

import os, torch, random
import numpy as np, pandas as pd


def HugginFaceLoad(language, weigths_source):

  prefix = 'data' if weigths_source == 'offline' else ''

  model = BloomForCausalLM.from_pretrained(os.path.join(prefix , params.model[language]))
  tokenizer = AutoTokenizer.from_pretrained(os.path.join(prefix , params.model[language]))
  tokenizer.pad_token = tokenizer.bos_token
  
  return tokenizer, model

class Data(Dataset):

  def __init__(self, data):

    self.data = data
    
  def __len__(self):
    return len(self.data[list(self.data.keys())[0]])

  def __getitem__(self, idx):

    if torch.is_tensor(idx):
      idx = idx.tolist()

    ret = {key: self.data[key][idx] for key in self.data.keys()}
    return ret

def compute_acc(mode, logits, data):

  with torch.no_grad():
    out = logits.argmax(dim=1).cpu()

    if mode == 'eval':
      return((1.0*(out == data)).sum()/len(data)).item()
    return((1.0*(out == data['labels'])).sum()/len(data['labels'])).item()
    
def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

def prepareDataLoader(data_train, data_dev = None, batch_size = None, eval=False) -> DataLoader:
  
  devloader = None
  trainloader = DataLoader(Data(data_train), batch_size=batch_size, shuffle=(not eval), num_workers=4, worker_init_fn=seed_worker)
  if data_dev is not None:
    devloader = DataLoader(Data(data_dev), batch_size=batch_size, shuffle=(not eval), num_workers=4, worker_init_fn=seed_worker)
  return trainloader, devloader


def plot_training(history, language, measure='loss'):

  plt.plot(history[measure])
  plt.plot(history['dev_' + measure])
  plt.legend(['train', 'dev'], loc='upper left')
  plt.ylabel(measure)
  plt.xlabel('Epoch')
  if measure == 'loss':
      x = np.argmin(history['dev_loss'])
  else: x = np.argmax(history['dev_acc'])

  plt.plot(x,history['dev_' + measure][x], marker="o", color="red")

  if os.path.exists('./logs') == False:
      os.system('mkdir logs')

  plt.savefig('./logs/train_history_{}.png'.format(language))
  

def load_data( path = None, eval=False, task='sentiment'):
  
  if task == 'sentiment':
    import hub
    
    ds = hub.load(f"hub://activeloop/sentiment-140-{'test' if eval else 'train'}")
    data_frame = pd.DataFrame({'tweet':ds.tweet_text.data(), 'label':ds.sentiment_type.data().reshape(-1)})
    o_len = len(data_frame)
    data_frame = data_frame[data_frame['label'] != 2]
    print(f"{bcolors.WARNING}{bcolors.BOLD}{o_len - len(data_frame)} Examples Wasted !!{bcolors.ENDC}")
    
    o_len = len(data_frame)
    data_frame['label'] = data_frame['label'].apply(lambda row: int(row > 2))

    emolex = pd.read_csv('data/emolex_en.csv', usecols=['English (en)', 'Positive', 'Negative'])
    x_pos = emolex[emolex['Positive'] == 1][['English (en)', 'Positive']].rename({'English (en)':'tweet', 'Positive':'label'}, axis='columns')
    x_neg = emolex[emolex['Negative'] == 1][['English (en)','Negative']].rename({'English (en)':'tweet', 'Negative':'label'}, axis='columns')
    x_neg['label'] = x_neg['label'].apply(lambda row: 0)

    data_frame = pd.concat([x_pos, x_neg, data_frame])

    print(f"{bcolors.OKCYAN}{bcolors.BOLD}{len(data_frame) - o_len} Examples Added !!{bcolors.ENDC}")
    return data_frame['tweet'].to_numpy(),  data_frame['label'].astype(int).to_numpy()
  
  data_frame = pd.read_csv(path)
  
  if task == 'gender':
    return data_frame['text'].to_numpy(),  data_frame['labels'].to_numpy()

  labels = data_frame[task.upper()[0]].astype(int).apply(lambda row: int(row == 1)).to_numpy() # 40-60
  print(f"{bcolors.OKCYAN}{bcolors.BOLD}{len(labels)} Examples Loaded with {np.sum(labels)/len(labels):.2f} Positive Balance for {task.upper()}!!{bcolors.ENDC}")
  return data_frame['text'].to_numpy(),  labels


def load_data_postgen(path):
  df = pd.read_csv(path, header=None)
  return df, {'text':df[8].to_numpy(), 'labels':np.zeros(len(df),)}