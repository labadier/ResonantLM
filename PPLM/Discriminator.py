from utils.utils import HugginFaceLoad, compute_acc
from utils.utils import prepareDataLoader
from sklearn.model_selection import StratifiedKFold

from utils.params import params, bcolors
import numpy as np


import torch, os

class ClassificationHead(torch.nn.Module):
  
    """Classification Head for  transformer encoders"""

    def __init__(self, class_size, embed_size):
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        logits = self.mlp(hidden_state)
        return logits

    def load(self, path, device):
      self.load_state_dict(torch.load(path, map_location=device))
      print(f"{bcolors.OKCYAN}{bcolors.BOLD}Classification Head Weights Loaded{bcolors.ENDC}") 

class Discriminator(torch.nn.Module):

  """Transformer encoder followed by a Classification Head"""

  def __init__(
          self,
          language='en',
          classifier_head=None, 
          weigths_mode='online'
    ):
    super(Discriminator, self).__init__()
    self.tokenizer, self.encoder = HugginFaceLoad(language, weigths_mode)
    self.embed_size = self.encoder.transformer.config.hidden_size
    self.max_length = params.ML
    self.lang = language
    self.loss_criterion = torch.nn.CrossEntropyLoss()
    self.best_acc = None
    
    self.classifier_head = classifier_head
    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.to(device=self.device)

  def load(self, path):
    self.classifier_head.load_state_dict(torch.load(path, map_location=self.device))
    print(f"{bcolors.OKCYAN}{bcolors.BOLD}Classifier Weights Loaded{bcolors.ENDC}") 

  def save(self, path):
    torch.save(self.classifier_head.state_dict(), path)

  def train_custom(self):
    for param in self.encoder.parameters():
      param.requires_grad = False
    self.classifier_head.train()

  def average_hidden_states(self, ids):

    mask = ids.attention_mask.ne(0).unsqueeze(2).repeat(1, 1, self.embed_size).float().to(self.device).detach()
    hidden_states = self.encoder.transformer(**ids).last_hidden_state
    masked_hidden = hidden_states * mask
    avg_hidden = torch.sum(masked_hidden, dim=1) / (
                torch.sum(mask, dim=1).detach() + params.EPSILON
        )
    return avg_hidden

  def forward(self, data):

    ids = self.tokenizer(data['text'], return_tensors='pt', padding=True, max_length=self.max_length).to(device=self.device)
    X = self.average_hidden_states(ids)

    return self.classifier_head(X)
  
  def predict(self, data):
    
    trainloader, _ = prepareDataLoader(data_train=data, batch_size=128)
    
    self.eval()
    with torch.no_grad():
      out, log = None, None
      for data in trainloader:   
        labels = data['labels']

        dev_out = self(data)
        out = dev_out if out is None else torch.cat((out, dev_out), 0)
        log = labels if log is None else torch.cat((log, labels), 0)

      dev_acc = compute_acc('eval', out, log)
      print(f"{bcolors.OKCYAN}{bcolors.BOLD}Dev acc: {dev_acc}{bcolors.ENDC}")
  
  def makeOptimizer(self, lr, decay):
    return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=decay)

  def computeLoss(self, outputs, data):
    return self.loss_criterion(outputs, data['labels'].to(self.device) )



def train_model(model_name, model, trainloader, devloader, epoches, lr, decay, output, split=1):

  eloss, eacc, edev_loss, edev_acc = [], [], [], []

  optimizer = model.makeOptimizer(lr=lr, decay=decay)
  batches = len(trainloader)

  for epoch in range(epoches):

    running_loss = 0.0
    perc = 0
    acc = 0
    
    model.train_custom()
    last_printed = ''

    for j, data in enumerate(trainloader, 0):

      torch.cuda.empty_cache()               
      optimizer.zero_grad()

      outputs = model(data)
      loss = model.computeLoss(outputs, data)
   
      loss.backward()
      optimizer.step()

      # print statistics
      with torch.no_grad():
        if j == 0:
          acc = compute_acc(model_name, outputs, data)
          running_loss = loss.item()
        else: 
          acc = (acc + compute_acc(model_name, outputs, data))/2.0
          running_loss = (running_loss + loss.item())/2.0

      if (j+1)*100.0/batches - perc  >= 1 or j == batches-1:
        
        perc = (1+j)*100.0/batches
        last_printed = f'\rEpoch:{epoch+1:3d} of {epoches} step {j+1} of {batches}. {perc:.1f}% loss: {running_loss:.3f}'
        
        print(last_printed , end="")

    model.eval()
    eloss.append(running_loss)
    with torch.no_grad():
      out = None
      log = None
      
      for data in devloader:
        torch.cuda.empty_cache() 

        labels = data['labels']

        dev_out = model(data)
        out = dev_out if out is None else torch.cat((out, dev_out), 0)
        log = labels if log is None else torch.cat((log, labels), 0)


      dev_loss = model.loss_criterion(out, log.to(model.device)).item()
      dev_acc = compute_acc('eval', out, log)

      eacc.append(acc)
      edev_loss.append(dev_loss)
      edev_acc.append(dev_acc) 

    band = False
    if model.best_acc is None or model.best_acc < dev_acc:
      model.save(os.path.join(output, f'{model_name}_{split}.pt'))
      model.best_acc = dev_acc
      band = True

    ep_finish_print = f' acc: {acc:.3f} | dev_loss: {dev_loss:.3f} dev_acc: {dev_acc:.3f}'

    if band == True:
      print(bcolors.OKBLUE + bcolors.BOLD + last_printed + ep_finish_print + '\t[Weights Updated]' + bcolors.ENDC)
    else: print(last_printed + ep_finish_print)

  return {'loss': eloss, 'acc': eacc, 'dev_loss': edev_loss, 'dev_acc': edev_acc}

def train_model_CV(model_name, lang, data, splits = 5, epoches = 4, batch_size = 8, 
                    lr = 1e-5,  decay=2e-5, output='logs', model_mode='offline'):

  history = []
  skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)

  for i, (train_index, test_index) in enumerate(skf.split(np.zeros_like(data['labels']), data['labels'])):  
    
    history.append({'loss': [], 'acc':[], 'dev_loss': [], 'dev_acc': []})
    model = Discriminator(language=lang, 
                          classifier_head=ClassificationHead(params.CLASS_SIZE, params.EMBD_SIZE),
                          weigths_mode=model_mode)
    
    trainloader, devloader = prepareDataLoader(data_train={key:data[key][train_index] for key in data.keys()},
                                              data_dev = {key:data[key][test_index] for key in data.keys()}, batch_size=batch_size)
    history.append(train_model(model_name, model, trainloader, devloader, epoches, lr, decay, output, i+1))
    
    print('Training Finished Split: {}'. format(i+1))
    del trainloader
    del devloader
    del model
    break
  return history