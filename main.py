#%%
import argparse, sys, os, numpy as np, torch, random

from PPLM.Discriminator import train_model, train_model_CV, Discriminator
from PPLM.Discriminator import ClassificationHead
from PPLM.PPLM import run_pplm
from utils.params import bcolors, params
from utils.utils import plot_training, load_data



torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def check_params(args=None):
  parser = argparse.ArgumentParser(description='Language Model Encoder')

  parser.add_argument('-l', metavar='language', default='en', help='Task Language')
  parser.add_argument('-mode', metavar='mode', help='task')
  parser.add_argument('-model', metavar='model', help='model to encode')
  parser.add_argument('-phase', metavar='phase', help='Phase')
  parser.add_argument('-lr', metavar='lrate', default = params.LR, type=float, help='learning rate')
  parser.add_argument('-tmode', metavar='tmode', default = 'online', help='Encoder Weights Mode')
  parser.add_argument('-decay', metavar='decay', default = params.DECAY, type=float, help='learning rate decay')
  parser.add_argument('-epoches', metavar='epoches', type=int, help='Trainning Epoches')
  parser.add_argument('-bs', metavar='batch_size', default=params.BS, type=int, help='Batch Size')
  parser.add_argument('-tp', metavar='train_path', help='Data path Training set')
  parser.add_argument('-seed', metavar='seed', help='Seed for generting text')
  
  return parser.parse_args(args)

if __name__ == '__main__':


  parameters = check_params(sys.argv[1:])

  language = parameters.l
  phase = parameters.phase
  learning_rate = parameters.lr
  mode_weigth = parameters.tmode
  decay = parameters.decay
  epoches = parameters.epoches
  batch_size = parameters.bs
  train_path = parameters.tp
  mode = parameters.mode
  model_name = parameters.model
  
  if mode == 'discriminator':

    if phase == 'train':
      if os.path.exists('./logs') == False:
        os.system('mkdir logs')

      
      text, labels = load_data(train_path)
      dataTrain = {'text':text, 'labels': labels}
  
      history = train_model_CV(model_name=params.model[language].split('/')[-1], lang=language, data=dataTrain,
                            epoches=epoches, batch_size=batch_size, lr=learning_rate, decay=decay, model_mode='online')
    
      plot_training(history[-1], language, 'acc')
      exit(0)

    if phase == 'evaluate':

      '''
        Get Encodings for each author's message from the Transformer-based encoders
      '''
      text, labels = load_data(train_path, eval = True)
      data = {'text':text, 'labels': labels}

      model = Discriminator(language=language, 
                            classifier_head=ClassificationHead(params.CLASS_SIZE, params.EMBD_SIZE))

      if os.path.isfile(f"logs/{params.model[language].split('/')[-1]}_1.pt"):
          model.load(f"logs/{params.model[language].split('/')[-1]}_1.pt")
      else: 
        print(f"{bcolors.FAIL}{bcolors.BOLD}No Weights Loaded{bcolors.ENDC}")
        exit(1)

      model.predict(data)

  if mode == 'generate':
    run_pplm()
