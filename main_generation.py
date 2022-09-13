import argparse, sys, os, numpy as np, torch, random, csv

from PPLM.Discriminator import train_model, train_model_CV, Discriminator
from PPLM.Discriminator import ClassificationHead
from PPLM.PPLM import run_pplm, get_classifier
from utils.params import bcolors, params, PPLM as lmparams
from utils.utils import plot_training, load_data, load_data_postgen
import torch.nn.functional as F


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def check_params(args=None):
  parser = argparse.ArgumentParser(description='Language Model Encoder')

  parser.add_argument('-l', metavar='language', default='en', help='Task Language')
  parser.add_argument('-dt', metavar='disrciminator_task', default=lmparams.discrim, help='Task Discriminator')
  parser.add_argument('-mode', metavar='mode', default='generator', help='task')
  parser.add_argument('-phase', metavar='phase', help='Phase')
  parser.add_argument('-lr', metavar='lrate', default = params.LR, type=float, help='learning rate')
  parser.add_argument('-tmode', metavar='tmode', default = 'online', help='Encoder Weights Mode')
  parser.add_argument('-decay', metavar='decay', default = params.DECAY, type=float, help='learning rate decay')
  parser.add_argument('-epoches', metavar='epoches', type=int, help='Trainning Epoches')
  parser.add_argument('-bs', metavar='batch_size', default=params.BS, type=int, help='Batch Size')
  parser.add_argument('-tp', metavar='train_path', help='Data path Training set')
  parser.add_argument('-seed', metavar='seed',  default =params.seed, help='Seed for generting text')
  parser.add_argument('-sw', metavar='sw', default =lmparams.semantic_weight, type=float, help='Semantic Wight for perturbing generation')
  parser.add_argument('-gm', metavar='gm', default =lmparams.gm_scale, type=float, help='Scaling for generation distribution')
  parser.add_argument('-bias', metavar='bias', default =lmparams.class_label, type=str, help='Class for Biasing Generation')
  parser.add_argument('-nsamples', metavar='nsamples', default =lmparams.num_samples, type=int, help='Number of samples for generation')
  parser.add_argument('-wpath', metavar='wpath', default = 'logs', help='Weight Path')
   
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
  seed = parameters.seed
  semantic_weight = parameters.sw
  gm = parameters.gm
  bias = parameters.bias
  nsamples = parameters.nsamples
  wpath = parameters.wpath

  if mode == 'discriminator':

    d_task = [d_task for d_task in lmparams.DISCRIMINATOR_MODELS_PARAMS.keys() if not d_task.find(parameters.dt.lower())][0]

    if phase == 'train':
      if os.path.exists('./logs') == False:
        os.system('mkdir logs')

      
      text, labels = load_data(path = train_path, task = d_task)
      dataTrain = {'text':text, 'labels': labels}
  
      history = train_model_CV(model_name=d_task, lang=language, data=dataTrain, output=wpath,
                            epoches=epoches, batch_size=batch_size, lr=learning_rate, decay=decay, 
                            model_mode=mode_weigth)
    
      plot_training(history[-1], language, 'acc')
      exit(0)

    if phase == 'evaluate':

      '''
        Get Encodings for each author's message from the Transformer-based encoders
      '''
      # text, labels = load_data(path = train_path, eval = True, task = d_task)
      # data = {'text':text, 'labels': labels}
      df, data = load_data_postgen(train_path)

      discrim = parameters.dt.lower()
      model = Discriminator(language=language, 
                            classifier_head=ClassificationHead(params.CLASS_SIZE, params.EMBD_SIZE),
                            weigths_mode=mode_weigth)

      if os.path.isfile(lmparams.DISCRIMINATOR_MODELS_PARAMS[discrim]['path']):
          model.load(lmparams.DISCRIMINATOR_MODELS_PARAMS[discrim]['path'])
      else: 
        print(f"{bcolors.FAIL}{bcolors.BOLD}No Weights Loaded{bcolors.ENDC}")
        exit(1)

      outs = model.predict(data)
      with open('postgen.csv', 'w') as csfile:
        spamwriter = csv.writer(csfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(df)):
          spamwriter.writerow(df.iloc[0].to_list() + [F.softmax(outs[i])[1]])

  if mode == 'generator':

    printunperturbed = True

    for i in lmparams.DISCRIMINATOR_MODELS_PARAMS.keys():
      if i.lower()[0] not in parameters.dt.lower():
        continue

      z = run_pplm(pretrained_model=params.model[language], model_mode=mode_weigth,
              cond_text=seed, num_samples=nsamples, discrim=i.lower(),
              class_label= bias,
              length=lmparams.length, stepsize = lmparams.stepsize, temperature=lmparams.temperature,
              top_k = lmparams.top_k, sample=lmparams.sample, num_iterations=lmparams.num_iterations,
              grad_length=lmparams.grad_length, horizon_length=lmparams.horizon_length,
              window_length=lmparams.window_length, decay=lmparams.decay, gamma=lmparams.gamma,
              gm_scale=gm, kl_scale=lmparams.kl_scale, seed=lmparams.seed,
              verbosity=lmparams.verbosity, semantic_weight=semantic_weight, print_unperturbed=printunperturbed)
      
      with open('experimental_setup.csv', 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in z:
          spamwriter.writerow(i)

      printunperturbed &= False
