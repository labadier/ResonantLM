
class params:

  model = {'en': 'gpt2-medium'} #! TODO change to gpt2-medium

  LR, DECAY = 1e-5,  2e-5
  SPLITS = 5
  IL = 64
  ML = 110
  BS = 64
  EPOCHES = 4
  MULTITASK = 'stl'
  PRET_MODE = 'offline'
  OUTPUT = '.'
  EPSILON = 1e-8

  seed = "Why you don\'t like him"
  CLASS_SIZE = 2
  EMBD_SIZE = 1024 #! TODO change to 1024

class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

  # HEADER = ''
  # OKBLUE = ''
  # OKCYAN = ''
  # OKGREEN = ''
  # WARNING = ''
  # FAIL = ''
  # ENDC = ''
  # BOLD = ''
  # UNDERLINE = ''


class PPLM:

  num_samples=3
  discrim='sentiment'
  class_label='pos'
  length=30
  stepsize=0.03
  temperature=1
  top_k=10
  sample=True
  num_iterations=3
  grad_length=50
  horizon_length=2
  window_length=0
  decay=False
  gamma=1.5
  gm_scale=0.4 #! moved to 0.8 from 0.9
  kl_scale=0.01
  seed=113
  verbosity='regular'
  semantic_weight = 0.0
  
  DISCRIMINATOR_MODELS_PARAMS = {
    "sentiment": {
        "path": "logs/sentiment.pt",
        "class_size": 2,
        "embed_size": params.EMBD_SIZE, 
        "default_class": 1,
        "pretrained_model": params.model['en'],
    },
    "conscientiousness": { #! Ready
        "path": "logs/conscientiousness.pt",
        "class_size": 2,
        "embed_size": params.EMBD_SIZE, 
        "default_class": 1,
        "pretrained_model": params.model['en'],
    },
    "agreeableness": { #! Ready
        "path": "logs/agreeableness.pt",
        "class_size": 2,
        "embed_size": params.EMBD_SIZE, 
        "default_class": 1,
        "pretrained_model": params.model['en'],
    },    
    "openness": {
        "path": "logs/openness.pt",
        "class_size": 2,
        "embed_size": params.EMBD_SIZE, 
        "default_class": 1,
        "pretrained_model": params.model['en'],
    },
    "extraversion": {
        "path": "logs/extraversion.pt",
        "class_size": 2,
        "embed_size": params.EMBD_SIZE, 
        "default_class": 1,
        "pretrained_model": params.model['en'],
    },
    "neuroticism": {
        "path": "logs/extraversion.pt",
        "class_size": 2,
        "embed_size": params.EMBD_SIZE, 
        "default_class": 1,
        "pretrained_model": params.model['en'],
    },
}


# SW: 0.84  GM: 0.25
#python main.py -mode generator -l en -tmode offline -sw 0.8 -gm 0.8 -seed "I hate them"
#python main.py -mode generator -l en -tmode offline -sw 0.8 -gm 0.8 -seed "I'm not going to work cause I hate that"
#python main.py -mode generator -l en -tmode offline -sw 0.8 -gm 0.8 -seed "I hate to work"
#python main.py -mode generator -l en -tmode offline -sw 0.8 -gm 0.8 -seed "I don't even like your"

