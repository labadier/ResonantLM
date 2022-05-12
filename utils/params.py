
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

  CLASS_SIZE = 2
  EMBD_SIZE = 768 #! TODO change to 1024

  twitter_api = {
    'APP_CONSUMER_KEY' : 'qhHfmJ6x7aUWCvtZUq4P43TDV',
    'APP_CONSUMER_SECRET' : 'IKgDkNZcOffx3ocoq1jmdqEFTuxFEF2LBDEmquW2nhi70pEM2t',
    'ACCESS_TOKEN' : '3673976903-vcvLLCxdFGGJpUdIcFYAmrtQ38rwww4WzIHhJFA',
    'ACCESS_TOKEN_SECRET' : 'vvpy6dZAvmBMSV2LIDIyr67msItKXChXa091lHH1JApre',
  }

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

class PPLM:

  num_samples=1,
  discrim='sentiment',
  class_label='pos',
  length=100,
  stepsize=0.02,
  temperature=1.0,
  top_k=10,
  sample=True,
  num_iterations=3,
  grad_length=10000,
  horizon_length=1,
  window_length=0,
  decay=False,
  gamma=1.5,
  gm_scale=0.8, #! moved to 0.8
  kl_scale=0.01,
  seed=21,
  verbosity='regular'

