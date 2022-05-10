
class params:

  model = {'en:' 'gpt2-medium'}

  LR, DECAY = 1e-5,  2e-5
  SPLITS = 5
  IL = 64
  ML = 110
  BS = 64
  EPOCHES = 4
  MULTITASK = 'stl'
  PRET_MODE = 'offline'
  OUTPUT = '.'

  CLASS_SIZE = 2
  EMBD_SIZE = 1024

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
