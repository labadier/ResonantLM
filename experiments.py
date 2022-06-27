import os
import numpy as np

phrases = ["Women's Harley Davidson Jacket",
"White pure linen curtain",
"Restaurant LE CONFIDENTIAL MARRAKECH international meals",
"Papaya body cream",
"5 nights all inclusive hotel accommodation at Disneyland Paris"]

for i in phrases:
  for j in ['pos']:
    os.system(f'echo {j}')
    os.system(f'python main.py -mode generator -l en -tmode online -sw 0 -gm 0.4 -seed "{i}" -nsamples 3 -bias {j} -dt agreeableness')