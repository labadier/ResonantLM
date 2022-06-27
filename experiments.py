import os
import numpy as np

phrases = ["Women's Harley Davidson Jacket",
"White pure linen curtain",
"Restaurant LE CONFIDENTIAL MARRAKECH international meals",
"Papaya body cream",
"5 nights all inclusive hotel accommodation at Disneyland Paris"]

for i in phrases:
    os.system(f'python main.py -seed "{i}" -dt agreeableness')