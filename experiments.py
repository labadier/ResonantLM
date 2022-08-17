import os

phrases = ["Women's Harley Davidson Jacket",
"White pure linen curtain",
"Restaurant LE CONFIDENTIAL MARRAKECH international meals",
"Papaya body cream",
"5 nights all inclusive hotel accommodation at Disneyland Paris", 
"Natural cream", 
"cleaning products"]


gm_scales = [0.2, 0.4, 0.6, 0.8] 

for p in ['O', 'C', 'E', 'A', 'N']:
  print("========== Factea O ==============\n")
  for gm in gm_scales:
    print("=========== gm: {gm} =============\n")
    for i in phrases:
        os.system(f'python main_generation.py -seed "{i}" -dt {p} -gm {gm}')