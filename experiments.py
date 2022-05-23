import os
import numpy as np
os.system("echo \'Variating Semantic Weigth\'")
for i in range(0, 101, 5):
  os.system(f"echo \'SW: {i/100.0}  GM: 0.9\'")
  os.system(f"python main.py -mode generator -l en -tmode offline -sw {i/100.0} -gm 0.9")


os.system(f"echo \'Variating Probabilities distribution Weigthing\'")
for i in range(0, 101, 5):
  os.system(f"echo \'SW: 0.0  GM: {i/100.0}\'")
  os.system(f"python main.py -mode generator -l en -tmode online -sw 0 -gm {i/100.0}")


os.system("echo \'Variating Both\'")
for i in range(50):

  a = np.random.uniform(0, 1)
  b = np.random.uniform(0, 1)
  os.system(f"echo \'SW: {b:.2f}  GM: {a:.2f}\'")
  os.system(f"python main.py -mode generator -l en  -tmode online -sw {b} -gm {a}")