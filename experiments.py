import os
import numpy as np
print('Variating Semantic Weigth')
for i in range(0, 101, 5):
  print(f'SW: {i/100.0}  GM: 0.9')
  print(f"python main.py -mode generator -l en -tmode online -sw {1/100.0} -gm 0.9")
  os.system(f"python main.py -mode generator -l en -tmode online -sw {1/100.0} -gm 0.9")


print('Variating Probabilities distribution Weigthing')
for i in range(0, 101, 5):
  print(f'SW: 0.0  GM: {i/100.0}')
  os.system(f"python main.py -mode generator -l en -tmode online -sw 0 -gm {1/100.0}")


print('Variating Both')
for i in range(50):

  a = np.random.uniform(0, 1)
  b = np.random.uniform(0, 1)
  print(f'SW: {b:.2f}  GM: {a:.2f}')
  os.system(f"python main.py -mode generator -l en  -tmode online -sw {b} -gm {a}")