import numpy as np

def gradient(color1, color2, n):
  # convert colors to RGB:
  if type(color1) == str:
    color1 = color1.replace("#","")
    color1 = [int(color1[i:i+2], 16) for i in range(3)]
  if type(color2) == str:
    color2 = color2.replace("#","")
    color2 = [int(color2[i:i+2], 16) for i in range(3)]
  R = np.linspace(color1[0], color2[0], n)
  G = np.linspace(color1[1], color2[1], n)
  B = np.linspace(color1[2], color2[2], n)
  return [(round(R[i]), round(G[i]), round(B[i])) for i in range(n)]
    
