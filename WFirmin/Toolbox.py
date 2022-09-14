import numpy as np
import pandas as pd

def cut(v, n, unique=True, label="median", r=2):
  v = np.array(v)
  qs = np.quantile(v, np.linspace(0,1,n+1))
  if unique: qs = np.unique(qs)
  q = v.copy()
  
  if label == "median": method = lambda arr: np.median(arr)
  elif label == "mean": method = lambda arr: arr.mean()
  elif label == "left": method = lambda interval: round(interval[0],r)
  elif label == "right": method = lambda interval: round(interval[1],r)
  elif label == "interval": method = lambda interval: str((round(interval[0],r), round(interval[1],r)))
  
  if label == "interval":
    q = q.astype(str)
  
  if label in ["median","mean"]:
    for i in range(len(qs)-1):
      left = qs[i]
      right = qs[i+1]
      if i == len(qs)-2: mask = (left <= v) & (v <= right)
      else: mask = (left <= v) & (v < right)
      q[mask] = method(q[mask])
  
  elif label in ["left", "right", "interval"]:
    for i in range(len(qs)-1):
      left = qs[i]
      right = qs[i+1]
      if i == len(qs)-2: mask = (left <= v) & (v <= right)
      else: mask = (left <= v) & (v < right)
      q[mask] = method((left,right))
  
  return q
