import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import random

# Misc. Tools:

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





# Modeling:

def logit(v):
  v = np.array(v)
  return 1 / (np.e**(-v) + 1)

def MSE(true, pred):
  true = np.array(true)
  pred = np.array(pred)
  return ((true - pred)**2).mean()

# Example formula, uses b for betas and v for variable columns (dictionary or dataframe)
example_formula = "lambda b,v: (b[0] + v['x1']) * logit(b[1] + v['x2'])"
    
class Model:
  def __init__(self, dependent, formula, n_betas, betas=None):
    if not betas == None: 
      self.betas = betas
    else: 
      self.betas = [0] * n_betas
    self.formula = formula
    self.n_betas = n_betas
    self.dependent = dependent
    
  def predict(self, data, betas=None):
    # Data comes from a dictionary or dataframe, with columns that match those in the formula
    if betas is None:
      return self.formula(self.betas, data)
    else:
      return self.formula(betas, data)
  
  def fit(self, data, loss=MSE, warm_start=False, print_results=False, bootstrap=0):
    if bootstrap == 0:
      # loss(true, pred) is the loss function to minimize
      # define the function to optimize
      optimization_function = lambda b: loss(data[self.dependent], self.predict(data, b))
      # optimize
      if warm_start: 
        res = minimize(optimization_function, x0=self.betas)
      else: 
        res = minimize(optimization_function, x0=[0]*self.n_betas)
      if print_results: print(res)
      self.betas = res.x

    else: 
      betas = []
      for i in range(bootstrap):
        dataBootstrap = data.loc[random.choices(data.index, k=len(data))]
        # loss(true, pred) is the loss function to minimize
        # define the function to optimize
        optimization_function = lambda b: loss(dataBootstrap[self.dependent], self.predict(dataBootstrap, b))
        # optimize
        res = minimize(optimization_function, x0=[0]*self.n_betas)
        betas.append(res.x)
      # DataFrame of results: each row is a sample, each column is a parameter
      betas = pd.DataFrame(betas)
      self.betas = np.array(betas.mean())
      self.stdError = np.array(betas.std())
      self.pValue = norm.sf(abs(self.betas) / self.stdError)*2
      significance = {0.1:".", 0.05:"*", 0.01:"**",0.001:"***"}
      self.sig = pd.Series([""]*self.n_betas)
      for level in significance.keys():
        self.sig[self.pValue <= level] = significance[level]
      self.summary = pd.DataFrame([self.betas, self.stdError, self.pValue, self.sig],
        index=["Coefficient","Std Error","p-value","Significance"]).T
      if print_results: print(self.summary)
