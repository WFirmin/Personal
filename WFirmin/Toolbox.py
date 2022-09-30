import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import random
import seaborn as sns

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


# Concatenate lists:
def concatList(L):
  l = []
  for lst in L:
    l += lst
  return l

# Return CV groups from index:
def Resample(index, K, method="CV"):
  if method=="CV":
    sample = random.sample(index, len(index))
    split = len(index) // K
    splits = [split*i for i in range(K)] + [len(index)]
    return [sample[splits[i]:splits[i+1]] for i in range(K)]
  
  elif method=="bootstrap":
    return [random.choices(index, k=len(index)) for i in range(K)]



# Modeling:

def logit(v):
  v = np.array(v)
  return 1 / (np.e**(-v) + 1)

def MSE(true, pred, *args):
  true = np.array(true)
  pred = np.array(pred)
  return ((true - pred)**2).mean()

def elasticNet(loss, alpha, lam, betas):
  # Loss is original loss function returns, alpha is portion LASSO, lambda is multiplier (hypertune)
  return loss + lam * ((1-alpha)/2 * (betas**2).sum() + alpha * np.absolute(betas).sum())
  

# Example formula, uses b for betas and v for variable columns (dictionary or dataframe)
example_formula = "lambda b,v: (b[0] + v['x1']) * logit(b[1] + v['x2'])"
    
    
# Next add standardization, CV
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
  
  def fit(self, data, loss=MSE, 
    warm_start=False, print_results=False, resample=None, K=3,
    enet=False, lam=np.e**np.linspace(-20,1,20), alpha=1, standardize=False):
    
    X0 = lambda: self.betas if warm_start else [0]*self.n_betas
    
    if standardize:
      self.dataMeans = data.mean()
      self.dataStds = data.std()
      data = (data - self.dataMeans) / self.dataStds
      
    if resample in ["CV", "bootstrap"]:
      indices = Resample(list(data.index), K=K, method=resample)
    else:
      indices = [list(data.index)]
      
    if enet:
      if type(lam) in [int, float]: lam = np.array([lam])
      if type(alpha) in [int, float]: alpha = np.array([alpha])
      enet_results = pd.DataFrame()
      if resample == "CV":
        for a in alpha:
          for l in lam:
            for i in range(len(indices)):
              trainSample = concatList(indices[:i] + indices[i+1:])
              optimization_function = lambda b: elasticNet(loss(data.loc[trainSample, self.dependent], self.predict(data.loc[trainSample], b)), a, l, b)
              res = minimize(optimization_function, x0=X0())
              score = loss(data.loc[indices[i], self.dependent], self.predict(data.loc[indices[i]], res.x))
              enet_results = pd.concat([enet_results, 
                pd.DataFrame({"Alpha":a,"Lambda":l,"Group":i, "Score":score}, index=[0])], ignore_index=True)
      else:
        for a in alpha:
          for l in lam:
            for i in range(len(indices)):
              optimization_function = lambda b: elasticNet(loss(data.loc[indices[i],self.dependent], self.predict(data.loc[indices[i]], b)), a, l, b)
              res = minimize(optimization_function, x0=X0())
              score = loss(data.loc[indices[i], self.dependent], self.predict(data.loc[indices[i]], res.x))
              enet_results = pd.concat([enet_results,
                pd.DataFrame({"Alpha":a,"Lambda":l,"Group":i,"Score":score}, index=[0])], ignore_index=True)
      
      results1 = enet_results.loc[:,["Alpha","Lambda","Score"]].groupby(["Alpha","Lambda"])
      results = results1.mean().merge(results1.std(), on=["Alpha","Lambda"], suffixes=("_Mean","_Std"))
      self.enet_results = results
      self.regularization = results.iloc[np.argmin(results.Score_Mean)].name

      optimization_function = lambda b: elasticNet(loss(data[self.dependent], self.predict(data,b)), self.regularization[0], self.regularization[1], b)
      self.betas = minimize(optimization_function, x0=X0()).x

      if print_results: 
        print("Best: Alpha =",self.regularization[0],"; Lambda =",self.regularization[1])
        print("Best Coefficients:\n",self.betas)
        print(results)
        g = sns.relplot(x=np.log(enet_results.Lambda), y=np.log(enet_results.Score), hue="Alpha", data=enet_results, kind="line")\
          .set(xlabel="Log Lambda", ylabel="Log Score", title="Regularization Results")
        g.fig.savefig("test")

      
    else:
      # loss(true, pred) is the loss function to minimize
      # define the function to optimize
      optimization_function = lambda b: loss(data[self.dependent], self.predict(data, b))
      # optimize
      res = minimize(optimization_function, x0=X0())
      if print_results: print(res)
      self.betas = res.x
      
     
