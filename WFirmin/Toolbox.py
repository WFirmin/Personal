import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, t
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
  
  
# sample mean and standard deviation of mean
def meanStd(v):
  v = np.array(v)
  mean = v.mean()
  std = v.std(ddof=1)
  mStd = std / np.sqrt(len(v))
  return (mean, mStd)
  



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

# take list of coefficients from trials and deliver significance levels
def coefSig(coefs):
  coefs = pd.DataFrame(coefs)
  beta_names = ["b"+str(i) for i in range(coefs.shape[1])]
  coefs.columns = beta_names
  trial_results = pd.DataFrame(coefs.mean())
  trial_results.columns=["Coefficient"]
  K = len(coefs)
  trial_results["Std Error"] = coefs.std(ddof=1) / np.sqrt(K)
  trial_results["P-Value"] = t.sf(np.absolute(trial_results["Coefficient"] / trial_results["Std Error"]), df=K-1)*2
  sigs = {".":0.1, "*":0.05,"**":0.01,"***":0.001}
  sig = pd.Series([""]*coefs.shape[1], index=beta_names)
  for level in sigs.keys():
    sig[trial_results["P-Value"] <= sigs[level]] = level
  trial_results["Significance"] = sig
  betas = np.array(trial_results.Coefficient) # would be self if in class
  #self.results = trial_results
  return (betas, trial_results)


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
    enet=False, lam=np.e**np.linspace(-20,1,20), alpha=1, standardize=False,
    sample=None):
    
    X0 = lambda: self.betas if warm_start else [0]*self.n_betas
    if sample != None:
      if type(sample) == float:
        sampleIndex = random.sample(list(data.index), round(sample*len(data)))
      elif type(sample) == int:
        sampleIndex = random.sample(list(data.index), sample)
      else:
        raise TypeError("sample parameter must be a float or int")
      data = data.loc[sampleIndex]
      
    if standardize:
      self.dataMeans = data.drop(columns=self.dependent).mean()
      self.dataStds = data.drop(columns=self.dependent).std(ddof=1)
      data.loc[:,data.columns!=self.dependent] = (data.loc[:,data.columns!=self.dependent] - self.dataMeans) / self.dataStds
      
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
                pd.DataFrame({"Alpha":a,"Lambda":l,"Group":i, "Score":score, "Coefficients":",".join(res.x.astype(str))}, index=[0])], ignore_index=True)
      else:
        for a in alpha:
          for l in lam:
            for i in range(len(indices)):
              optimization_function = lambda b: elasticNet(loss(data.loc[indices[i],self.dependent], self.predict(data.loc[indices[i]], b)), a, l, b)
              res = minimize(optimization_function, x0=X0())
              score = loss(data.loc[indices[i], self.dependent], self.predict(data.loc[indices[i]], res.x))
              enet_results = pd.concat([enet_results,
                pd.DataFrame({"Alpha":a,"Lambda":l,"Group":i,"Score":score, "Coefficients":",".join(res.x.astype(str))}, index=[0])], ignore_index=True)
      
      results1 = enet_results.loc[:,["Alpha","Lambda","Score"]].groupby(["Alpha","Lambda"])
      results = results1.mean().merge(results1.std(ddof=1), on=["Alpha","Lambda"], suffixes=("_Mean","_Std"))
      results.Score_Std /= np.sqrt(K)
      self.enet_results = results
      self.regularization = results.iloc[np.argmin(results.Score_Mean)].name
      self.results = enet_results[(enet_results.Alpha == self.regularization[0]) & (enet_results.Lambda == self.regularization[1])]
      
      coefs = [np.array(c.split(',')).astype(float) for c in self.results.Coefficients]
      score = self.results.Score.copy()
      self.betas, self.results = coefSig(coefs)
      if resample != None and K > 1: self.score, self.scoreStd = meanStd(score)
      else: self.score = np.array(score).mean()
      if print_results: 
        print("Score:",self.score)
        if resample != None and K > 1:
          tc = t.isf(0.025,K-1)
          CI = [self.score - tc*self.scoreStd, self.score + tc*self.scoreStd]
          print("95% CI:", CI)
        print(self.results)
        print("\nBest: Alpha =",self.regularization[0],"; Lambda =",self.regularization[1])
        print(results)
        g = sns.relplot(x=np.log(enet_results.Lambda), y=np.log(enet_results.Score), hue="Alpha", data=enet_results, kind="line")\
          .set(xlabel="Log Lambda", ylabel="Log Score", title="Regularization Results")
        g.fig.savefig("test")

    else:
      results = []
      score = []
      if resample == "CV":
        for i in range(len(indices)):
          trainSample = concatList(indices[:i] + indices[i+1:])
          optimization_function = lambda b: loss(data.loc[trainSample, self.dependent], self.predict(data.loc[trainSample], b))
          res = minimize(optimization_function, x0=X0())
          results.append(res.x)
          score.append(loss(data.loc[indices[i], self.dependent], self.predict(data.loc[indices[i]], res.x)))
          
      else:
        for i in range(len(indices)):
          optimization_function = lambda b: loss(data.loc[indices[i], self.dependent], self.predict(data.loc[indices[i]], b))
          res = minimize(optimization_function, x0=X0())
          results.append(res.x)
          score.append(res.fun)
          
          
      self.betas, self.results = coefSig(results)
      if resample != None and K > 1: self.score, self.scoreStd = meanStd(score)
      else: self.score = np.array(score).mean()
      if print_results: 
        print("Score:",self.score)
        if resample != None and K > 1:
          tc = t.isf(0.025,len(score)-1)
          CI = [self.score - tc*self.scoreStd, self.score + tc*self.scoreStd]
          print("95% CI:", CI)
        print(self.results)

      
