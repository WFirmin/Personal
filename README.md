# Will Firmin's Personal Toolbox

## Setup:
Install via the below command in CMD:
```sh
pip install git+https://github.com/WFirmin/Personal.git
```

Import:
```python
import WFirmin.Toolbox as wf
```

## Toolbox
Contains main tools:
- cut(): creates quantile labels for data based on several different methods
- concatList(): concatenates a list of lists
- Resample(): returns a list of indices corresponding to a resampling method: cross validation or bootstrapping
- meanStd(): returns a tuple of an array's mean and the estimated standard deviation of the mean
- logit(): applies the logit function to an array
- MSE(): returns the mean squared error, given arrays of the true and predicted values
- LogLoss(): returns the log loss, given arrays of the true and predicted values
- elasticNet(): modifies a loss function to apply elastic net regularization
- coefSig(): given coefficients from a number of trials, this function returns the average of each coefficient and a DataFrame that displays information about each coefficient
- Model: framework for building custom parametric machine learning models.  Focuses on compounding simpler, interpretable models for insightful use.

## Visuals
Tools for visualizing data
- gradient(): creates a gradient between two given colors

## Modeling Tutorial:
#### Decomposing a probability into two events:
Suppose we observe Y and want to model it as a decomposition into two events:

$P(Y)=P(A)P(B|A)$

This modeling framework allows specification of the model

$P(Y|X)=Logit(\beta_AX_A)Logit(\beta_BX_B)$.

Example: let's model $Y=Logit(3-2x_1)Logit(-2+5x_2)$
```python
import numpy as np
import pandas as pd
import WFirmin.Toolbox as wf

# Create the data:
n = 10000
x1 = np.random.normal(size=n, scale=2)
x2 = np.random.normal(size=n, scale=2)
y = wf.logit(3 - 2*x1) * wf.logit(-2 + 5*x2)
r = np.random.uniform(size=n)
y = (r <= y).astype(int)
data = pd.DataFrame({"Y":y,"X1":x1,"X2":x2})

# Specify the model:
# Use v for variables in the data, b for coefficients
formula = lambda b,v: wf.logit(b[0] + b[1]*v["X1"]) * wf.logit(b[2] + b[3]*v["X2"])
reg = wf.Model(dependent="Y", formula=formula, n_betas=4)

# Fit the model:
reg.fit(data, loss=wf.LogLoss, resample="CV", K=3, print_results=True)
```
Output:
```
Score: 0.1802721797591775
95% CI: [0.15625091694427176, 0.20429344257408324]
    Coefficient  Std Error   P-Value Significance
b0     3.113760   0.128042  0.001687           **
b1    -2.025705   0.063896  0.000993          ***
b2    -2.109533   0.027244  0.000167          ***
b3     5.274653   0.138666  0.000690          ***
```



## Features in progress:
### Modeling Framework:
- Feature selection

