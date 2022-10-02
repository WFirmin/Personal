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
- elasticNet(): modifies a loss function to apply elastic net regularization
- coefSig(): given coefficients from a number of trials, this function returns the average of each coefficient and a DataFrame that displays information about each coefficient
- Model: framework for building custom parametric machine learning models.  Focuses on compounding simpler, interpretable models for insightful use.

## Visuals
Tools for visualizing data
- gradient(): creates a gradient between two given colors

## Modeling Tutorial:
#### Decomposing a probability into two events:
Suppose we observe Y and want to model it as a decomposition into two events:

$P(Y)=P(A)P(B)$

This modeling framework allows specification of the model

$P(Y|X)=Logit(\beta_AX_A)Logit(\beta_BX_B)$.

Example: let's model $Y=Logit(3-2x_1)Logit(-2+5x_2)$
```python
import numpy as np
import pandas as pd
import WFirmin.Toolbox as wf

# Create the data:
x1 = np.random.normal(size=1000)
x2 = np.random.normal(size=1000)
y = wf.logit(3 - 2*x1 + np.random.normal(scale=0.05, size=1000)) * wf.logit(-2 + 5*x2 + np.random.normal(scale=0.05, size=1000))
data = pd.DataFrame({"Y":y,"X1":x1,"X2":x2})

# Specify the model:
# The formula uses arrays, where b specifies coefficients and v specifies variables, as per their name in the data.
formula = lambda b,v: wf.logit(b[0] + b[1]*v["X1"]) * wf.logit(b[2] + b[3]*v["X2"])
reg = wf.Model(dependent="Y", formula=formula, n_betas=4)

# Fit the model:
reg.fit(data, resample="CV", K=5, print_results=True)
```
Output:
```
Score: 3.179862130366884e-05
95% CI: [2.658395032143908e-05, 3.70132922858986e-05]
    Coefficient  Std Error       P-Value Significance
b0     2.982256   0.002079  1.415818e-12          ***
b1    -1.996660   0.002696  1.994493e-11          ***
b2    -2.003128   0.000518  2.676821e-14          ***
b3     5.005381   0.002472  3.570912e-13          ***
```



## Features in progress:
### Modeling Framework:
- Feature selection
- Log loss function

