# Will Firmin's Personal Toolbox

## Setup:
Install via the below command in CMD:
```sh
pip install git+https://github.com/WFirmin/Personal.git
```

Import:
<br>import WFirmin.Toolbox as wf

## Toolbox
Contains main tools:
- cut(): creates quantile labels for data based on several different methods
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
x1 = np.random.normal(1000)
x2 = np.random.normal(1000)
y = logit(3 - 2*x1 + np.random.normal(scale=0.05, 1000)) * logit(-2 + 5*x2 + np.random.normal(scale=0.05, 1000))
data = pd.DataFrame({"Y":y,"X1":x1,"X2":x2})

# Specify the model:
formula = lambda b,v: logit(b[0] + b[1]*v["X1"]) * logit(b[2] + b[3]*v["X2"])
reg = Model(dependent="Y", formula=formula, n_betas=4)

# Fit the model:
reg.fit(data)

```



## Features in progress:
### Modeling Framework:
- Elastic Net regularization (complete, to be committed)
- Standardization (complete, to be committed)
- Cross validation (complete, to be committed)
- Feature selection
- Log loss function
- Sampling within data

