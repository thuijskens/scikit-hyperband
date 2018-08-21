# Hyperband

A scikit-learn compatible implementation of hyperband. Work in progress.

## Installation

Clone the git repository 

```bash
git clone https://github.com/thuijskens/scikit-hyperband.git
```

and `cd` into the project directory and and install the package using `setuptools` as follows

```bash
python setup.py install
```

## Example usage

`scikit-hyperband` implements a class `HyperbandSearchCV` that works exactly as `GridSearchCV` and `RandomizedSearchCV` from scikit-learn do, except that it runs the hyperband algorithm under the hood. 

Similarly to the existing model selection routines, `HyperbandSearchCV` works for (multi-label) classification and regression, and supports multi-metric scoring in the same way as scikit-learn supports it. 

`HyperbandSearchCV` implements the following extra parameters, specific to the hyperband algorithm:

- `resource_param`: The name of the cost parameter of the estimator that you're tuning.
- `eta`: The inverse of the proportion of configurations that are discarded in each round of hyperband.
- `min_iter`: The minimum amount of resource that should be allocated to the cost parameter ``resource_param`` for a single configuration of the hyperparameters.
- `max_iter`: The maximum amount of resource that can be allocated to the cost parameter ``resource_param`` for a single configuration of the hyperparameters.
- `skip_last`: The number of last rounds to skip. For example, this can be used to skip the last round of hyperband, which is standard randomized search. It can also be used to inspect intermediate results, although warm-starting HyperbandSearchCV is not supported.

```python
from hyperband import HyperbandSearchCV

from scipy.stats import randint as sp_randint
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
param_dist = {
    'max_depth': [3, None],
    'max_features': sp_randint(1, 11),
    'min_samples_split': sp_randint(2, 11),
    'min_samples_leaf': sp_randint(1, 11),
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

digits = load_digits()
X, y = digits.data, digits.target

search = HyperbandSearchCV(model, param_dist, 
                           resource_param='n_estimators',
                           scoring='roc_auc')
search.fit(X, y)
print(search.best_params_)
```
 