# Hyperband

[![Build Status](https://travis-ci.org/thuijskens/scikit-hyperband.svg?branch=master)](https://travis-ci.org/thuijskens/scikit-hyperband)
[![CircleCI](https://circleci.com/gh/thuijskens/scikit-hyperband/tree/master.svg?style=svg)](https://circleci.com/gh/thuijskens/scikit-hyperband/tree/master)
[![Coverage Status](https://coveralls.io/repos/github/thuijskens/scikit-hyperband/badge.svg?branch=master)](https://coveralls.io/github/thuijskens/scikit-hyperband?branch=master)

A scikit-learn compatible implementation of hyperband.

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

See the [documentation](https://thuijskens.github.io/scikit-hyperband/docs/) for the full parameter list.

```python
from hyperband import HyperbandSearchCV

from scipy.stats import randint as sp_randint
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer

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
y = LabelBinarizer().fit_transform(y)

search = HyperbandSearchCV(model, param_dist, 
                           resource_param='n_estimators',
                           scoring='roc_auc')
search.fit(X, y)
print(search.best_params_)
```

## References

* Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A. and Talwalkar, A., 2017. Hyperband: A novel bandit-based approach to hyperparameter optimization. The Journal of Machine Learning Research, 18(1), pp.6765-6816. http://www.jmlr.org/papers/volume18/16-558/16-558.pdf
* Whilst developing scikit-hyperband, I stumbled on an implementation of hyperband in [civisml-extensions](https://github.com/civisanalytics/civisml-extensions). Their implementation exposes more parallelism than this one does, although this implementation has a couple of extra options you can tweak (`skip_last`, multi-metric scoring)
* FastML has a nice blog post explaining hyperband [here](http://fastml.com/tuning-hyperparams-fast-with-hyperband/). 
