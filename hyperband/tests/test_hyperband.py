from hyperband import HyperbandSearchCV

from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint

from sklearn.datasets import load_digits
from sklearn.utils import check_random_state


def setup():
    model = RandomForestClassifier()
    rng = check_random_state(42)
    param_dist = {'max_depth': [3, None],
                  'max_features': sp_randint(1, 11),
                  'min_samples_split': sp_randint(2, 11),
                  'min_samples_leaf': sp_randint(1, 11),
                  'bootstrap': [True, False],
                  'criterion': ['gini', 'entropy']}
    
    digits = load_digits()
    X, y = digits.data, digits.target

    return model, param_dist, X, y, rng

"""
TODO: This test fails due to the random state not being properly fixed

def test_hyperband():
    model, param_dist, X, y, rng = setup()
    search = HyperbandSearchCV(model, param_dist, random_state=rng)
    search.fit(X, y)

    # results = pd.DataFrame(search.cv_results_)
    expected_params = {
        'bootstrap': False,
        'criterion': 'entropy',
        'max_depth': None,
        'max_features': 7,
        'min_samples_leaf': 2,
        'min_samples_split': 2,
        'n_estimators': 81
    }

    # assert(results.shape[0] == 186) TODO: sort out what the expected n_i and r_i values are
    assert(search.best_params_ == expected_params)
"""


def test_multimetric_hyperband():
    model, param_dist, X, y, rng = setup()

    # multimetric scoring is only supported for 1-D classification
    first_label = (y == 1)
    y[first_label] = 1
    y[~first_label] = 0

    multimetric = [
        'roc_auc',
        'accuracy'
    ]

    search = HyperbandSearchCV(model, param_dist, refit='roc_auc', scoring=multimetric,
                               random_state=rng)
    search.fit(X, y)

    assert('mean_test_roc_auc' in search.cv_results_.keys())
    assert('mean_test_accuracy' in search.cv_results_.keys())
    # assert(search.best_params_ == results[results.rank_test_roc_auc == 1].params.values)


def test_warm_start():
    model, param_dist, X, y, rng = setup()

    model.set_params(warm_start=True)
    search = HyperbandSearchCV(model, param_dist, random_state=rng, warm_start=True,
                               verbose=1)
    search.fit(X, y)


def test_min_resource_param():
    model, param_dist, X, y, rng = setup()
    search = HyperbandSearchCV(model, param_dist, min_iter=3, random_state=rng,
                               verbose=1)
    search.fit(X, y)

    assert(search.cv_results_['param_n_estimators'].data.min() == 3)
