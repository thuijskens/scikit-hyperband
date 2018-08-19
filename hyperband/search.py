"""
=========
Hyperband
=========

This module contains a scikit-learn compatible implementation of the hyperband
algorithm[^1].

Compared to the civismlext implementation, this supports multimetric scoring,
and the option to turn the last round of hyperband (the randomized search
round) off.

References
----------

.. [1] Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A. and Talwalkar, A.,
   2017. Hyperband: A novel bandit-based approach to hyperparameter
   optimization. The Journal of Machine Learning Research, 18(1),
   pp.6765-6816.

"""
import copy
from collections import defaultdict
from functools import partial

import numpy as np
from scipy.stats import rankdata

from sklearn.base import is_classifier, clone
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.utils import check_random_state
from sklearn.utils.validation import indexable
from sklearn.utils.fixes import MaskedArray

from sklearn.model_selection._search import BaseSearchCV, ParameterSampler
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _aggregate_score_dicts, _fit_and_score


__all__ = ['HyperbandSearchCV']


def _store_results(results, n_splits, n_candidates, key_name,
                   array, weights=None, splits=False, rank=False):
    """A small helper to store the scores/times to the cv_results_
    Taken from sklearn.model_selection._search.BaseSearchCV
    """
    array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                      n_splits)
    if splits:
        for split_i in range(n_splits):
            results["split%d_%s"
                    % (split_i, key_name)] = array[:, split_i]

    array_means = np.average(array, axis=1, weights=weights)
    results['mean_%s' % key_name] = array_means
    # Weighted std is not directly available in numpy
    array_stds = np.sqrt(np.average((array -
                                     array_means[:, np.newaxis]) ** 2,
                                    axis=1, weights=weights))
    results['std_%s' % key_name] = array_stds

    if rank:
        results["rank_%s" % key_name] = np.asarray(
            rankdata(-array_means, method='min'), dtype=np.int32)

    return results


class HyperbandSearchCV(BaseSearchCV):
    """Hyperband search on hyper parameters.

    HyperbandSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.

    If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Read more in the :ref:`User Guide <randomized_parameter_search>`.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    eta : float
        The inverse of the proportion of configurations that are discarded
        in each round of hyperband.

    max_iter : int
        The maximum amount of resource that can be allocated to the cost
        parameter ``resource_param`` for a single configuration of the
        hyperparameters.

    resource_param : str
        The name of the cost parameter for the estimator ``estimator``
        to be fitted. Typically, this is the number of decision trees
        ``n_estimators`` in an ensemble or the number of iterations
        for estimators trained with stochastic gradient descent.

    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's default scorer (if available) is used.

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    refit : boolean, or string default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a string denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``RandomizedSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_parameters_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    random_state : int, RandomState instance or None, optional, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : boolean, optional, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |        0.8        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |        0.9        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |        0.7        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.8, 0.9, 0.7],
            'split1_test_score'  : [0.82, 0.5, 0.7],
            'mean_test_score'    : [0.81, 0.7, 0.7],
            'std_test_score'     : [0.02, 0.2, 0.],
            'rank_test_score'    : [3, 1, 1],
            'split0_train_score' : [0.8, 0.9, 0.7],
            'split1_train_score' : [0.82, 0.5, 0.7],
            'mean_train_score'   : [0.81, 0.7, 0.7],
            'std_train_score'    : [0.03, 0.03, 0.04],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
            'params'             : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        For multi-metric evaluation, this attribute is present only if
        ``refit`` is specified.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    References
    ----------

    .. [1] Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A. and Talwalkar, A.,
           2017. Hyperband: A novel bandit-based approach to hyperparameter
           optimization. The Journal of Machine Learning Research, 18(1),
           pp.6765-6816.

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    --------
    :class:`GridSearchCV`:
        Does exhaustive search over a grid of parameters.

    :class:`ParameterSampler`:
        A generator over parameter settings, constructed from
        param_distributions.

    """
    def __init__(self, estimator, param_distributions, eta=3, max_iter=81,
                 resource_param='n_estimators', scoring=None, n_jobs=1,
                 iid=True, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', random_state=None,
                 error_score='raise', return_train_score=False):
        self.param_distributions = param_distributions
        self.eta = eta
        self.max_iter = max_iter
        self.resource_param = resource_param
        self.random_state = random_state
        super(HyperbandSearchCV, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=None,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)

    def _process_results(self, out, n_splits, scorers, refit_metric):
        """return results dict and best dict for given outputs
        Taken from sklearn.model_selection._search.BaseSearchCV.fit"""

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            (train_score_dicts, test_score_dicts, test_sample_counts,
             fit_time, score_time, parameters) = zip(*out)
        else:
            (test_score_dicts, test_sample_counts,
             fit_time, score_time, parameters) = zip(*out)

        candidate_params = parameters[::n_splits]
        n_candidates = len(candidate_params)

        # test_score_dicts and train_score dicts are lists of dictionaries and
        # we make them into dict of lists
        test_scores = _aggregate_score_dicts(test_score_dicts)
        if self.return_train_score:
            train_scores = _aggregate_score_dicts(train_score_dicts)

        results = dict()

        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates,),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)

        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        # Computed the (weighted) mean and std for test scores alone
        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits],
                                      dtype=np.int)

        for scorer_name in scorers.keys():
            # Computed the (weighted) mean and std for test scores alone
            results = _store_results(results, n_splits, n_candidates,
                                     'test_%s' % scorer_name, test_scores[scorer_name],
                                     splits=True, rank=True, weights=test_sample_counts if self.iid else None)
            if self.return_train_score:
                results = _store_results(
                    results, n_splits, n_candidates,
                    'train_%s' % scorer_name, train_scores[scorer_name], splits=True)

        results = _store_results(
            results, n_splits, n_candidates, 'fit_time', fit_time)
        results = _store_results(
            results, n_splits, n_candidates, 'score_time', score_time)

        best_index = results["rank_test_%s" % refit_metric].argmin()

        return results, best_index

    def fit(self, X, y=None, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        scorers, self.multimetric_ = _check_multimetric_scoring(
            self.estimator, scoring=self.scoring)

        if self.multimetric_:
            if self.refit is not False and (
                    not isinstance(self.refit, six.string_types) or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers):
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key "
                                 "to refit an estimator with the best "
                                 "parameter setting on the whole data and "
                                 "make the best_* attributes "
                                 "available for that metric. If this is not "
                                 "needed, refit should be set to False "
                                 "explicitly. %r was passed." % self.refit)
            else:
                refit_metric = self.refit
        else:
            refit_metric = 'score'

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)
        pre_dispatch = self.pre_dispatch
        random_state = check_random_state(self.random_state)

        # Here is where hyperband comes into play
        s_max = int(np.floor(np.log(self.max_iter) / np.log(self.eta)))
        B = (s_max + 1) * self.max_iter

        all_results = []

        for s in range(s_max, -1, -1):
            n = int(np.ceil(B / self.max_iter / (s + 1) * np.power(self.eta, s)))

            # initial number of iterations per config
            r = self.max_iter / np.power(self.eta, s)
            configurations = list(ParameterSampler(param_distributions=self.param_distributions,
                                                   n_iter=n,
                                                   random_state=random_state))

            for i in range(s + 1):
                n_configs = np.floor(n / np.power(self.eta, i))  # n_i
                n_iterations = int(r * np.power(self.eta, i))  # r_i
                n_to_keep = int(np.floor(n_configs / self.eta))

                # Create a queue with jobs for joblib
                jobs = []
                for configuration in configurations:
                    parameters = copy.deepcopy(configuration)
                    parameters[self.resource_param] = n_iterations

                    for train, test in cv.split(X, y, groups):
                        jobs.append(delayed(_fit_and_score)(
                            clone(base_estimator), X, y, scorers,
                            train, test, self.verbose, parameters,
                            fit_params=fit_params,
                            return_train_score=self.return_train_score,
                            return_n_test_samples=True,
                            return_times=True, return_parameters=True,
                            error_score=self.error_score
                        ))

                out = Parallel(
                    n_jobs=self.n_jobs, verbose=self.verbose,
                    pre_dispatch=pre_dispatch)(jobs)
                all_results += out

                if n_to_keep > 0:
                    results, _ = self._process_results(out, n_splits, scorers, refit_metric)
                    top_configurations = [x for _, x in sorted(zip(results['rank_test_%s' % refit_metric],
                                                                   results['params']),
                                                               key=lambda x: x[0])]

                    configurations = top_configurations[:n_to_keep]

        # Collect all results. This is where hyperband ends and sklearn code
        # takes over
        all_models, best_index = self._process_results(all_results, n_splits, scorers, refit_metric)

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ if refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_params_ = all_models['params'][best_index]
            self.best_score_ = all_models["mean_test_%s" % refit_metric][best_index]

        if self.refit:
            self.best_estimator_ = clone(base_estimator).set_params(
                **self.best_params_)
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers['score']

        self.cv_results_ = all_models
        self.n_splits_ = n_splits

        return self


if __name__ == '__main__':
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier

    from scipy.stats import randint as sp_randint

    # generate some fake data and model
    # get some data
    digits = load_digits()
    X, y = digits.data, digits.target

    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(2, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # build a classifier
    clf = RandomForestClassifier(n_estimators=20)

    search = HyperbandSearchCV(estimator=clf, param_distributions=param_dist)

    search.fit(X, y)

    print(search.best_estimator_)
    print(search.best_params_)
    print(search.best_score_)

    print('\n\n')
    import pandas as pd
    print(pd.DataFrame(search.cv_results_))



