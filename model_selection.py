import numpy as np

from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold
from sklearn.model_selection._split import check_cv, _RepeatedSplits

# following code is adapted from: https://github.com/biocore/calour/blob/master/calour/training.py#L144

class SortedStratifiedKFold(StratifiedKFold):
    '''Stratified K-Fold cross validator.
    Please see :class:`sklearn.model_selection.StratifiedKFold` for
    documentation for parameters, etc. It is very similar to that
    except this is for regression of numeric values.
    This implementation basically assigns a unique label (int here) to
    each consecutive `n_splits` values after y is sorted. Then rely on
    StratifiedKFold to split. The idea is borrowed from this `blog
    <http://scottclowe.com/2016-03-19-stratified-regression-partitions/>`_.
    See Also
    --------
    RepeatedSortedStratifiedKFold
    '''

    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        super(SortedStratifiedKFold, self).__init__(n_splits, shuffle, random_state)

    def _sort_partition(self, y):
        n = len(y)
        cats = np.empty(n, dtype='u4')
        div, mod = divmod(n, self.n_splits)
        cats[:n - mod] = np.repeat(range(div), self.n_splits)
        cats[n - mod:] =  [ x%div for x in range(mod)] # cahnged from div + 1
        return cats[np.argsort(np.argsort(y))]

    def split(self, X, y, groups=None):
        y_cat = self._sort_partition(y)
        return super(SortedStratifiedKFold, self).split(X, y_cat, groups)


class RepeatedSortedStratifiedKFold(_RepeatedSplits):
    '''Repeated Stratified K-Fold cross validator.
    Please see :class:`sklearn.model_selection.RepeatedStratifiedKFold` for
    documentation for parameters, etc. It is very similar to that
    except this is for regression of numeric values.
    See Also
    --------
    SortedStratifiedKFold
    '''

    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        super(RepeatedSortedStratifiedKFold, self).__init__(SortedStratifiedKFold, n_repeats, random_state,
                                                            n_splits=n_splits)


