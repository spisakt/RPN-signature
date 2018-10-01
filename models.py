from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn import preprocessing
from nilearn.connectome import ConnectivityMeasure
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FastICA
from sklearn.cross_decomposition import PLSRegression
# just to define some default models

def pipe_conn_scale_fsel_model(conn=ConnectivityMeasure(kind="tangent", vectorize=True, discard_diagonal=True),
                               scaler=preprocessing.StandardScaler(),
                               fsel=SelectKBest(f_regression),
                               model=ElasticNet(max_iter=10000),
                               p_grid={'fsel__k': [30], 'model__alpha': [.6], 'model__l1_ratio': [.02]}
                               ):
    mymodel = Pipeline(
        [('conn', conn), ('scaler', scaler), ('fsel', fsel),
         ('model', model)])


    return mymodel, p_grid


def pipe_scale_fsel_model(scaler=preprocessing.StandardScaler(),
                          fsel=SelectKBest(f_regression),
                          model=ElasticNet(max_iter=100000),
                          p_grid={'fsel__k': [30], 'model__alpha': [.6], 'model__l1_ratio': [.02]}
                          ):
    mymodel = Pipeline(
        [('scaler', scaler), ('fsel', fsel),
         ('model', model)])
    return mymodel, p_grid


def pipe_scale_fsel_fgen_model(scaler=preprocessing.StandardScaler(),
                          fsel=SelectKBest(f_regression),
                          fgen=preprocessing.PolynomialFeatures(),
                          model=ElasticNet(max_iter=100000),
                          p_grid={'fsel__k': [30], 'model__alpha': [.6], 'model__l1_ratio': [.02]}
                          ):
    mymodel = Pipeline(
        [('scaler', scaler),
         ('fsel', fsel),
         ('fgen', fgen),
         ('model', model)])
    return mymodel, p_grid

def pipe_scale_fsel_ridge(scaler=preprocessing.StandardScaler(),
                          fsel=SelectKBest(f_regression),
                          model=Ridge(max_iter=10000),
                          p_grid={'fsel__k': [30], 'model__alpha': [.6]}
                          ):
    mymodel = Pipeline(
        [('scaler', scaler), ('fsel', fsel),
         ('model', model)])
    return mymodel, p_grid

def pipe_scale_dimred_model(scaler=preprocessing.StandardScaler(),
                          dimred=FastICA(max_iter=10000),
                          model=ElasticNet(max_iter=10000),
                          p_grid={'dimred__n_components': [10, 100], 'model__alpha': [.1, .5, 1.], 'model__l1_ratio': [.1, .5, .9]}
                          ):
    mymodel = Pipeline(
        [('scaler', scaler), ('dimred', dimred),
         ('model', model)])
    return mymodel, p_grid


def pipe_scale_model(scaler=preprocessing.StandardScaler(),
                          model=ElasticNet(max_iter=10000),
                          p_grid={'model__alpha': [.6], 'model__l1_ratio': [.02]}
                          ):
    mymodel = Pipeline(
        [('scaler', scaler),
         ('model', model)])
    return mymodel, p_grid

def pipe_scale_lasso(scaler=preprocessing.StandardScaler(),
                          model=ElasticNet(),
                          p_grid={'model__alpha': [.1, 1., 10.],}
                          ):
    mymodel = Pipeline(
        [('scaler', scaler),
         ('model', model)])
    return mymodel, p_grid

def pipe_scale_ridge(scaler=preprocessing.StandardScaler(),
                          model=ElasticNet(),
                          p_grid={'model__alpha': [.1, 1., 10.],}
                          ):
    mymodel = Pipeline(
        [('scaler', scaler),
         ('model', model)])
    return mymodel, p_grid

def pipe_scale_pls(scaler=preprocessing.StandardScaler(),
                          model=PLSRegression(),
                          p_grid={'model__n_components': [1, 2, 3],}
                          ):
    mymodel = Pipeline(
        [('scaler', scaler),
         ('model', model)])
    return mymodel, p_grid
