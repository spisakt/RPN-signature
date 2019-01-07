import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting
sns.set_style("white")

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def plot_matrix(mat, labels, modules, outfile="", zero_diag=True):
    # plot group-mean matrix
    norm = MidpointNormalize(midpoint=0)

    if zero_diag:
        mat[range(mat.shape[0]), range(mat.shape[0])] = 0

    plotting.plot_matrix(mat, labels=labels.tolist(), auto_fit=True, norm=norm,
                         cmap=ListedColormap(sns.diverging_palette(220, 15, sep=1, n=31)), figure=(10, 10))

    prev=""
    idx=0
    for i in range(len(labels)):
        if modules[i]!=prev:
            plt.plot([-5, len(labels) + 0.5], [i-0.5, i-0.5], linewidth=1, color='gray')
            plt.plot([i - 0.5, i - 0.5], [-5, len(labels) + 0.5], linewidth=1, color='gray')

            idx=idx+1
            prev=modules[i]

    if outfile:
        figure = plt.gcf()
        figure.savefig(outfile, bbox_inches='tight')
        plt.close(figure)
    else:
        plotting.show()


def plot_connectome(mat, label_map, threshold='95%', GS=True):

    if GS:
        mat=mat[1:, 1:] #fisrt row and column is global signal
    coords = plotting.find_parcellation_cut_coords(label_map)
    view = plotting.view_connectome(mat, coords, threshold=threshold)
    view.open_in_browser()

def plot_labelmap(label_map):

    cols =[ "#FCF9F5",
            "#C06A45",  # CER
            "#5B5BFF",  # DMN
            "#D73E68",  # FP
            "#8D18AB",  # LIM
            "#0AFE47",  # MOT
            "#FF9C42",  # VAT_SAL_SUB
            "#FFFFAA"  # VIS
            ]

    from nilearn import surface
    from nilearn import plotting, datasets
    import numpy.linalg as npl
    import nibabel as nb

    fsaverage = datasets.fetch_surf_fsaverage()

    from PAINTeR import utils
    s = utils.load_surface_obj('/Users/tspisak/tmp/wm_gm_simp2.obj')


    s2 = surface.load_surf_mesh(fsaverage['pial_left'])

    from nibabel.affines import apply_affine

    img = nb.load(label_map)
    data = img.get_data()

    import pandas as pd
    from PAINTeR import global_vars
    l = pd.read_csv(global_vars._ATLAS_LABELS_, sep="\t")
    modules = l['modules'].values
    lut = pd.factorize(modules)[0] + 1
    lut = np.array([0] + lut.tolist())
    data = lut[np.array(data, dtype=int)]


    parcellation=np.repeat(0, len(s[0]))
    for i in range(len(s[0])):
        coord = np.round(apply_affine(npl.inv(img.affine), s[0][i])).astype(int)
        if coord[0]-1 >= data.shape[0] or coord[1]-1 >= data.shape[1] or coord[2]-1 >= data.shape[2]:
            parcellation[i] = 0
        else:
            parcellation[i] = data[coord[0]-1, coord[1]-1, coord[2]-1]

    import matplotlib.cm as cm
    view = plotting.view_surf(s, surf_map=parcellation,
                                     cmap = ListedColormap(sns.color_palette(cols)), # ListedColormap(cm.get_cmap('tab20').colors)
                                     threshold=0, symmetric_cmap=False)
    view.open_in_browser()

def plot_prediction(observed, predicted, outfile="", covar=[], robust=False, sd=True, text=""):
    color = "black"
    if len(covar):
        g = sns.jointplot(observed, predicted, scatter=False, color=color, kind="reg", robust=robust, x_ci="sd", )
        plt.scatter(observed, predicted,
                    c=covar, cmap=ListedColormap(sns.color_palette(["#5B5BFF","#D73E68"])))
    else:
        g = sns.jointplot(observed, predicted, kind="reg", color=color, robust=robust, x_ci="sd")
    #sns.regplot(observed, predicted, color="b", x_bins=10, x_ci=None)



    if sd:
        xlims=np.array(g.ax_joint.get_xlim())
        if robust:
            res = sm.RLM(predicted, sm.add_constant(observed)).fit()
            coefs = res.params
            residual = res.resid
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(observed, predicted)
            coefs=[intercept, slope]
            regline = slope * observed + intercept
            residual = observed - regline

        S = np.sqrt(np.mean(residual**2))
        upper = coefs[1] * xlims + coefs[0] + S/2
        lower = coefs[1] * xlims + coefs[0] - S/2

        plt.plot(xlims, upper, ':', color=color, linewidth=1, alpha=0.3)
        plt.plot(xlims, lower, ':', color=color, linewidth=1, alpha=0.3)

    if text:
        plt.text(np.min(observed) - (np.max(predicted)-np.min(predicted))/3,
                 np.max(predicted) + (np.max(predicted)-np.min(predicted))/3,
                 text, fontsize=10)

    if outfile:
        figure = plt.gcf()
        figure.savefig(outfile, bbox_inches='tight')
        plt.close(figure)
    else:
        plt.show()

