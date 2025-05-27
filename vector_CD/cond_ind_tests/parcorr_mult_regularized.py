"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from __future__ import print_function
from scipy import stats
import numpy as np
import sys
import warnings
from sklearn.linear_model import LinearRegression
from numpy.linalg import eigh, eigvalsh

# from . import independence_tests_base
from .independence_tests_base import CondIndTest

class ParCorrMult(CondIndTest):
    r"""Partial correlation test for multivariate X and Y.

    Multivariate partial correlation is estimated through regularized regression
    and some test for multivariate dependency among the residuals.

    Notes
    -----
    To test :math:`X \perp Y | Z`, first :math:`Z` is regressed out from
    :math:`X` and :math:`Y` assuming the  model

    .. math::  X & =  Z \beta_X + \epsilon_{X} \\
        Y & =  Z \beta_Y + \epsilon_{Y}

    using OLS regression. Then different measures for the dependency among the residuals
    can be used. Currently only a test for zero correlation on the maximum of the residuals'
    correlation is performed.

    Parameters
    ----------
    correlation_type : {'max_corr'}
        Which dependency measure to use on residuals.
    **kwargs :
        Arguments passed on to Parent class CondIndTest.
    """
    # documentation
    @property
    def measure(self):
        """
        Concrete property to return the measure of the independence test
        """
        return self._measure

    def __init__(self, correlation_type='max_corr',
                    regularization_model=None,
                     **kwargs):
        self._measure = 'par_corr_mult'
        self.two_sided = True
        self.residual_based = True
        self.regularization_model = regularization_model
        if self.regularization_model is None:
            self.regularization_model = LinearRegression()

        self.correlation_type = correlation_type

        if self.correlation_type not in ['max_corr', 'gcm','gcm_gmb','linear_hsic_shuffle','linear_hsic_shuffle_kci','linear_hsic_approx']:
            raise ValueError("correlation_type must be in ['max_corr','gcm', 'linear_hsic'].")

        CondIndTest.__init__(self, **kwargs)

    def _get_single_residuals(self, array, xyz, target_var,
                              standardize=True,
                              return_means=False):
        """Returns residuals of linear multiple regression.

        Performs a OLS regression of the variable indexed by target_var on the
        conditions Z. Here array is assumed to contain X and Y as the first two
        rows with the remaining rows (if present) containing the conditions Z.
        Optionally returns the estimated regression line.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        target_var : {0, 1}
            Variable to regress out conditions from.

        standardize : bool, optional (default: True)
            Whether to standardize the array beforehand. Must be used for
            partial correlation.

        return_means : bool, optional (default: False)
            Whether to return the estimated regression line.

        Returns
        -------
        resid [, mean] : array-like
            The residual of the regression and optionally the estimated line.
        """

        dim, T = array.shape
        dim_z = (xyz == 2).sum()

        # Standardize
        if standardize:
            array -= array.mean(axis=1).reshape(dim, 1)
            std = array.std(axis=1)
            for i in range(dim):
                if std[i] != 0.:
                    array[i] /= std[i]
            if np.any(std == 0.) and self.verbosity > 0:
                warnings.warn("Possibly constant array!")
            # array /= array.std(axis=1).reshape(dim, 1)
            # if np.isnan(array).sum() != 0:
            #     raise ValueError("nans after standardizing, "
            #                      "possibly constant array!")

        y = np.transpose(array[np.where(xyz==target_var)[0], :])
        # print("y.shape ", y.shape)
        if dim_z > 0:
            z = np.transpose(array[np.where(xyz==2)[0], :])
            reg = self.regularization_model.fit(z, y)
            mean = reg.predict(z)
            # print("mean.shape ", mean.shape)
            # z = np.transpose(array[np.where(xyz==2)[0], :])
            # beta_hat = np.linalg.lstsq(z, y, rcond=None)[0]
            # mean = np.dot(z, beta_hat)
            resid = y - mean
        else:
            resid = y
            mean = None

        # print("resid.shape ", resid.shape)
        if return_means:
            return (np.transpose(resid), np.transpose(mean))

        return np.transpose(resid)

    def get_dependence_measure(self, array, xyz):
        """Return multivariate kernel correlation coefficient.

        Estimated as some dependency measure on the
        residuals of a linear OLS regression.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        Returns
        -------
        val : float
            Partial correlation coefficient.
        """

        dim, T = array.shape
        dim_x = (xyz==0).sum()
        dim_y = (xyz==1).sum()

        x_vals = self._get_single_residuals(array, xyz, target_var=0)
        y_vals = self._get_single_residuals(array, xyz, target_var=1)

        array_resid = np.vstack((x_vals.reshape(dim_x, T), y_vals.reshape(dim_y, T)))
        xyz_resid = np.array([index_code for index_code in xyz if index_code != 2])

        val = self.mult_corr(array_resid, xyz_resid)

        return val

    def mult_corr(self, array, xyz, standardize=True):
        """Return multivariate dependency measure.

        Parameters
        ----------
        array : array-like
            data array with X, Y in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        standardize : bool, optional (default: True)
            Whether to standardize the array beforehand. Must be used for
            partial correlation.

        Returns
        -------
        val : float
            Multivariate dependency measure.
        """

        dim, n = array.shape
        dim_x = (xyz==0).sum()
        dim_y = (xyz==1).sum()

        # Standardize
        if standardize:
            array -= array.mean(axis=1).reshape(dim, 1)
            std = array.std(axis=1)
            for i in range(dim):
                if std[i] != 0.:
                    array[i] /= std[i]
            if np.any(std == 0.) and self.verbosity > 0:
                warnings.warn("Possibly constant array!")
            # array /= array.std(axis=1).reshape(dim, 1)
            # if np.isnan(array).sum() != 0:
            #     raise ValueError("nans after standardizing, "
            #                      "possibly constant array!")

        x = array[np.where(xyz==0)[0]]
        y = array[np.where(xyz==1)[0]]

        if self.correlation_type == 'max_corr':
            # Get (positive or negative) absolute maximum correlation value
            corr = np.corrcoef(x, y)[:len(x), len(x):].flatten()
            val = corr[np.argmax(np.abs(corr))]

            # val = 0.
            # for x_vals in x:
            #     for y_vals in y:
            #         val_here, _ = stats.pearsonr(x_vals, y_vals)
            #         val = max(val, np.abs(val_here))

        elif 'linear_hsic' in self.correlation_type:
            # For linear kernel and standardized data (centered and divided by std)
            # biased V -statistic of HSIC reduces to sum of squared inner products
            # over all dimensions
            # val = ((x.dot(y.T)/float(n))**2).sum()   #JAKOBS Version, matches the causal-learn version disregarding the n**2

            # Adapted from causal-learn KCI.py   ##### standardize is with dof =1 in causal-learn (here dof=0)
            val, Kxc, Kyc = self.hsic_vstat (x,y,n)

            # print('val_kci',val)

        elif 'gcm' in self.correlation_type: #self.correlation_type == 'gcm':

            val = self.mult_corr_gcm(array,xyz)[0]


        else:
            raise NotImplementedError("Currently only"
                                      "correlation_type == 'max_corr','gcm','hsic' implemented.")

        return val

    def hsic_vstat(self,x,y,n):

        H = np.identity(n) - 1.0 / n
        # print('H shape',H.shape)
        Kx = x.T.dot(x)
        # print('x shape',x.shape)
        # print('Kx shape',Kx.shape)
        Kxc = H.dot(Kx.dot(H))
        Ky = y.T.dot(y)
        Kyc = H.dot(Ky.dot(H))
        val = np.sum(Kxc*Kyc)

        return val, Kxc, Kyc


    def mult_corr_gcm(self, array, xyz, standardize=True):
        """Return multivariate dependency measure and simulated test statistic using gaussian multiplier bootstrap.

        Parameters
        ----------
        array : array-like
            data array with X, Y in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        standardize : bool, optional (default: True)
            Whether to standardize the array beforehand. Must be used for
            partial correlation.

        Returns
        -------
        val : float
            Multivariate dependency measure.
        """

        dim, n = array.shape
        dim_x = (xyz==0).sum()
        dim_y = (xyz==1).sum()

        # Standardize
        if standardize:
            array -= array.mean(axis=1).reshape(dim, 1)
            std = array.std(axis=1)
            for i in range(dim):
                if std[i] != 0.:
                    array[i] /= std[i]
            if np.any(std == 0.) and self.verbosity > 0:
                warnings.warn("Possibly constant array!")
            # array /= array.std(axis=1).reshape(dim, 1)
            # if np.isnan(array).sum() != 0:
            #     raise ValueError("nans after standardizing, "
            #                      "possibly constant array!")

        x = array[np.where(xyz==0)[0]]
        y = array[np.where(xyz==1)[0]]


        #d_X = res_X.shape[1]; d_Y = res_Y.shape[1]; nn = res_X.shape[0]
        # rep(times) in R is really np.tile.
        # Translating R_mat = rep(....) * as.numeric(...)[, rep(....)]

        #print('old shape of x', x.shape)

        nn = x.shape[1]
        #print(nn)
        x = np.transpose(x)
        y = np.transpose(y)

        left = np.tile(x, reps = dim_y)  # rep(resid.XonZ, times = d_Y)
        #print(left.shape)
        left = left.flatten('F')
        #print(left.shape)
        right = y[:, np.tile(np.arange(dim_y), reps = dim_x)].flatten(order = 'F')   # as.numeric(as.matrix(resid.YonZ)[, rep(seq_len(d_Y), each=d_X)])
        R_mat = np.multiply(left, right)
        R_mat = R_mat.flatten(order = 'F')
        R_mat = np.reshape(R_mat, (nn, dim_x * dim_y), order = 'F')
        R_mat = np.transpose(R_mat)


        norm_con = np.sqrt(np.mean(R_mat ** 2, axis = 1) - np.mean(R_mat, axis = 1)**2)
        norm_con = np.expand_dims(norm_con, axis = 1)
        R_mat = R_mat / norm_con



        ###### TESTING Analytic P-val computation #######
        # res_X = x
        # res_Y = y
        # R = np.multiply(res_X, res_Y)
        # R_sq = R ** 2
        # meanR = np.mean(R)
        # test_stat = np.sqrt(nn) * meanR / np.sqrt(np.mean(R_sq) - meanR ** 2)

        # # print('analytic test stat')
        # # print(test_stat)
        # pval = 2 * (1 - stats.norm.cdf(np.abs(test_stat)))
        # print('analytic pval',pval)


        #############


        # Test statistic
        val = np.max(np.abs(np.mean(R_mat, axis = 1))) * np.sqrt(nn)
        # print('monte carlo test stat')
        # print(val)

        nsim = self.sig_samples

        noise = np.random.randn(nn, nsim)
        test_stat_sim = np.abs(R_mat @ noise)
        test_stat_sim = np.amax(test_stat_sim, axis = 0) / np.sqrt(nn)
        #print('shape test_stat_sim',test_stat_sim.shape )

        pval = (np.sum(test_stat_sim >= val) + 1) / (nsim + 1)

        # print('GMB pval',pval)



        return val,pval#test_stat_sim


    def get_shuffle_significance(self, array, xyz, value,
                                 return_null_dist=False):
        """Returns p-value for shuffle significance test.

        For residual-based test statistics only the residuals are shuffled.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        value : number
            Value of test statistic for unshuffled estimate.

        Returns
        -------
        pval : float
            p-value
        """

        dim, T = array.shape
        dim_x = (xyz==0).sum()
        dim_y = (xyz==1).sum()

        x_vals = self._get_single_residuals(array, xyz, target_var=0)
        y_vals = self._get_single_residuals(array, xyz, target_var=1)

        array_resid = np.vstack((x_vals.reshape(dim_x, T), y_vals.reshape(dim_y, T))) #array
        xyz_resid = np.array([index_code for index_code in xyz if index_code != 2]) #xyz

        if self. correlation_type == 'max_corr' or self. correlation_type == 'gcm' or self. correlation_type == 'linear_hsic_shuffle':

            null_dist = self._get_shuffle_dist(array_resid, xyz_resid,
                                               self.get_dependence_measure,
                                               sig_samples=self.sig_samples,
                                               sig_blocklength=self.sig_blocklength,
                                               verbosity=self.verbosity)

            pval = (null_dist >= np.abs(value)).mean()

            # Adjust p-value for two-sided measures
            if pval < 1.:
                pval *= 2.

        elif self. correlation_type == 'gcm_gmb': # gmb = gaussian_multiplier_bootstrap

            #test_stat, test_stat_sim = self.mult_corr_gcm(array_resid,xyz_resid)
            #nsim = self.sig_samples

            # p value
            #pval = (np.sum(test_stat_sim >= test_stat) + 1) / (nsim + 1)

            # array_resid = np.vstack((x_vals.reshape(dim_x, T), y_vals.reshape(dim_y, T)))
            # xyz_resid = np.array([index_code for index_code in xyz if index_code != 2])

            val, pval = self.mult_corr_gcm(array_resid,xyz_resid)

        elif self.correlation_type == 'linear_hsic_approx' or self.correlation_type == 'linear_hsic_shuffle_kci':


            # array_resid = np.vstack((x_vals.reshape(dim_x, T), y_vals.reshape(dim_y, T)))
            # xyz_resid = np.array([index_code for index_code in xyz if index_code != 2])

            x = array_resid[np.where(xyz==0)[0]]
            y = array_resid[np.where(xyz==1)[0]]
            n = x.shape[1]

            test_stat, Kxc, Kyc = self.hsic_vstat (x,y,n)
            T = Kxc.shape[0]

            if self.correlation_type == 'linear_hsic_approx':


                mean_appr = np.trace(Kxc) * np.trace(Kyc) / T
                var_appr = 2 * np.sum(Kxc ** 2) * np.sum(Kyc ** 2) / T / T # same as np.sum(Kx * Kx.T) ..., here Kx is symmetric
                k_appr = mean_appr ** 2 / var_appr
                theta_appr = var_appr / mean_appr

                pval = 1 - stats.gamma.cdf(test_stat, k_appr, 0, theta_appr)

            elif self.correlation_type == 'linear_hsic_shuffle_kci':

                null_dstr = self.null_sample_spectral(Kxc, Kyc)
                pval = sum(null_dstr.squeeze() > test_stat) / float(self.sig_samples)


        # Adjust p-value for dimensions of x and y (conservative Bonferroni-correction)
        # pval *= dim_x*dim_y

        if return_null_dist:
            return pval, null_dist
        return pval


    def null_sample_spectral(self,Kxc,Kyc):
        ## From causal-learn KCI.py

        T = Kxc.shape[0]
        if T > 1000:
            num_eig = int(np.floor(T / 2))
        else:
            num_eig = T
        lambdax = eigvalsh(Kxc)
        lambday = eigvalsh(Kyc)
        lambdax = -np.sort(-lambdax)
        lambday = -np.sort(-lambday)
        lambdax = lambdax[0:num_eig]
        lambday = lambday[0:num_eig]
        lambda_prod = np.dot(lambdax.reshape(num_eig, 1), lambday.reshape(1, num_eig)).reshape(
            (num_eig ** 2, 1))

        thresh = 1e-5  # Default value in causal-learn
        lambda_prod = lambda_prod[lambda_prod > lambda_prod.max() * thresh]
        f_rand = np.random.chisquare(1, (lambda_prod.shape[0], self.sig_samples))
        null_dstr = lambda_prod.T.dot(f_rand) / T
        return null_dstr

    def get_analytic_significance(self, value, T, dim, xyz):
        """Returns analytic p-value depending on correlation_type.

        Assumes two-sided correlation. If the degrees of freedom are less than
        1, numpy.nan is returned.

        Parameters
        ----------
        value : float
            Test statistic value.

        T : int
            Sample length

        dim : int
            Dimensionality, ie, number of features.

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        Returns
        -------
        pval : float or numpy.nan
            P-value.
        """
        dim_x = (xyz==0).sum()
        dim_y = (xyz==1).sum()

        # Get the number of degrees of freedom
        deg_f = T - dim

        if self.correlation_type == 'max_corr':
            if deg_f < 1:
                pval = np.nan
            elif abs(abs(value) - 1.0) <= sys.float_info.min:
                pval = 0.0
            else:
                trafo_val = value * np.sqrt(deg_f/(1. - value*value))
                # Two sided significance level
                pval = stats.t.sf(np.abs(trafo_val), deg_f) * 2


        #### analytic p-value for gcm exists only when x and y are one dimensional

        elif self.correlation_type == 'gcm':

            #dim, T = array.shape

            # x_vals = self._get_single_residuals(array, xyz, target_var=0)
            # y_vals = self._get_single_residuals(array, xyz, target_var=1)

            # array_resid = np.vstack((x_vals.reshape(dim_x, T), y_vals.reshape(dim_y, T)))
            # res_X = array_resid[np.where(xyz==0)[0]]
            # res_Y = array_resid[np.where(xyz==1)[0]]
            # nn = res_X.shape[1]

            if dim_x == 1 and dim_y == 1:

            #     R = np.multiply(res_X, res_Y)
            #     R_sq = R ** 2
            #     meanR = np.mean(R)
            #     test_stat = np.sqrt(nn) * meanR / np.sqrt(np.mean(R_sq) - meanR ** 2)
                pval = 2 * (1 - stats.norm.cdf(np.abs(value)))
                # print('analytic p val',pval,'\n------')

            else:
                raise NotImplementedError("Analytic p value of GCM implemented for univariate x and y only ")

        else:
            raise NotImplementedError("Currently only"
                                      "correlation_type == 'max_corr','gcm', 'linear_hsic' implemented.")

        # Adjust p-value for dimensions of x and y (conservative Bonferroni-correction)
        pval *= dim_x*dim_y

        return pval

    def get_model_selection_criterion(self, j, parents, tau_max=0, corrected_aic=False):
        """Returns Akaike's Information criterion modulo constants.

        Fits a linear model of the parents to each variable in j and returns
        the average score. Leave-one-out cross-validation is asymptotically
        equivalent to AIC for ordinary linear regression models. Here used to
        determine optimal hyperparameters in PCMCI, in particular the
        pc_alpha value.

        Parameters
        ----------
        j : int
            Index of target variable in data array.

        parents : list
            List of form [(0, -1), (3, -2), ...] containing parents.

        tau_max : int, optional (default: 0)
            Maximum time lag. This may be used to make sure that estimates for
            different lags in X, Z, all have the same sample size.

        Returns:
        score : float
            Model score.
        """

        Y = [(j, 0)]
        X = [(j, 0)]   # dummy variable here
        Z = parents
        array, xyz, _ = self.dataframe.construct_array(X=X, Y=Y, Z=Z,
                                                    tau_max=tau_max,
                                                    mask_type=self.mask_type,
                                                    return_cleaned_xyz=False,
                                                    do_checks=True,
                                                    verbosity=self.verbosity)

        dim, T = array.shape

        y = self._get_single_residuals(array, xyz, target_var=0)

        n_comps = y.shape[0]
        score = 0.
        for y_component in y:
            # Get RSS
            rss = (y_component**2).sum()
            # Number of parameters
            p = dim - 1
            # Get AIC
            if corrected_aic:
                comp_score = T * np.log(rss) + 2. * p + (2.*p**2 + 2.*p)/(T - p - 1)
            else:
                comp_score = T * np.log(rss) + 2. * p
            score += comp_score

        score /= float(n_comps)
        return score


if __name__ == '__main__':

    import tigramite
    from tigramite.data_processing import DataFrame
    from sklearn.linear_model import Ridge, Lasso, RidgeCV, MultiTaskLassoCV, ElasticNetCV, LassoLarsIC, MultiTaskElasticNetCV
    from sklearn.cross_decomposition import PLSRegression
    # import numpy as np
    import timeit

    #### For analytic GCM: X and Y must be univariate:
    # correlation_type = 'gcm'
    # Comment out significance, sig_blocklength and sig_samples

    #### FOR GCM with Gaussian multiplier bootstrap, X and Y can be multivariate
    # correlation_type = 'gcm_gcm'
    # significance = 'shuffle_test',
    # sig_blocklength=1,
    # sig_samples=200



    seed = None
    random_state = np.random.default_rng(seed=seed)
    cmi = ParCorrMult(
            correlation_type = 'gcm_gmb',
            regularization_model = LinearRegression(),            # 0.66
            # regularization_model = RidgeCV(),                      # 0.68
            #regularization_model = MultiTaskLassoCV(),             # 0.722
            # regularization_model = MultiTaskElasticNetCV(),        # 0.77
            # regularization_model = PLSRegression(n_components=2), # 0.95  0.69  0.78  0.64 0.68
            significance = 'shuffle_test',
            sig_blocklength=1,
            sig_samples=200,
            # verbosity=5,
        )

    dimxy = 10
    dimz = 10
    samples = 500
    realizations = 100
    n_conf = dimz
    coef = 0.
    # coef = 0.3

    alpha = 0.05

    rate = np.zeros(realizations)
    pvals = np.zeros(realizations)
    # rate_1 = np.zeros(realizations)

    for i in range(realizations):
        print('realization', i)
        print('============')
        data = random_state.standard_normal((samples, 2*dimxy + dimz))
        confounder = random_state.standard_normal((samples, 1))

        # Only subgroup of size n_conf is internally dependent
        # data[:,int(2*dimxy):int(2*dimxy)+n_conf] += confounder

        # All of Z is internally dependent
        data[:,int(2*dimxy):] += confounder


        # Only subgroup of sizer n_conf is acting as confounder for X and Y
        data[:,0:dimxy] += data[:,int(2*dimxy):int(2*dimxy+n_conf)].mean(axis=1).reshape(samples, 1)
        data[:,dimxy:int(2*dimxy)] += data[:,int(2*dimxy):int(2*dimxy)+n_conf].mean(axis=1).reshape(samples, 1)


        data[:,dimxy:int(2*dimxy)] += coef*data[:,0:dimxy]
        dataframe = DataFrame(data,
            vector_vars={0:[(i,0) for i in range(dimxy)], 1:[(j,0) for j in range(dimxy, 2*dimxy)], 2:[(k, 0) for k in range(2*dimxy, 2*dimxy+dimz)]}
            )

        cmi.set_dataframe(dataframe)

        pvals[i] = cmi.run_test(
                X=[(0,0)],
                Y=[(1,0)], #, (3, 0)],
                # Z=[(5,0)]
                Z = [(2, 0)]
                )[1]

        rate[i] = pvals[i] <= 0.05


    # Bootstrap-based error
    boot_samples = 1000
    boots = np.zeros((boot_samples))
    for b in range(boot_samples):
        rand = np.random.randint(0, realizations, realizations)
        boots[b] = (pvals[rand] <= alpha).mean()
    rate_error = boots.std()

    print(rate.mean(),rate_error)












    ###########################################

        # val_1, pval_1, dep = cmi.run_test(
        #         X=[(0,0)],
        #         Y=[(1,0)], #, (3, 0)],
        #         # Z=[(5,0)]
        #         Z = [(2, 0)],
        #         alpha_or_thres=alpha
        #         )

        # rate[i] = pval <= 0.05
        # rate_1[i] = dep

        # print(pval)
        # cmi.get_model_selection_criterion(j=0, parents=[(1, 0), (2, 0)], tau_max=0, corrected_aic=False)

        # print(cmi.run_test(X=[(0,0),(1,0)], Y=[(2,0), (3, 0)], Z=[(5,0)]))
    # print(rate.mean())
    # print(rate_1.mean())


    ######### GCM : Shah Peters Model ###########

    # # X1 := exp(−Z2/2) · sin(Z) + 0.3 · η1,
    # # X2 := exp(−Z2/2) · sin(Z) + τ · X1 + 0.3 · η2,
    # # Y1 := exp(−Z2/2) · sin(Z) + 0.3 · η3,
    # # Y2 := exp(−Z2/2) · sin(Z) + τ · Y1 + ρ · X2 + 0.3 · η4

    # dimx = 2
    # dimy = 2
    # dimz = 1
    # tau=0.4
    # rho = 0.0 # rho zero is independence

    # for i in range(realizations):

    #     noise = np.random.normal(size = (samples,4))
    #     Z = np.random.normal(size=samples)


    #     X1 = np.exp(-Z * Z / 2) * np.sin(Z) + 0.3 * noise[:,0]
    #     X2 = np.exp(-Z * Z / 2) * np.sin(Z) + tau * X1 + 0.3 * noise[:,1]
    #     Y1 = np.exp(-Z * Z / 2) * np.sin(Z) + 0.3 * noise[:,2]
    #     Y2 = np.exp(-Z * Z / 2) * np.sin(Z) + tau * Y1  + rho * X2 + 0.3 * noise[:,3]


    #     data = np.hstack((X1,X2,Y1,Y2,Z))
    #     data = data.reshape((samples,dimx+dimy+dimz))


    #     dataframe = DataFrame(data,
    #         vector_vars={0:[(i,0) for i in range(dimx)],
    #         1:[(j,0) for j in range(dimx, dimx+dimy)],
    #         2:[(k, 0) for k in range(dimx+dimy, dimx+dimy+dimz)]}
    #         )
    #     cmi.set_dataframe(dataframe)

    #     pval = cmi.run_test(
    #             X=[(0,0)],
    #             Y=[(1,0)], #, (3, 0)],
    #             # Z=[(5,0)]
    #             Z = [(2, 0)]
    #             )[1]

    #     rate[i] = pval <= 0.05

    # print('shah-peters data gen',rate.mean())











