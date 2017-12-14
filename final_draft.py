# Final Project:
# Daniel Richford:
# Computational Physics:
# Fall 2017
print("+--------------------------------+")
print("| This program is my submission  |")
print("| for A. Maller's Computational  |")
print("| Physics class given at the GC  |")
print("| in Fall 2017.                  |")
print("+--------------------------------+")
print("| This program takes a Gaussian  |")
print("| distribution and decomposes it |")
print("| into two constituent Gaussian  |")
print("| distributions -- taking the    |")
print("| invariant electron yield and   |")
print("| distance-of-closest-approach   |")
print("| distribution recorded during a |")
print("| particle collision, removing   |")
print("| background contributions, and  |")
print("| extracting the relative        |")
print("| fraction of charm and bottom   |")
print("| quarks created.                |")
print("+--------------------------------+")

# 1. Start accounting for time.
print("| 1. Starting stopwatch!")
import time # time
start_time = time.time()

# 2. Import statements
print("| 2. Importing packages!")

import os # path.isdir, makedirs
import emcee # EnsembleSampler
import numpy # diff, vstack, hstack, pi, zeros, prod, loadtxt, sum, ones, diag, dot, any, inf, log, array, linspace, arange, random.randn, mean, percentile, sqrt, abs, histogram2d, argsoft, cumsum, exp, empty, concatenate, savetxt
import scipy.special # gammaln
import matplotlib.pyplot # subplots, close, figure
import importlib # reload
import ppg077data # file of data
from ROOT import TH1, TFile
#from numba import jit
end_import_time = time.time()


# 3. Definitions
start_definition_time = time.time()
print("| 3. Writing definitions!")
print("|    3.1 unfold()")
# 3.1 Unfold
#@jit
def unfold(step,
           alpha,
           bfrac,
           w,
           outdir,
           dca_filename,
           ept_data,
           nwalkers=200,
           nburnin=10,
           nsteps=10):
    start_unfold_time = time.time()
    # 3.1.0 Display running parameters
    print("| | +--------------------------------+")
    print("| | | 4.{}.0 Running Parameters".format(step))
    print("| | + - - - - - - - - - - - - - - - -+")
    print("| | | Step Number     : {}".format(step))
    print("| | | Bottom Fraction : {}".format(bfrac))
    print("| | | alpha           : {}".format(alpha))
    print("| | | weight of DCA   : {}".format(w[0]))
    print("| | | weight of EpT   : {}".format(w[1]))
    print("| | | DCA File Name   : {}".format(dca_filename))
    print("| | | Number of")
    print("| | |   Random Walkers: {}".format(nwalkers))
    print("| | | Steps of")
    print("| | |   Burn-in       : {}".format(nburnin))
    print("| | | Number of Steps : {}".format(nsteps))
#    print("| | | PDF Graph Dir.  : {}".format(pdfdir))
#    print("| | | CSV File Dir.   : {}".format(csvdir))
    print("| | +------------------------------+")
    print("| | | 4.{}.1 Definitions in Unfold():".format(step))
    print("| | |        4.{}.1.1 eptdata()".format(step))
    # 3.1.1: Get Electron Spectrum Data from File (ppg077data.py)
    def eptdata():
        # invariant yield from cross section data (42 mb)
        # multiply by 2*pi*pt*dpt
        d = importlib.reload(ppg077data)
        lo = 0
        for i, x in enumerate(d.eptx):
            if x>=eptx[0]:
                lo = i
                break
        hi = -1
        for i, x in enumerate(d.eptx):
            if x>=eptx[-1]:
                hi = i
                break
        scale = (1.0/42.0)*(numpy.diff(d.eptbins[lo:hi+2])*2.0*numpy.pi*d.eptx[lo:hi+1])
        return numpy.vstack((d.xsec_pp[lo:hi+1] * scale, \
                             d.stat_pp[lo:hi+1] * scale, \
                             d.stat_pp[lo:hi+1] * scale, \
                             d.syst_pp[lo:hi+1] * scale, \
                             d.syst_pp[lo:hi+1] * scale)).T

    # 3.1.2: Turn ROOT Histogram into NumPy Array
    print("| | |        4.{}.1.2: histogram_to_array(histogram, option)".format(step))
    def histogram_to_array(histogram, option=""):
        if not isinstance(histogram, TH1):
            #print("error -- object not a valid root historam")
            return
        dimensions = [histogram.GetNbinsX(), histogram.GetNbinsY(), histogram.GetNbinsZ()]
        n = numpy.prod(dimensions)
        a = numpy.zeros(tuple(dimensions))
        if a.ndim==1:
            for i in range(dimensions[0]):
                a[i] = histogram.GetBinContent(i+1)
        if a.ndim==2:
            for j in range(dimensions[1]):
                for i  in range(dimensions[0]):
                    a[i,j] = histogram.GetBinContent(i+1,j+1)
        if a.ndim==3:
            for k in range(dimensions[2]):
                for j in range(dimensions[1]):
                    for i in range(dimensions[0]):
                        a[i,j,k] = histogram.GetBinContent(i+1,j+1,k+1)
        return a

    # 3.1.3: Get DCA Data from File
    print("| | |        4.{}.1.3: dcadata(dca_ept_bin, filename)".format(step))
    def dcadata(dca_ept_bin, filename="rootfiles/qm12dca.root"):
        # Column 0 is data, column 1 is background
        f = TFile(filename)
        dcaname = "qm12PPdca{}".format(dca_ept_bin)
        backgroundname = "qm12PPbkg{}".format(dca_ept_bin)
        hdca = f.Get(dcaname)
        hbcg = f.Get(backgroundname)
        return numpy.vstack((histogram_to_array(hdca),histogram_to_array(hbcg))).T

    # 3.1.4: Create Charm and Bottom Matrices for DCA from Files
    print("| | |        4.{}.1.4: dcamatrix(dca_ept_bin)".format(step))
    def dcamatrix(dca_ept_bin):
        c_file = "csv/c_to_dca_{}.csv".format(dca_ept_bin)
        b_file = "csv/b_to_dca_{}.csv".format(dca_ept_bin)
        c_matrix = numpy.loadtxt(c_file,delimiter=",")
        b_matrix = numpy.loadtxt(b_file,delimiter=",")
        c_hadron_pt = numpy.loadtxt("csv/c_pt.csv",delimiter=",")
        b_hadron_pt = numpy.loadtxt("csv/b_pt.csv",delimiter=",")
        c_matrix /= c_hadron_pt
        b_matrix /= b_hadron_pt
        return numpy.hstack((c_matrix,b_matrix))

    # 3.1.5: Create Charm and Bottom Matrices for Spectra from Files
    print("| | |        4.{}.1.5: eptmatrix()".format(step))
    def eptmatrix():
        c_matrix = numpy.loadtxt("csv/c_to_ept.csv",delimiter=",")
        b_matrix = numpy.loadtxt("csv/b_to_ept.csv",delimiter=",")
        c_hadron_pt = numpy.loadtxt("csv/c_pt.csv",delimiter=",")
        b_hadron_pt = numpy.loadtxt("csv/b_pt.csv",delimiter=",")
        c_matrix /= c_hadron_pt
        b_matrix /= b_hadron_pt
        return numpy.hstack((c_matrix,b_matrix))

    # 3.1.6 Generate Inclusive Hadron pT from Simulation (From Files)
    print("| | |        4.{}.1.6: genpt(bfrac)".format(step))
    def genpt(bfrac):
        chpt = numpy.loadtxt('csv/c_pt.csv', delimiter=',')
        bhpt = numpy.loadtxt('csv/b_pt.csv', delimiter=',')
        chptsum = numpy.sum(chpt)
        bhptsum = numpy.sum(bhpt)
        chpt *= ((1 - bfrac) * (chptsum + bhptsum)) / chptsum
        bhpt *= (bfrac * (chptsum + bhptsum)) / bhptsum
        genpt = numpy.hstack((chpt, bhpt))
        e = numpy.sqrt(genpt)
        genpt = numpy.vstack((genpt, e, e)).T
        return genpt

    # 3.1.7: Regularization Part 1: Make a Regularization Matrix for later use
    print("| | |        4.{}.1.7: finite_difference_matrix(ndim)".format(step))
    def finite_difference_matrix(number_dimensions):
        d = numpy.ones(number_dimensions)
        a = numpy.diag(d[:-1], k = -1) + numpy.diag(-2 *d) + numpy.diag(d[:-1],k=1)
        a[0,0] = a[-1,-1] = -1
        a *= number_dimensions/2
        return a

    # 3.1.8: Function to Analyze in MCMC, Part 1: 1-D Gaussian Distribution (Spectrum)
    print("| | |        4.{}.1.8: ln_gauss(x, mu, inv_cov_mtx)".format(step))
    def ln_gauss(x, mu, inv_cov_mtx):
        #print("x",x)
        #print("mu",mu)
        diff = x - mu
        #ln_det_sigma = numpy.sum(numpy.log(1.0/numpy.diag(inv_cov_mtx)))
        #ln_prefactors = -0.5 * (x.shape[0]*numpy.log(2*numpy.pi)+ln_det_sigma)
        #return ln_prefactors - 0.5*numpy.dot(diff, numpy.dot(inv_cov_mtx, diff))
        return -0.5 * numpy.dot(diff, numpy.dot(inv_cov_mtx,diff))

    # 3.1.9: Function to Analyze in MCMC, Part 2: 1-D Poisson Distribution (DCA)
    print("| | |        4.{}.1.9: ln_poisson(x, mu -- uses scipy.special.gammaln())".format(step))
    def ln_poisson(x, mu):
        if numpy.any(x<0.0) or numpy.any(mu <= 0.0):
            return -numpy.inf
        return numpy.sum(x * numpy.log(mu) - scipy.special.gammaln(x + 1.0) - mu)

    # 3.1.10: Function to Analyzie in MCMC, Part 3: Matrix-Based Gaussian Distribution
    print("| | |        4.{}.1.10: multi_variable_gauss(x, matrix_list_entry, data_list_entry)".format(step))
    def multi_variable_gauss(x, matrix_list_entry, data_list_entry):
        # X -- Guessed answer
        # matrix_list_entry -- matrix for charm and bottom that maps
        #     x to the predicted result
        # data_list_entry -- data -- column 0 values, column 1 error

        c, b = idx["c"], idx["b"]
        charm_prediction = numpy.dot(matrix_list_entry[:,c], x[c])
        bottom_prediction = numpy.dot(matrix_list_entry[:,b], x[b])
        prediction = charm_prediction + bottom_prediction
        ##print("prediction",prediction)
        inv_cov_data = numpy.diag(1.0 / (data_list_entry[:,1]**2))
        ##print("***\ndata_list_entry[:,0]",data_list_entry[:,0])
        return ln_gauss(data_list_entry[:,0], prediction, inv_cov_data)

    # 3.1.11: Get DCA Shape from the matrices we made (uses poisson)
    print("| | |        4.{}.1.11: dca_shape(x, matrix_list, data_list)".format(step))
    def dca_shape(x, matrix_list, data_list):
        # Poisson
        # X -- guessed answer
        # matrix_list_entry -- matrix for charm and bottom that maps
        #     x to the predicted result
        # data_list_entry -- data -- column 0 signal+bckg, column 1 background
        c, b = idx["c"], idx["b"]
        result = 0.0
        i = 0
        max_n_points = max([d.shape[0] for d in data_list])
        dcashape_prediction = numpy.zeros((len(data_list),max_n_points))
        for matrix_list_entry, data_list_entry in zip(matrix_list, data_list):
            # Make Prediction
            charm_prediction = numpy.dot(matrix_list_entry[:,c],x[c])
            bottom_prediction = numpy.dot(matrix_list_entry[:,b],x[b])
            prediction = charm_prediction + bottom_prediction
            # Scale to match data
            scale = (numpy.sum(data_list_entry[:,0]) - 0.0*numpy.sum(data_list_entry[:,0]))/numpy.sum(prediction) # second index should be [:,1]
    #        prediction = prediction*scale + data_list_entry[:,1]
            prediction = prediction*scale # + data_list_entry[:,1]
            # record result
            result += ln_poisson(data_list_entry[:,0],prediction)
            i +=1
        return result

    # 3.1.12: Function to Analyze in MCMC, Part 4: Regularization
    print("| | |        4.{}.1.12: reg(x, x_prior, alpha, L)".format(step))
    def reg(x, x_prior, alpha, L):
        c, b = idx["c"], idx["b"]
        # before taking L into account
        regularized_charm = x[c]/x_prior[c]
        regularized_bottom = x[b]/x_prior[b]
        # take L into account
        regularized_charm = numpy.dot(L[:,c],regularized_charm)
        regularized_bottom = numpy.dot(L[:,b],regularized_bottom)
        # removed boundary points
        regularized_charm = regularized_charm[1:-1]
        regularized_bottom = regularized_bottom[1:-1]
        # return
        return - alpha*alpha * (numpy.dot(regularized_charm,regularized_charm) + numpy.dot(regularized_bottom,regularized_bottom))

    # 3.1.13: Function to Analyzie in MCMC, Part 5: Putting it all together
    print("| | |        4.{}.1.13: lppdf_ept_dca(x, mtxs, data, wt, prior, alpha, lim, reg-L)".format(step))
    def lppdf_ept_dca(x, # position for MCMC random walker
                      matrix_list, # '''EpT mtx + all the DCA vs pT matrices'''
                      data_list,
                      weight_vector, #'''EpT vs DCA''' 
                      x_prior, # '''prior values of parameters''' 
                      alpha, #'''regularization paramter''' 
                      x_lim,  #''' a constraint''' 
                      L):  #'''a matrix to handle regularization''' 
        # Check that we're within a range
        if numpy.any(x < x_lim[:,0]) or numpy.any(x > x_lim[:,1]):
            return -numpy.inf
        ##print("data_list",data_list, "-- ", data_list[0])
        ##print("Matrix_list", matrix_list," -- ", matrix_list[0])
        lppdf_ept = w[0] * multi_variable_gauss(x, matrix_list[0], data_list[0])
        lppdf_dca = w[1] * dca_shape(x, matrix_list[1:], data_list[1:])
        lppdf_reg = reg(x, x_prior, alpha, L)
        return lppdf_ept + lppdf_dca + lppdf_reg

    # 3.1.14: Global Parameters to use in Function 3.1 Unfold()
    print("| | | 4.{}.2 Global Parameters for use in Unfold()".format(step))
    # if __name__ == "__main__":
    # 3.1.14.1 Bin Information
    # 3.1.14.1.1 Edges
    eptbins = numpy.array([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.5, 5, 6, 7, 8, 9]) # pT for electrons
    #print("len(eptbins) ={}".format(len(eptbins)))
    cptbins = numpy.array([0., 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 12, 15, 20]) # pT for charm
    #print("len(cptbins) = {}".format(len(cptbins)))
    bptbins = cptbins # pT for bottom (same as charm)
    hptbins = numpy.hstack((cptbins[:-1], cptbins[-1] + bptbins)) # pT for hadrons # make a matrix using charm and bottom bins (off-set bottom)

    #eptbins = numpy.linspace(0.1,10.0,22)
    #cptbins = numpy.linspace(0.1,10.0,18)
    #bptbins = cptbins
    #hptbins = numpy.hstack((cptbins[:-1], cptbins[-1]+bptbins))

    dcabins = numpy.linspace(-0.7, 0.7, 351) # recommendation from colleague
    dcaptbins = numpy.array((1.5, 2.0, 2.5, 3.0, 4.0, 5.0)) # DCA vs pT for electrons
    #dcaptbins = numpy.linspace(1.0,5.0,6)
    # 3.1.14.1.2 Widths
    eptw = numpy.diff(eptbins) # difference from one bin to the next
    cptw = numpy.diff(cptbins)
    bptw = numpy.diff(bptbins)
    hptw = numpy.diff(hptbins)
    dcaw = numpy.diff(dcabins)
    dcaptw = numpy.diff(dcaptbins)
    # 3.1.14.1.3 Centers
    eptx = eptbins[:-1] + eptw / 2 # from the edge to half the width
    cptx = cptbins[:-1] + cptw / 2
    bptx = bptbins[:-1] + bptw / 2
    hptx = hptbins[:-1] + hptw / 2
    dcax = dcabins[:-1] + dcaw / 2
    dcaptx = dcaptbins[:-1] + dcaptw / 2
    # 3.1.14.2 Dimensional Information
    ncpt = len(cptx)
    #print("ncpt = len(cptx)",ncpt)
    nbpt = len(bptx)
    nhpt = len(hptx)
    nept = len(eptx)
    ndca = len(dcax)
    ndcapt = len(dcaptx)

    # 3.1.14.4 Indices for hadrons with quark flavors
    idx = {"c": numpy.arange(0,ncpt),
           "b": numpy.arange(ncpt, ncpt+ nbpt)}
    #print("idx =",idx)
    # 3.1.14.4 Misc. Configuration
    number_dimensions = nhpt
    c, b = idx["c"], idx["b"]
    x_ini = x_prior = None # initial and prior probabilities
    function = args = None

    print("| | | 4.{}.3 Output Directories".format(step))
    # 3.1.14.5 Output
    pdfdir = "{}/pdfs/{}/".format(outdir, step + 1) # create a subdirectory for the graphs from each step
    if (not os.path.isdir(pdfdir)): # if there is no directory, make it
        os.makedirs(pdfdir)
    csvdir = "{}/csv/{}/".format(outdir, step + 1) # create a subdirectory for the data/histograms generated at each step for later use
    if (not os.path.isdir(csvdir)): # if there is no directory, make it
        os.makedirs(csvdir)

    # 3.1.14.6 Matrices
    eptmatrix = eptmatrix()
  #  #print("eptmatrix",eptmatrix) 
    dcamatrix = [dcamatrix(i) for i in range(ndcapt)]

    # 3.1.14.7Some data to test
    # if step > 0:
    #     bfrac_file = outdir + "/csv/{}/bfrac.csv".format(step)
    #     bfrac = numpy.loadtxt(bfrac_file,delimiter=",")
    ept = eptdata()
    ##print("ept", ept)
    dca = [dcadata(i) for i in range(ndcapt)]

    # 3.1.14.10 Generated Inclusive Hadron pT from Simulation (MCMC initialization point/comparison)
    genpt_full = genpt(bfrac)
    # print("genpt_full\n",genpt_full)
    # print("c",c)
    # print("eptmatrix[:,c]",eptmatrix[:,c])
    # print("genpt_full[:,0][c]",genpt_full[:])
    cept = numpy.dot(eptmatrix[:, c], genpt_full[:,0][c])
    bept = numpy.dot(eptmatrix[:, b], genpt_full[:,0][b])
    hept = cept + bept
    norm_factor = numpy.sum(ept[:, 0]) / numpy.sum(hept)
    genpt_full *= norm_factor
    gpt = genpt_full[:, 0]

    # MCMC Stuff
    print("| | | 4.{}.4 Metropolis-Coupled Markov-Chain Monte Carlo".format(step))
    print("| | |        4.{}.4.1 MCMC initialization".format(step))
    # 3.1.15.1 Set Initialization for Markov Chains
    #print("|    | Markov Chain Initialization")
    if step == 0:
        # first step
        x_ini = gpt
    else:
        # subsequent steps
        csvi = "{}/csv/{}/pq.csv".format(outdir, step)
        x_ini = numpy.loadtxt(csvi, delimiter=',')[:,0]
    #print("|    | - x_ini: {}",format(x_ini))

    # 3.1.15.2 Set Prior Prob. Vector
    x_prior = gpt # from simulation
    #print("|     | - x_prior: {}".format(x_prior))

    # 3.1.15.3 Create List of Combined Data
    data_list = [ept]
    ##print("data_list",data_list)
    [data_list.append(d) for d in dca]
    ##print("data_list",data_list)
    matrix_list = [eptmatrix]
    [matrix_list.append(m) for m in dcamatrix]
    #print("|     +------------------------------+")

    # 3.1.15.4 Specify Parameters
    weight_vector = (eptw, dcaw)
    x_lim = numpy.vstack((0.001*x_ini, 10.0*x_ini)).T
    L = numpy.hstack((finite_difference_matrix(ncpt),finite_difference_matrix(nbpt)))
    args = [matrix_list, data_list, weight_vector, x_prior, alpha, x_lim, L]
    function = lppdf_ept_dca

    #3.1.15.5 MCMC Ensemble initialization
    ndim = nhpt
    x0 = numpy.zeros((nwalkers,ndim))
    print("| | |        4.{}.4.2 Initializing {} {}-dimensional random walkers.".format(step,nwalkers,ndim))
    x0[:, c] = x_ini[c] * (1 + 0.1 * numpy.random.randn(nwalkers, ncpt)) #charm
    x0[:, b] = x_ini[b] * (1 + 0.1 * numpy.random.randn(nwalkers, nbpt)) #bottom
    if step < 1:
        print("Error messages about invalid value in subtract() and greater().")

    # 3.1.15.6:  make a sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, function, args=args)
    pos, prob, state = sampler.run_mcmc(x0, nburnin)
    sampler.reset()
    sampler.run_mcmc(pos, nsteps)
    #check ascceptabnce fraction
    print("| | |        --> Result: Mean Acceptance Fraction = {0:.3f}".format(numpy.mean(sampler.acceptance_fraction)))

    # 3.1.15.7: Posterior probability Quantiles
    samples = sampler.chain.reshape((-1,ndim))
    pq = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),zip(*numpy.percentile(samples, [16,50,84], axis=0)))
    #print("pq",pq)
    pq = numpy.array(list(pq)) # extra call to list in P3
    #print("pq",pq)
    #Covariance Matrix
    cov = numpy.zeros([ndim,ndim])
    N = samples.shape[0]
    for m in range(ndim):
        for n in range(ndim):
            # print("cov[m,n]:",cov[m,n])
            # print("samples[:,m]",samples[:,m])
            # print("pq[m,0]",pq[m,0])
            # print("samples[:,n]",samples[:,n])
            # print("pq[n,0]",pq[n,0])
            cov[m,n] = (1.0/(N-1))*numpy.sum((samples[:,m] - pq[m,0])*(samples[:,n] - pq[n,0]))
    # 3.1.15.8: Compare quantiles and diagonas
    # print("| | |  RESULT: posterior quantiles[:,1]\n{}".format(pq[:,1]))
    # print("| | |  RESULT: posterior quantiles[:,2]\n{}".format(pq[:,2]))
    # print("| | |  RESULT: diag(cov):\n{}".format(numpy.sqrt(numpy.diag(cov))))

###
# Results
###

# Refold
    # 3.1.16: Refold
    # 3.1.16.1. Definition
    print("| | | 4.{}.5 Refold".format(step))
    print("| | |        4.{}.5.1 ept_refold()".format(step))
    def ept_refold(x, eptmatrix):
        c, b = idx["c"], idx["b"]
        cfold = numpy.dot(eptmatrix[:,c],x[c])
        #print("cfold = {}".format(cfold))
        bfold = numpy.dot(eptmatrix[:,b],x[b])
        #print("bfold = {}".format(bfold))
        hfold = cfold + bfold
        #print("hfold = {}".format(hfold))
        bfrac = bfold / hfold
        #print("bfrac = {}".format(bfrac))
        return cfold, bfold, hfold, bfrac
    ceptr, beptr, heptr, bfrac_ept = ept_refold(pq,eptmatrix)
    if step < 1:
        print("Error message about trouble with true-divide.") # becaues columns 2 and 3 of c,b,hfold are all 0.
    # 3.1.16.2 Jacobian
    J = numpy.zeros([nept,ndim])
    B = numpy.hstack([numpy.zeros([nept,ncpt]), eptmatrix[:,b]])
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            J[i,j] = -1 * eptmatrix[i,j]*numpy.sum(numpy.dot(B[i,:],pq[:,0]))/(numpy.sum(numpy.dot(eptmatrix[i,:],pq[:,0])))**2
            J[i,j] += B[i,j]/numpy.sum(numpy.dot(eptmatrix[i,:],pq[:,0]))
    # 3.1.16.3 Covariance matrices
    cov_bf = numpy.dot(J, numpy.dot(cov, J.T))
    bfrac_cov = numpy.vstack((bfrac_ept[:,0],numpy.array(numpy.sqrt(numpy.diag(cov_bf))),numpy.array(numpy.sqrt(numpy.diag(cov_bf))))).T


    # 3.1.16.4 percentiles
    bf_samples = numpy.zeros([samples.shape[0], nept])
    for i in range(samples.shape[0]):
        samp = numpy.vstack((samples[i,:],numpy.zeros(nhpt),numpy.zeros(nhpt))).T
        ce, be, he, bf = ept_refold(samp,eptmatrix)
        bf_samples[i] = bf[:,0] # just taking the middle value
    bf_pq = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*numpy.percentile(bf_samples, [16,50,84], axis=0)))
    bf_pq = numpy.array(list(bf_pq))

    # 3.1.17 Save various things for next step
    #if step == 0:
    if step > 2:
        colors = {'c':'forestgreen', \
                  'b':'dodgerblue', \
                  'bkg':'saddlebrown', \
                  'tot':'crimson', \
                  'bf':'crimson'}
        print("| | | 4.{}.6 Charm-Bottom Component Plot!".format(step))
        def plot_result_cb(pq, figname='hpt-cb.pdf'):
            # Plot charm and bottom hadron pT distributions
            fig, ax = matplotlib.pyplot.subplots(figsize=(6, 7))
            ax.set_yscale('log')
            ax.set_xlabel(r'$p_T$ (GeV/c)')
            ax.set_ylabel(r'hadron yield (arb. u.)')
            c, b = idx['c'], idx['b']
            ax.errorbar(cptx, \
                        pq[c, 0] / hptw[c], \
                        yerr=[pq[c, 2] / hptw[c], \
                              pq[c, 1] / hptw[c]], \
                        linestyle='None', \
                        fmt='o', \
                        color=colors['c'], \
                        ecolor=colors['c'], \
                        label=r'$c$ hadrons', \
                        capthick=2)
            ax.errorbar(bptx, \
                        pq[b, 0] / hptw[b], \
                        yerr=[pq[b, 2] / hptw[b], \
                              pq[b, 1] / hptw[b]], \
                        linestyle='None', \
                        fmt='o', \
                        color=colors['b'], \
                        ecolor=colors['b'], \
                        label=r'$b$ hadrons', \
                        capthick=2)
            ax.legend()
            fig.savefig(figname)
            matplotlib.pyplot.close(fig)
            return
        plot_result_cb(pq, pdfdir + "ht-cb.pdf")
#        print("| | | 4.{}.7 DCA Result".format(step))
        print("| | | 4.{}.7 Big Hadron Triangle Plot!".format(step))
        
        print("| | | 4.{}.8 Electron Triangle Plot!".format(step))
        def plot_triangle_bf(d, figname='posterior-triangle-bf.pdf', plotcor = True):
            K = d.shape[1]
            # factor = 2.0           # size of one side of one panel
            factor = 1.0           # size of one side of one panel
            lbdim = 2.5 * factor   # size of left/bottom margin
            trdim = 0.2 * factor   # size of top/right margin
            whspace = 0.05         # w/h space size
            plotdim = factor * K + factor * (K - 1.) * whspace
            dim = lbdim + plotdim + trdim    
            frac = factor / dim

            # index for zoomed plot
            # zidx = 8

            # make the figure
            fig = matplotlib.pyplot.figure(figsize=(dim, dim))

            # full figure axes
            ax_full = fig.add_axes([0, 0, 1, 1], )
            ax_full.axis('off')

            # Axis labels
            fig.text(0.50, 0.01, r'$p_T^e$', fontsize=90, color=colors['bf'])
            fig.text(0.005, 0.50, r'$p_T^e$', fontsize=90, color=colors['bf'])

            ax_full.arrow(0.05, (1.0 - trdim/dim),
                          0, -1*(1.0 - trdim/dim - lbdim/dim),
                          length_includes_head=True,
                          shape='full',
                          fc=colors['bf'], ec=colors['bf'],
                          # head_width=20, head_length=5, linewidth=10,
                          transform=ax_full.transAxes)

            ax_full.arrow(lbdim/dim, 0.05,
                          1*(1.0 - trdim/dim - lbdim/dim), 0,
                          length_includes_head=True,
                          shape='full',
                          fc=colors['bf'], ec=colors['bf'],
                          # head_width=20, head_length=5, linewidth=10,
                          transform=ax_full.transAxes)

            # Individual correlations and marginals
            axes = []
            for i in range(K):
                for j in range(K):
                    # Calcualte the bottom left corner of the desired axis
                    # Plots start in the top left
                    x1 = lbdim/dim + i * (frac + whspace/dim)
                    y1 = 1.0 - trdim/dim - (frac + whspace/dim) * (j + 1)
                    # print("({}, {}): [{}, {}, {}, {}]".format(i, j, x1, y1, 
                    #       x1+frac, y1+frac))

                    # Plot the Pearson correlation coefficient if desired
                    # Else move on
                    if i > j:
                        if plotcor:
                            ax = fig.add_axes([x1, y1, frac, frac])
                            ax.set_xticklabels([])
                            ax.set_yticklabels([])

                            mi = numpy.mean(d[:, i])
                            mj = numpy.mean(d[:, j])
                            N = d.shape[0]

                            num = numpy.sum(d[:, i] * d[:, j]) - N * mi * mj

                            denomi = numpy.sum(d[:, i]*d[:, i]) - N * mi**2

                            denomj = numpy.sum(d[:, j]*d[:, j]) - N * mj**2

                            denom = numpy.sqrt(denomi) * numpy.sqrt(denomj)

                            p = num / denom

                            ax.text(0.5, 0.5, '{:.2f}'.format(p),
                                    fontsize=numpy.abs(p) * 30,
                                    color='gray',
                                    ha='center', va='center')

                        continue

                    ax = fig.add_axes([x1, y1, frac, frac])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])

                    c = 'gray'

                    if i == j:
                        ax.hist(d[:, i], 100, 
                                color='k', fc=colors['bf'], histtype='stepfilled')
                    else:
                        rg = [[d[:, i].min(), d[:, i].max()], 
                                 [d[:, j].min(), d[:, j].max()]]

                        H, X, Y = numpy.histogram2d(d[:, i], d[:, j], bins=20,
                                                 range=rg)

                        # compute the bin centers
                        X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

                        # Compute the density levels.
                        alevels = 1.0 - numpy.exp(-0.5 * numpy.arange(0.5, 2.1, 0.5) ** 2)
                        Hflat = H.flatten()
                        inds = numpy.argsort(Hflat)[::-1]
                        Hflat = Hflat[inds]
                        sm = numpy.cumsum(Hflat)
                        sm /= sm[-1]
                        V = numpy.empty(len(alevels))
                        for k, v0 in enumerate(alevels):
                            try:
                                V[k] = Hflat[sm <= v0][-1]
                            except:
                                V[k] = Hflat[0]

                        H2 = H.min() + numpy.zeros((H.shape[0] + 4, H.shape[1] + 4))
                        H2[2:-2, 2:-2] = H
                        H2[2:-2, 1] = H[:, 0]
                        H2[2:-2, -2] = H[:, -1]
                        H2[1, 2:-2] = H[0]
                        H2[-2, 2:-2] = H[-1]
                        H2[1, 1] = H[0, 0]
                        H2[1, -2] = H[0, -1]
                        H2[-2, 1] = H[-1, 0]
                        H2[-2, -2] = H[-1, -1]
                        X2 = numpy.concatenate([
                            X1[0] + numpy.array([-2, -1]) * numpy.diff(X1[:2]),
                            X1,
                            X1[-1] + numpy.array([1, 2]) * numpy.diff(X1[-2:]),
                        ])
                        Y2 = numpy.concatenate([
                            Y1[0] + numpy.array([-2, -1]) * numpy.diff(Y1[:2]),
                            Y1,
                            Y1[-1] + numpy.array([1, 2]) * numpy.diff(Y1[-2:]),
                        ])
                        # Filled contour
                        levels = [0.0,0.1,0.2,0.5,0.75,1.0,numpy.inf]
                        if len(levels) > 1 and numpy.min(numpy.diff(levels)) <= 0.0:
                            if hasattr(self, '_corner_mask') and self._corner_mask == 'legacy':
                                warnings.warn("Contour levels are not increasing")
                            else:
                                raise ValueError("Contour levels must be increasing")
                        # contourf_kwargs = dict()
                        # contourf_kwargs["levels"] = contourf_kwargs.get("levels",levels)
                        # contourf_kwargs["colors"] = contourf_kwargs.get("colors", c)
                        # contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased", False)
                        # ax.contourf(X2, Y2, H2.T, numpy.concatenate([[H.max()], V, [0]]),**contourf_kwargs)
                        # ax.contourf(X2, Y2, H2.T, numpy.concatenate([[H.max()], V, [0]]))
                        # #Contour lines
                        # contour_kwargs = dict()
                        # contour_kwargs["levels"] = contour_kwargs.get("levels", levels)
                        # contour_kwargs["colors"] = contour_kwargs.get("colors", 'k')
                        # ax.contour(X2, Y2, H2.T, V, **contour_kwargs)
            fig.savefig(figname, transparent=True)
        plot_triangle_bf(bf_samples, pdfdir + "triangle-bf.pdf")
        print("| | | 4.{}.8 Saving Data as Intial Conditions for Step {}".format(step,step + 1))
    else:
        print("| | | 4.{}.6 Saving Data".format(step))
    numpy.savetxt("{}pq.csv".format(csvdir), pq, delimiter=",")
    #cov matrices
    # bfraction quantiles
    #use bfraction to avoid decay matrices
    # print("array from eptx", numpy.array([eptx]))
    # print("bf_pq", bf_pq)
    # print("T eptx array", numpy.array([eptx]).T)
    bfpq_save = numpy.hstack((numpy.array([eptx]).T, bf_pq))
    numpy.savetxt("{}bf-pq.csv".format(csvdir), bfpq_save, delimiter=",")

    bfcov_save = numpy.hstack((numpy.array([eptx]).T, bfrac_cov))
    numpy.savetxt("{}bf-cov.csv".format(csvdir), bfcov_save, delimiter=",")

    bfrac_save = numpy.hstack((numpy.array([eptx]).T, bfrac_ept))
    numpy.savetxt("{}bfrac.csv".format(csvdir), bfrac_save, delimiter=",")
    # initialization vectors for use in subsequents steps
    numpy.savetxt("{}prior.csv".format(csvdir),
               numpy.vstack((numpy.hstack([cptx, bptx]), x_prior)).T,
               delimiter=",")
    numpy.savetxt("{}ini.csv".format(csvdir),
               numpy.vstack((numpy.hstack([cptx, bptx]), x_prior)).T,
               delimiter=",")
    print("| |/")
    print("| /")
    print("|/")
    end_unfold_time = time.time()
    unfold_time.append(end_unfold_time-start_unfold_time)

#@jit
print("|    3.2 run_unfolding()")
def run_unfolding(stepss):
    print("| +------------------------------+")
    print("| | 4. Starting unfolding!")
    for step in range(0,stepss):
        print("| +------------------------------+")
        print("| | Unfolding Step {}".format(step))
        unfold(step=step, # What step we are on
               alpha=1.00, # Smoothing factor
               bfrac=0.007, # Initial guess of fraction of bottom quarks
               # must change this to use the last b-fraction
               w=[1, 1], #relative weight of DCA and Yield that goes in
               outdir='output',
               dca_filename='fakedata/pp/fakedata-dca.root', #dca data
               ept_data = "fakedata/pp/fakedata-ept.csv", # ept data
               nwalkers=500, #number of random walks
               nburnin=100, #steps not recorded
               nsteps=100) #number of steps
end_definition_time = time.time()

if __name__ == "__main__":
    main_time_start = time.time()
    stepss = 2
    unfold_time = []
    run_unfolding(stepss)
    main_time_end = time.time()
    print("+------------------------------+")
    print("| 5.{} Timing information")
    print("+ - - - - - - - - - - - - - - -+")
    print("| Unfolding Steps:")
    for time in unfold_time:
        print("|    {:.2f} s".format(time))
    print("| Unfolding System: {:.2f} s".format(main_time_end-main_time_start))
    print("| Defintions: {:.2f} µs".format((end_definition_time-start_definition_time)*1e6))
    print("| Imports: {:.2f} s".format(end_import_time-start_time))
    print("| TOTAL: {:.2f} min".format(((end_import_time-start_time)+(end_definition_time-start_definition_time)+(main_time_end-main_time_start))/60))
    print("+------------------------------+")
    print("~~~~~~~~~~~~~ END ~~~~~~~~~~~~~")
