# Final Project:
# Daniel Richford:
# Computational Physics:
# Fall 2017
print("+------------------------------+")
print("|This program is my submission |")
print("|for A. Maller's Computational |")
print("|Physics class given at the GC |")
print("|in Fall 2017.                 |")
print("+------------------------------+")
print("|This program takes a Gaussian |")
print("|distribution and decomposes it|")
print("|into two constituent Gaussian |")
print("|distributions -- taking the   |")
print("|invariant electron yield and  |")
print("|distance-of-closest-approach  |")
print("|distribution recorded during a|")
print("|particle collision, removing  |")
print("|background contributions, and |")
print("|extracting the relative       |")
print("|fraction of charm and bottom  |")
print("|quarks created.               |")
print("+------------------------------+")

# 1. Start accounting for time.
print("| 1. Starting stopwatch!")
import time
start_time = time.time()

# 2. Import statements
print("| 2. Importing packages!")

import os
import emcee
import numpy
import scipy.special
import importlib
import ppg077data
import plotting_functions as pf
from ROOT import TH1, TFile
#from numba import jit


# 3. Definitions
start_definition_time = time.time()
print("| 3. Writing definitions!")
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
    print("| +------------------------------+")
    print("| | Running Parameters           |")
    print("| +------------------------------+")
    print("| | Step Number     : {}".format(step))
    print("| | Bottom Fraction : {}".format(bfrac))
    print("| | alpha           : {}".format(alpha))
    print("| | weight of DCA   : {}".format(w[0]))
    print("| | weight of EpT   : {}".format(w[1]))
    print("| | DCA File Name   : {}".format(dca_filename))
    print("| | Number of")
    print("| |   Random Walkers: {}".format(nwalkers))
    print("| | Steps of")
    print("| |   Burn-in       : {}".format(nburnin))
    print("| | Number of Steps : {}".format(nsteps))
#    print("| | PDF Graph Dir.  : {}".format(pdfdir))
#    print("| | CSV File Dir.   : {}".format(csvdir))
    print("| +------------------------------+")
    print("| Contents:")
    print("|    3.1 Unfold")
    print(" |    3.1.1. eptdata()")
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
    print(" |    3.1.2: histogram_to_array(histogram, option)")
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
    print(" |    3.1.3: dcadata(dca_ept_bin, filename)")
    def dcadata(dca_ept_bin, filename="rootfiles/qm12dca.root"):
        # Column 0 is data, column 1 is background
        f = TFile(filename)
        dcaname = "qm12PPdca{}".format(dca_ept_bin)
        backgroundname = "qm12PPbkg{}".format(dca_ept_bin)
        hdca = f.Get(dcaname)
        hbcg = f.Get(backgroundname)
        return numpy.vstack((histogram_to_array(hdca),histogram_to_array(hbcg))).T

    # 3.1.4: Create Charm and Bottom Matrices for DCA from Files
    print(" |    3.1.4: dcamatrix(dca_ept_bin)")
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
    print(" |    3.1.5: eptmatrix()")
    def eptmatrix():
        c_matrix = numpy.loadtxt("csv/c_to_ept.csv",delimiter=",")
        b_matrix = numpy.loadtxt("csv/b_to_ept.csv",delimiter=",")
        c_hadron_pt = numpy.loadtxt("csv/c_pt.csv",delimiter=",")
        b_hadron_pt = numpy.loadtxt("csv/b_pt.csv",delimiter=",")
        c_matrix /= c_hadron_pt
        b_matrix /= b_hadron_pt
        return numpy.hstack((c_matrix,b_matrix))

    # 3.1.6 Generate Inclusive Hadron pT from Simulation (From Files)
    print(" |    3.1.6: genpt(bfrac)")
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
    print(" |    3.1.7: finite_difference_matrix(ndim)")
    def finite_difference_matrix(number_dimensions):
        d = numpy.ones(number_dimensions)
        a = numpy.diag(d[:-1], k = -1) + numpy.diag(-2 *d) + numpy.diag(d[:-1],k=1)
        a[0,0] = a[-1,-1] = -1
        a *= number_dimensions/2
        return a

    # 3.1.8: Function to Analyze in MCMC, Part 1: 1-D Gaussian Distribution (Spectrum)
    print(" |    3.1.8: ln_gauss(x, mu, inv_cov_mtx)")
    def ln_gauss(x, mu, inv_cov_mtx):
        #print("x",x)
        #print("mu",mu)
        diff = x - mu
        #ln_det_sigma = numpy.sum(numpy.log(1.0/numpy.diag(inv_cov_mtx)))
        #ln_prefactors = -0.5 * (x.shape[0]*numpy.log(2*numpy.pi)+ln_det_sigma)
        #return ln_prefactors - 0.5*numpy.dot(diff, numpy.dot(inv_cov_mtx, diff))
        return -0.5 * numpy.dot(diff, numpy.dot(inv_cov_mtx,diff))

    # 3.1.9: Function to Analyze in MCMC, Part 2: 1-D Poisson Distribution (DCA)
    print(" |    3.1.9: ln_poisson(x, mu -- uses scipy.special.gammaln())")
    def ln_poisson(x, mu):
        if numpy.any(x<0.0) or numpy.any(mu <= 0.0):
            return -numpy.inf
        return numpy.sum(x * numpy.log(mu) - scipy.special.gammaln(x + 1.0) - mu)

    # 3.1.10: Function to Analyzie in MCMC, Part 3: Matrix-Based Gaussian Distribution
    print(" |    3.1.10: multi_variable_gauss(x, matrix_list_entry, data_list_entry)")
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
    print(" |    3.1.11: dca_shape(x, matrix_list, data_list)")
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
    print(" |    3.1.12: reg(x, x_prior, alpha, L)")
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
    print(" |    3.1.11: lppdf_ept_dca(x, matrix_list, data_list, weight_vector, x_prior, alpha, x_lim, L)")
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
    print(" |    3.1.14: Global Parameters for use in Unfold()")
    # if __name__ == "__main__":
    # 3.1.14.1 Bin Information
    # 3.1.14.1.1 Edges
    eptbins = numpy.array([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.5, 5, 6, 7, 8, 9]) # pT for electrons
    cptbins = numpy.array([0., 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 12, 15, 20]) # pT for charm
    bptbins = cptbins # pT for bottom (same as charm)
    hptbins = numpy.hstack((cptbins[:-1], cptbins[-1] + bptbins)) # pT for hadrons # make a matrix using charm and bottom bins (off-set bottom)
    dcabins = numpy.linspace(-0.7, 0.7, 351) # recommendation from colleague
    dcaptbins = numpy.array((1.5, 2.0, 2.5, 3.0, 4.0, 5.0)) # DCA vs pT for electrons

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
    nbpt = len(bptx)
    nhpt = len(hptx)
    nept = len(eptx)
    ndca = len(dcax)
    ndcapt = len(dcaptx)

    # 3.1.14.4 Indices for hadrons with quark flavors
    idx = {"c": numpy.arange(0,ncpt),
           "b": numpy.arange(ncpt, ncpt+ nbpt)}
   # #print("idx =",idx)
    # 3.1.14.4 Misc. Configuration
    number_dimensions = nhpt
    c, b = idx["c"], idx["b"]
    x_ini = x_prior = None # initial and prior probabilities
    function = args = None

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
    ept = eptdata()
    ##print("ept", ept)
    dca = [dcadata(i) for i in range(ndcapt)]

    # 3.1.14.10 Generated Inclusive Hadron pT from Simulation (MCMC initialization point/comparison)
    genpt_full = genpt(bfrac)
    # #print("genpt_full\n",genpt_full)
    # #print("eptmatrix[:,c]",eptmatrix[:,c])
    # #print("genpt_full[:,0][c]",genpt_full[:])
    cept = numpy.dot(eptmatrix[:, c], genpt_full[:,0][c])
    bept = numpy.dot(eptmatrix[:, b], genpt_full[:,0][b])
    hept = cept + bept
    norm_factor = numpy.sum(ept[:, 0]) / numpy.sum(hept)
    genpt_full *= norm_factor
    gpt = genpt_full[:, 0] 

    # MCMC Stuff
    print(" |    3.1.15 MCMC initialization")
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
    print("| | Initializing {} {}-dimensional random walkers.".format(nwalkers,ndim))
    x0[:, c] = x_ini[c] * (1 + 0.1 * numpy.random.randn(nwalkers, ncpt)) #charm
    x0[:, b] = x_ini[b] * (1 + 0.1 * numpy.random.randn(nwalkers, nbpt)) #bottom

    #predictions for animatins later

    # 3.1.15.6:  make a sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, function, args=args)
    pos, prob, state = sampler.run_mcmc(x0, nburnin)
    sampler.reset()
    sampler.run_mcmc(pos, nsteps)
    #check ascceptabnce fraction
    print("mean ac. fr. = {0:.3f}".format(numpy.mean(sampler.acceptance_fraction)))

    # 3.1.15.7: Posterior probability Quantiles
    samples = sampler.chain.reshape((-1,ndim))
    pq = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),zip(*numpy.percentile(samples, [16,50,84], axis=0)))
    print("pq",pq)
    pq = numpy.array(list(pq)) # extra call to list in P3
    print("pq",pq)
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
    print("posterior quantiles[:,1]\n{}".format(pq[:,1]))
    print("posterior quantiles[:,2]\n{}".format(pq[:,2]))
    print("diag(cov):\n{}".format(numpy.sqrt(numpy.diag(cov))))

###
# Results
###

# Refold
    # 3.1.16: Refold
    # 3.1.16.1. Definition
    print(" |    3.1.16.1 ept_refold()")
    def ept_refold(x, eptmatrix):
        c, b = idx["c"], idx["b"]
        cfold = numpy.dot(eptmatrix[:,c],x[c])
        bfold = numpy.dot(eptmatrix[:,b],x[b])
        hfold = cfold + bfold
        bfrac = bfold/hfold
        return cfold, bfold, hfold, bfrac
    ceptr, beptr, heptr, bfrac_ept = ept_refold(pq,eptmatrix)
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
    if step > 1:
        #pf.plot_triangle_bf(bf_samples, pdfdir + "triangle-bf.pdf")
        print("")
    numpy.savetxt("{}pq.csv".format(csvdir), pq, delimiter=",")
    #cov matrices
    # bfraction quantiles
    #use bfraction to avoid decay matrices
    if step > 1:
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
    end_unfold_time = time.time()
#@jit
print("|    3.2 run_unfolding()")
def run_unfolding():
    print("| +------------------------------+")
    print("| | Starting unfolding!")
    for step in range(0,4):
        print("| +------------------------------+")
        print("| | Unfolding Step {}\n".format(step))
        print("| +------------------------------+")
        unfold(step=step, # What step we are on
               alpha=1.00, # Smoothing factor
               bfrac=0.007, # Initial guess of fraction of bottom quarks
               w=[1, 1], #relative weight of DCA and Yield that goes in
               outdir='output',
               dca_filename='fakedata/pp/fakedata-dca.root', #dca data
               ept_data = "fakedata/pp/fakedata-ept.csv", # ept data
               nwalkers=250, #number of random walks
               nburnin=10, #steps not recorded
               nsteps=10) #number of steps

if __name__ == "__main__":
    main_time_start = time.time()
    run_unfolding()
    print("~~ END ~~")
