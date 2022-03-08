'''
Christian Bunker
M^2QM at UF
September 2021

Quasi 1 body transmission through spin impurities project, part 5:
Scattering of single electron off of two localized spin-1/2 impurities
with mirror. N12 = dist btwn imps, N23 = dist btwn imp 2, mirror are
tunable in order to realize CNOT
'''

from code import wfm
from code.wfm import utils

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5

##################################################################################
#### tune N12 and N23
    
if False: 

    # choose boundary condition
    source = np.zeros(8);
    source[6] = 1;

    # tight binding params
    tl = 1.0; # hopping everywhere
    Jeff = 0.1; # exchange strength
    Vb = 5.0; # barrier in RL (i.e. perfect mirror)

    # fix kpa0 at 0
    N12 = 2;
    factor = 99; # 99, 198, 988
    ka0 =  np.pi/(N12 - 1)/factor; # free space wavevector, should be << pi
                                    # increasing just broadens the Vg peak
    kpa0 = np.pi/(N12 - 1)/factor; # wavevector in gated SR
    E_rho = 2*tl-2*tl*np.cos(ka0); # energy of ka0 wavevector, which determines rhoJa
                                    # measured from bottom of the band!!
    rhoJa = Jeff/(np.pi*np.sqrt(tl*E_rho));
    myVg = 0 # -2*tl*np.cos(ka0) +2*tl;
    kpa = np.arccos((-2*tl*np.cos(ka0)-myVg)/(-2*tl));
    print(">>> rhoJa, myVg, k'a/pi = ",rhoJa, myVg, kpa/np.pi);
    
    # mesh of x23s (= N23*a)
    kx23lims = (0,np.pi);
    #kx23vals = np.linspace(*kx23lims, 99);
    NBmax = int(kx23lims[-1]/ka0);
    NBvals = np.linspace(1,NBmax,99, dtype = int);
    kx23vals = ka0*NBvals;
    Vg= 0;

    # iter over impurity-mirror distances, get transmission
    Rvals = [];
    for NB in NBvals:

        # construct hams
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks also
        hblocks, tnn = wfm.utils.h_cicc_eff(Jeff, tl, 1, N12, 1+N12+NB);
        tnnn = np.zeros_like(tnn)[:-1];

        # add gate voltages
        for blocki in range(len(hblocks)):

            # in SR
            if(blocki > 0 and blocki < N12 + 1): 
                hblocks[blocki] += myVg*np.eye(np.shape(hblocks[0])[0]);

            # btwn SR and barrier
            if(blocki > N12):
                hblocks[blocki] += Vg*np.eye(np.shape(hblocks[0])[0]);
        
        # add barrier to RL
        hblocks[-1] += Vb*np.eye(len(source));
        if(verbose and NB == NBvals[0]): print(hblocks);

        # get T from this setup
        Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, -2*tl*np.cos(ka0), source, reflect = True));

    # package into one array
    Rvals = np.array(Rvals);
    if(verbose): print("shape(Tvals) = ",np.shape(Rvals));   
    info = np.zeros_like(kx23vals);
    info[0], info[1], info[2], info[3] = tl, Jeff, rhoJa, ka0; # save info we need
    data = [info, kx23vals];
    for sigmai in range(len(source)): # 
        data.append(Rvals[:,sigmai]); # data has columns of N0val, k0val, corresponding T vals
    # save data
    fname = "data/gate/";
    fname +="N_rhoJa"+str(int(np.around(rhoJa)))+".npy";
    np.save(fname,np.array(data));
    if verbose: print("Saved data to "+fname);


# plot each file given at command line   
fig, axes = plt.subplots();
axes = [axes];
datafs = sys.argv[1:];
for fi in range(len(datafs)):

    # unpack
    print("Loading data from "+datafs[fi]);
    data = np.load(datafs[fi]);
    tl, Jeff, rhoJa, k_rho = data[0,0], data[0,1], data[0,2], data[0,3];
    kNavals = data[1];
    Rvals = data[2:].T;

    # plot by channel
    print(">>>",np.shape(Rvals));
    styles = ["solid","dashed","dashdot"];
    sigmas = [1,2,4]
    for sigmai in range(len(sigmas)):
        axes[0].plot(kNavals/np.pi, Rvals[:,sigmas[sigmai]], color = "black", linestyle = styles[sigmai], linewidth = 2);

    # totals
    totals = np.sum(Rvals, axis = 1);
    axes[0].plot(kNavals/np.pi, totals, color = "red");

# format and show
axes[0].set_xlim(0.0,1.0);
axes[0].set_xticks([0,1]);
axes[0].set_xlabel("$kaN_{B}/\pi$", fontsize = "x-large");
axes[0].set_ylim(0.0,1);
axes[0].set_yticks([0,1]);
axes[0].set_ylabel("$R$", fontsize = "x-large");
plt.show();






