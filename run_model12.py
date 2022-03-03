'''
Christian Bunker
M^2QM at UF
September 2021

Quasi 1 body transmission through spin impurities project, part 2:
Scattering of single electron off of two localized spin-1/2 impurities
Following cicc, imp spins are confined to single sites, separated by x0
    imp spins can flip
    e-imp interactions treated by (effective) J Se dot Si
    look for resonances in transmission as function of kx0 for fixed E, k
    ie as impurities are pulled further away from each other
    since this is discrete, separate by x0 = N0 a lattice spacings
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
#### entanglement generation (cicc Fig 6)

if False: # compare T vs rhoJa for N not fixed

    # siam inputs
    tl = 1.0;
    Jeff = 0.1;

    # choose boundary condition
    source = np.zeros(8); # incident up, imps = down, down
    source[4] = 1;
    spinstate = "baa"

    # iter over rhoJ, getting T
    Tvals, Rvals = [], [];
    xlims = 0.02, 4.0
    rhoJvals = np.linspace(xlims[0], xlims[1], 99)
    for rhoJa in rhoJvals:

        # energy and K fixed by J, rhoJ
        E_rho = Jeff*Jeff/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                                # this E is measured from bottom of band !!!
        k_rho = np.arccos((E_rho-2*tl)/(-2*tl));
        if(verbose > 5):
            print("E, E - 2t, J, E/J = ",E_rho, E_rho -2*tl, Jeff, E_rho/Jeff);
            print("ka = ",k_rho);
            print("rhoJa = ", (Jeff/np.pi)/np.sqrt(E_rho*tl));
        E_rho = E_rho - 2*tl; # measure from mu
        
        # location of impurities, fixed by kx0 = pi
        kx0 = 1*np.pi;
        N0 = max(1,int(kx0/(k_rho)));
        #assert(N0 == 1);
        print(">>> N0 = ",N0);

        # construct hams
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
        i1, i2 = 1, 1+N0;
        hblocks, tnn = wfm.utils.h_cicc_eff(Jeff, tl, i1, i2, i2+2);
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

        # get T from this setup
        Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E_rho , source));
        Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E_rho , source, reflect = True));

    # save data to .npy
    Tvals, Rvals = np.array(Tvals), np.array(Rvals);
    data = np.zeros((2+2*len(source),len(rhoJvals)));
    data[0,0] = tl;
    data[0,1] = Jeff;
    data[1,:] = rhoJvals;
    data[2:10,:] = Tvals.T;
    data[10:,:] = Rvals.T;
    fname = "data/model12/Nx";
    print("Saving data to "+fname);
    np.save(fname, data);


if True: # compare T vs rhoJa for N=2 fixed

    # siam inputs
    tl = 1.0;
    Jeff = 0.1;

    # choose boundary condition
    source = np.zeros(8); # incident up, imps = down, down
    source[4] = 1;
    spinstate = "baa"

    # iter over rhoJ, getting T
    Tvals, Rvals = [], [];
    xlims = 0.05, 4.0
    rhoJvals = np.linspace(xlims[0], xlims[1], 99)
    for rhoJa in rhoJvals:

        # energy and K fixed by J, rhoJ
        E_rho = Jeff*Jeff/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                                # this E is measured from bottom of band !!!
        k_rho = np.arccos((E_rho-2*tl)/(-2*tl));
        if(verbose > 5):
            print("E, E - 2t, J, E/J = ",E_rho, E_rho -2*tl, Jeff, E_rho/Jeff);
            print("ka = ",k_rho);
            print("rhoJa = ", (Jeff/np.pi)/np.sqrt(E_rho*tl));
        E_rho = E_rho - 2*tl; # measure from mu
        
        # location of impurities, fixed by kx0 = pi
        kx0 = 1*np.pi;
        Vg = E_rho; # gate voltage
        kpa = np.arccos((E_rho-Vg)/(-2*tl));
        N0 = int(np.pi/(kpa));
        print(N0)
        assert(N0 == 1);

        # construct hams
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
        i1, i2 = 1, 1+N0;
        hblocks, tnn = wfm.utils.h_cicc_eff(Jeff, tl, i1, i2, i2+2);
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

        # get T from this setup
        Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E_rho , source));
        Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E_rho , source, reflect = True));

    # save data to .npy
    Tvals, Rvals = np.array(Tvals), np.array(Rvals);
    data = np.zeros((2+2*len(source),len(rhoJvals)));
    data[0,0] = tl;
    data[0,1] = Jeff;
    data[1,:] = rhoJvals;
    data[2:10,:] = Tvals.T;
    data[10:,:] = Rvals.T;
    fname = "data/model12/N2";
    print("Saving data to "+fname);
    np.save(fname, data);


########################################################################
#### plot data

# open command line file
dataf = sys.argv[1];
print("Loading data from "+dataf);
data = np.load(dataf);
tl = data[0,0];
Jeff = data[0,1];
rhoJvals = data[1];
Tvals = data[2:10];
Rvals = data[10:];
print("- shape rhoJvals = ", np.shape(rhoJvals));
print("- shape Tvals = ", np.shape(Tvals));
print("- shape Rvals = ", np.shape(Rvals));

# plot
fig, ax = plt.subplots();
ax.plot(rhoJvals, Tvals[4], label = "$|i\,>$", color = "black", linewidth = 2);
ax.plot(rhoJvals, Tvals[1]+Tvals[2], label = "$|+>$", color = "black", linestyle = "dashed", linewidth = 2);
totals = np.sum(Tvals, axis = 0) + np.sum(Rvals, axis = 0);
ax.plot(rhoJvals, totals, color="red");

# inset
if True:
    rhoEvals = Jeff*Jeff/(rhoJvals*rhoJvals*np.pi*np.pi*tl);
    axins = inset_axes(ax, width="50%", height="50%");
    axins.plot(rhoEvals-2*tl,Tvals[1]+Tvals[2], color = "black", linestyle = "dashed", linewidth = 2); # + state
    axins.set_xlim(-2.05,0);
    axins.set_xticks([-2,0]);
    axins.set_xlabel("$E/t$", fontsize = "x-large");
    axins.set_ylim(0,0.2);
    axins.set_yticks([0,0.2]);
    axins.set_ylabel("$T$", fontsize = "x-large");

# format
ax.set_xlim(min(rhoJvals),max(rhoJvals));
ax.set_xticks([0,1,2,3,4]);
ax.set_xlabel("$J/\pi \sqrt{tE}$", fontsize = "x-large");
ax.set_ylim(0,1.0);
ax.set_yticks([0,1]);
ax.set_ylabel("$T$", fontsize = "x-large");
plt.show();



    
