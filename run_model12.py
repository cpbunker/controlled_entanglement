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


if False: # compare T vs rhoJa for N=2 fixed

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
    rhoJavals = np.linspace(xlims[0], xlims[1], 99)
    for rhoJa in rhoJavals:

        # energy and K fixed by J, rhoJ
        E_rho = Jeff*Jeff/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                                # this E is measured from bottom of band !!!
        Energy = E_rho - 2*tl; # regular energy
        
        # optical distances, N = 2 fixed
        ka = np.arccos((Energy)/(-2*tl));
        Vg = Energy + 2*tl; # gate voltage
        kpa = np.arccos((Energy-Vg)/(-2*tl));
        N0 = 1;
        print(ka, kpa, Vg);

        # construct hams
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
        i1, i2 = 1, 1+N0;
        hblocks, tnn = wfm.utils.h_cicc_eff(Jeff, tl, i1, i2, i2+2);
        hblocks[1] += Vg*np.eye(len(source)); # Vg shift in SR
        hblocks[2] += Vg*np.eye(len(source));
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping
        if(verbose > 3 and rhoJa == rhoJavals[0]): print(hblocks);

        # get T from this setup
        Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source));
        Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source, reflect = True));

    # save data to .npy
    Tvals, Rvals = np.array(Tvals), np.array(Rvals);
    data = np.zeros((2+2*len(source),len(rhoJavals)));
    data[0,0] = tl;
    data[0,1] = Jeff;
    data[1,:] = rhoJavals;
    data[2:10,:] = Tvals.T;
    data[10:,:] = Rvals.T;
    fname = "data/model12/N2";
    print("Saving data to "+fname);
    np.save(fname, data);


########################################################################
#### plot data

# open command line file
datafs = sys.argv[1:];
fig, axes = plt.subplots(len(datafs), sharex = True);
if( len(datafs)== 1): axes = [axes];
for fi in range(len(datafs)):
    dataf = datafs[fi];
    print("Loading data from "+dataf);
    data = np.load(dataf);
    tl = data[0,0];
    Jeff = data[0,1];
    rhoJavals = data[1];
    Evals = Jeff*Jeff/(rhoJavals*rhoJavals*np.pi*np.pi*tl)-2*tl;
    Tvals = data[2:10];
    Rvals = data[10:];
    print("- shape rhoJvals = ", np.shape(rhoJavals));
    print("- shape Tvals = ", np.shape(Tvals));
    print("- shape Rvals = ", np.shape(Rvals));

    # plot
    axes[fi].plot(rhoJavals, Tvals[4], label = "$|i\,>$", color = "black", linewidth = 2);
    axes[fi].plot(rhoJavals, Tvals[1]+Tvals[2], label = "$|+>$", color = "black", linestyle = "dashed", linewidth = 2);
    totals = np.sum(Tvals, axis = 0) + np.sum(Rvals, axis = 0);
    axes[fi].plot(rhoJavals, totals, color="red");
    axes[fi].set_ylim(0,1.0);
    axes[fi].set_yticks([0,1]);
    axes[fi].set_ylabel("$T$", fontsize = "x-large");

    # inset
    if False:
        axins = inset_axes(ax, width="50%", height="50%");
        axins.plot(Evals,Tvals[1]+Tvals[2], color = "black", linestyle = "dashed", linewidth = 2); # + state
        axins.set_xlim(min(Evals)-0.01,max(Evals));
        axins.set_xticks([-2.0,-1.6]);
        axins.set_xlabel("$E/t$", fontsize = "x-large");
        axins.set_ylim(0,0.2);
        axins.set_yticks([0,0.2]);
        axins.set_ylabel("$T$", fontsize = "x-large");

# format
axes[fi].set_xlim(min(rhoJavals),max(rhoJavals));
axes[fi].set_xticks([0,1,2,3,4]);
axes[fi].set_xlabel("$J/\pi \sqrt{t(E+2t)}$", fontsize = "x-large");
plt.show();

if False: # plot E, 1/E two separate axes
    
    # plot vs E
    fig, axes = plt.subplots(2)
    axes[0].plot(Evals, Tvals[1]+Tvals[2], label = "$|+>$", color = "black", linestyle = "solid", linewidth = 2);
    totals = np.sum(Tvals, axis = 0) + np.sum(Rvals, axis = 0);
    axes[0].plot(Evals, totals, color="red");
    axes[0].set_xlim(-2.0,-1.6);
    axes[0].set_xticks([-2,-1.8,-1.6]);
    axes[0].set_xlabel("$E/t$", fontsize = "x-large");
    axes[0].set_ylim(0,0.25);
    axes[0].set_yticks([0,0.25]);
    axes[0].set_ylabel("$T_+$", fontsize = "x-large");

    # plot vs 1/E
    axes[1].plot(rhoJavals,Tvals[1]+Tvals[2], color = "black", linestyle = "solid", linewidth = 2); # + state
    axes[1].set_xlim(0,4);
    axes[1].set_xticks([0,2,4]);
    axes[1].set_xlabel("$J/\pi \sqrt{t(E+2t)}$", fontsize = "x-large");
    axes[1].set_ylim(0,0.25);
    axes[1].set_yticks([0,0.25]);
    axes[1].set_ylabel("$T_+$", fontsize = "x-large");
    plt.show();



    
