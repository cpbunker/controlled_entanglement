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
import sys

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;
pair = (1,2);
sourcei = 4;

# fig standardizing
myfontsize = 14;
mycolors = ["black","darkblue","darkgreen","darkred", "darkmagenta","darkgray","darkcyan"];
mystyles = ["solid", "dashed","dotted","dashdot"];
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)"];
plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

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

    # iter over E, getting T
    Tvals, Rvals = [], [];
    logElims = -4,-1
    Evals = np.logspace(*logElims,199);
    for Eval in Evals:

        # energy and K fixed by J, rhoJ
        #E_rho = Jeff*Jeff/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                                # this E is measured from bottom of band !!!
        Energy = Eval - 2*tl;
        
        # location of impurities, fixed by kx0 = pi
        k_rho = np.arccos(Energy/(-2*tl));
        kx0 = 1*np.pi;
        N0 = max(1,int(kx0/(k_rho)));
        print(">>> N0 = ",N0);

        # construct hams
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
        i1, i2 = 1, 1+N0;
        hblocks, tnn = wfm.utils.h_cicc_eff(Jeff, tl, i1, i2, i2+2, pair);
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

        # get T from this setup
        Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source));
        Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source, reflect = True));

    # save data to .npy
    Tvals, Rvals = np.array(Tvals), np.array(Rvals);
    data = np.zeros((2+2*len(source),len(Evals)));
    data[0,0] = tl;
    data[0,1] = Jeff;
    data[1,:] = Evals;
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

    # iter over E, getting T
    Tvals, Rvals = [], [];
    #rhoJavals = np.linspace(xlims[0], xlims[1], 99);
    logElims = -4,-1
    Evals = np.logspace(*logElims,199);
    for Eval in Evals:

        # energy and K fixed by J, rhoJ
        #E_rho = Jeff*Jeff/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                                # this E is measured from bottom of band !!!
        Energy = Eval - 2*tl;
        
        # optical distances, N = 2 fixed
        ka = np.arccos((Energy)/(-2*tl));
        Vg = Energy + 2*tl; # gate voltage
        N0 = 1;

        # construct hams
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
        i1, i2 = 1, 1+N0;
        hblocks, tnn = wfm.utils.h_cicc_eff(Jeff, tl, i1, i2, i2+2, pair);
        hblocks[1] += Vg*np.eye(len(source)); # Vg shift in SR
        hblocks[2] += Vg*np.eye(len(source));
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping
        if(verbose > 3 and Eval == Evals[0]): print(hblocks);

        # get T from this setup
        Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source));
        Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source, reflect = True));

    # save data to .npy
    Tvals, Rvals = np.array(Tvals), np.array(Rvals);
    data = np.zeros((2+2*len(source),len(Evals)));
    data[0,0] = tl;
    data[0,1] = Jeff;
    data[1,:] = Evals;
    data[2:10,:] = Tvals.T;
    data[10:,:] = Rvals.T;
    fname = "data/model12/N2";
    print("Saving data to "+fname);
    np.save(fname, data);


########################################################################
#### plot data

# open command line file
datafs = sys.argv[1:];
fig, axes = plt.subplots(1,len(datafs), sharey = True);
fig.set_size_inches(7/2,6/2);
if( len(datafs)== 1): axes = [axes];
for fi in range(len(datafs)):
    dataf = datafs[fi];
    print("Loading data from "+dataf);
    data = np.load(dataf);
    tl = data[0,0];
    Jeff = data[0,1];
    xvals = data[1];
    Tvals = data[2:10];
    Rvals = data[10:];
    totals = np.sum(Tvals, axis = 0) + np.sum(Rvals, axis = 0);
    print("- shape xvals = ", np.shape(xvals));
    print("- shape Tvals = ", np.shape(Tvals));
    print(np.max(Tvals[pair[0]]))

    # plot
    axes[fi].set_title(mypanels[fi], x=0.12, y = 0.88);
    axes[fi].plot(xvals, Tvals[sourcei],color = "black", linestyle = "solid",linewidth = mylinewidth);
    axes[fi].plot(xvals, Tvals[pair[0]], color = "black", linestyle = "dashed", linewidth = mylinewidth);
    #axes[fi].plot(xvals, Tvals[pair[1]], color = "black", linestyle = "dotted", linewidth = mylinewidth);
    axes[fi].plot(xvals, totals, color="red");
    axes[fi].set_xscale('log', subs = []);
    axes[fi].set_xlim(10**(-4), 10**(-1));
    axes[fi].set_xticks([10**(-4),10**(-3),10**(-2),10**(-1)])
    axes[fi].set_xlabel('$(E+2t)/t$',fontsize = myfontsize);
    
# format
axes[0].set_ylim(0,1.0);
axes[0].set_yticks([0,0.5,1]);
axes[0].set_ylabel('$T$', fontsize = myfontsize);  
plt.tight_layout();
plt.show();
#plt.savefig('model12.pdf');




    
