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
import matplotlib.cm
import sys

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;
pair = (1,2); # pair[0] is the + state after entanglement
sourcei = 4;

# fig standardizing
myxvals = 199;
myfontsize = 14;
mycolors = ["black","darkblue","darkgreen","darkred", "darkmagenta","darkgray","darkcyan"];
mymarkers = ["o","^","s","d","X","P","*"];
mymarkevery = 5;
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)"];
plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

##################################################################################
#### entanglement generation (cicc Fig 6)

if False: # compare T vs N directly
    num_plots = 1
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);

    # siam inputs
    tl = 1.0;
    Jeff = 0.1;

    # choose boundary condition
    source = np.zeros(8); # incident up, imps = down, down
    source[4] = 1;
    spinstate = "baa"

    # sweep over energy
    Evals = [10**(-4),10**(-3),10**(-2),10**(-1)];
    for Evali in range(len(Evals)):

        # energy
        Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
        Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper

        # sweep over N
        Nvals = np.linspace(2,21,20,dtype = int);
        Rvals = np.empty((len(Nvals),len(source)), dtype = float);
        Tvals = np.empty((len(Nvals),len(source)), dtype = float);
        for Nvali in range(len(Nvals)):
        
            # location of impurities
            N0 = Nvals[Nvali] - 1;
            print(">>> N0 = ",N0);

            # construct hams
            # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
            i1, i2 = 1, 1+N0;
            hblocks, tnn = wfm.utils.h_cicc_eff(Jeff, tl, i1, i2, i2+2, pair);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # get R, T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source);
            Rvals[Nvali] = Rdum;
            Tvals[Nvali] = Tdum;

        # plot T_- vs N
        axes[0].plot(Nvals,Tvals[:,pair[1]], color = mycolors[Evali], marker = mymarkers[Evali], markevery = mymarkevery, linewidth = mylinewidth);
        totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
        #axes[1].plot(Nvals, totals, color="red");

    # format
    axes[-1].set_xlabel('$N$',fontsize = myfontsize);
    axes[0].set_ylabel('$T_{-}$', fontsize = myfontsize );
    plt.tight_layout();
    plt.savefig('figs/Nlimit.pdf');
    plt.show();


if False: # compare T vs E at different J
    num_plots = 1
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);

    # tight binding params
    tl = 1.0;

    # choose boundary condition
    source = np.zeros(8); # incident up, imps = down, down
    source[4] = 1;
    spinstate = "baa"

    # sweep over J
    Jvals = [0.1,1,2,10]
    for Ji in range(len(Jvals)):
        Jeff = Jvals[Ji];

        # sweep over energy
        logElims = -5,0
        Evals = np.logspace(*logElims,myxvals);
        Rvals = np.empty((len(Evals),len(source)), dtype = float);
        Tvals = np.empty((len(Evals),len(source)), dtype = float);
        for Evali in range(len(Evals)):

            # energy
            Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper
        
            # location of impurities
            N0 = 1;

            # construct hams
            # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
            i1, i2 = 1, 1+N0;
            hblocks, tnn = wfm.utils.h_cicc_eff(Jeff, tl, i1, i2, i2+2, pair);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # get R, T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source);
            Rvals[Evali] = Rdum;
            Tvals[Evali] = Tdum;

        # plot T_- vs N
        axes[0].plot(Evals,Tvals[:,pair[0]], color = mycolors[Ji], marker=mymarkers[Ji], markevery = mymarkevery, linewidth = mylinewidth);
        totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
        #axes[0].plot(Evals, totals, color="red");

    # format
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    axes[0].set_ylabel('$T_{+}$', fontsize = myfontsize );
    plt.tight_layout();
    plt.savefig('figs/Jlimit.pdf');
    plt.show();

if True: # compare T vs NSR at different J
    num_plots = 1
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);

    # tight binding params
    tl = 1.0;
    Jeff = 0.01;

    # choose boundary condition
    source = np.zeros(8); # incident up, imps = down, down
    source[4] = 1;
    spinstate = "baa"

    # sweep over energy
    Nmaxvals = np.array([5,10,50,100]);
    velvals = Nmaxvals*3*Jeff/(2*np.pi);
    kavals = np.arcsin(velvals/2);
    Evals = 2*tl - 2*tl*np.cos(kavals)
    for Evali in range(len(Evals)):
        
        # energy
        Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
        Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper

        # sweep over NSR
        NSRlims = 1,100
        NSRvals = np.linspace(*NSRlims,50, dtype = int);
        Rvals = np.empty((len(NSRvals),len(source)), dtype = float);
        Tvals = np.empty((len(NSRvals),len(source)), dtype = float);
        for NSRvali in range(len(NSRvals)):
            NSRval = NSRvals[NSRvali];

            # construct hams
            # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
            hblocks, tnn = wfm.utils.h_cicc_hacked(Jeff, tl, NSRval, NSRval+2, pair);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # get R, T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source);
            Rvals[NSRvali] = Rdum;
            Tvals[NSRvali] = Tdum;

        # plot T_- vs N
        axes[0].plot(NSRvals,Tvals[:,pair[0]], color = mycolors[Evali], marker=mymarkers[Evali], markevery = 10, linewidth = mylinewidth);
        totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
        #axes[0].plot(Evals, totals, color="red");

    # format
    axes[0].axhline(8/9, color = "grey");
    axes[0].set_xlim(0,100);
    axes[0].set_ylim(0,1.0);
    axes[0].set_ylabel('$T_{+}$', fontsize = myfontsize );
    axes[-1].set_xlabel('$N$',fontsize = myfontsize);
    plt.tight_layout();
    plt.savefig('figs/Jlimit2.pdf');
    plt.show();

    
