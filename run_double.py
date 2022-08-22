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
mymarkevery = 50;
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)","(e)","(f)","(g)","(h)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

# tight binding params
tl = 1.0;

# choose boundary condition
source = np.zeros(8); 
source[sourcei] = 1; # down up up

##################################################################################
#### effects of spatial separation

if False: # check similarity to menezes prediction at diff N
    num_plots = 3;
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);

    # iter over spatial separation
    Jval = 1.0;
    Nvals = [2,5,50];
    for Nvali in range(len(Nvals)):
        Nval = Nvals[Nvali];

        # construct hams
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
        i1, i2 = [1], [Nval];
        hblocks, tnn = wfm.utils.h_cicc_eff(Jval, tl, i1, i2, pair);
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

        # sweep over range of energies
        # def range
        logElims = -3,0
        Evals = np.logspace(*logElims,myxvals);
        kavals = np.arccos((Evals-2*tl)/(-2*tl));
        jprimevals = Jval/(4*tl*kavals);
        jprimevals = jprimevals*2*np.sqrt(1/2); # renormalize J!!!
        menez_Tf = jprimevals*jprimevals/(1+(5/2)*jprimevals*jprimevals+(9/16)*np.power(jprimevals,4));
        menez_Tnf = (1+jprimevals*jprimevals/4)/(1+(5/2)*jprimevals*jprimevals+(9/16)*np.power(jprimevals,4));
        Rvals = np.empty((len(Evals),len(source)), dtype = float);
        Tvals = np.empty((len(Evals),len(source)), dtype = float); 
        for Evali in range(len(Evals)):

            # energy
            Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper

            if(Evali < 1): # verbose
                Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, all_debug = False, verbose = verbose);
            else: # not verbose
                 Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, all_debug = False);
            Rvals[Evali] = Rdum;
            Tvals[Evali] = Tdum;

        # plot tight binding results
        axes[0].plot(Evals,Tvals[:,pair[0]], color = mycolors[Nvali], marker = mymarkers[Nvali], markevery = mymarkevery, linewidth = mylinewidth);
        #axes[2].plot(Evals,Tvals[:,sourcei], color = mycolors[Nvali], marker = mymarkers[Nvali], markevery = mymarkevery, linewidth = mylinewidth);
        totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
        #axes[3].plot(Evals, totals, color="red", label = "total ");

        # contamination of |+> by |->
        axes[2].plot(Evals,Tvals[:,pair[1]]/(Tvals[:,pair[0]]+Tvals[:,pair[1]]), color = mycolors[Nvali], marker = mymarkers[Nvali], markevery = mymarkevery, linewidth = mylinewidth);

        # plot differences
        axes[1].plot(Evals,abs(Tvals[:,pair[0]]-menez_Tf)/menez_Tf,color = mycolors[Nvali], marker = mymarkers[Nvali], markevery = mymarkevery, linewidth = mylinewidth);
        #axes[3].plot(Evals,abs(Tvals[:,sourcei]-menez_Tnf)/menez_Tnf,color = mycolors[Nvali], marker = mymarkers[Nvali], markevery = mymarkevery, linewidth = mylinewidth);
        
    # format
    axes[0].set_ylim(0,0.4)
    axes[0].set_ylabel('$T_{+}$', fontsize = myfontsize );
    axes[1].set_ylim(0,1.0);
    axes[1].set_ylabel('$|T_{+}-T_{+,c}|/T_{+,c}$', fontsize = myfontsize );
    #axes[3].set_ylim(0,1.0);
    axes[2].set_ylabel('$T_{-}/(T_{+}+T_{-})\,$', fontsize = myfontsize );
    
    # show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.93, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    plt.savefig('figs/separation.pdf');
    plt.show();

if False: # compare T- vs N to see how T- is suppressed at small N
    num_plots = 1
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);

    # sweep over energy
    Jval = 0.1;
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
            i1, i2 = [1], [1+N0];
            hblocks, tnn = wfm.utils.h_cicc_eff(Jval, tl, i1, i2, pair);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # get R, T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source);
            Rvals[Nvali] = Rdum;
            Tvals[Nvali] = Tdum;

        # plot T_- vs N
        axes[0].plot(Nvals,Tvals[:,pair[1]], color = mycolors[Evali], marker = mymarkers[Evali], markevery = 5, linewidth = mylinewidth);
        totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
        #axes[1].plot(Nvals, totals, color="red");

    # format
    axes[-1].set_xlabel('$N$',fontsize = myfontsize);
    axes[0].set_ylabel('$T_{-}$', fontsize = myfontsize );
    plt.tight_layout();
    plt.savefig('figs/Nlimit2.pdf');
    plt.show();

##################################################################################
#### effects of J

if False: # compare T+ vs E at different J
    num_plots = 2;
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);

    # sweep over J
    Jvals = [0.5,1.0,5.0,10.0]
    for Jvali in range(len(Jvals)):
        Jval = Jvals[Jvali];

        # sweep over energy
        logElims = -3,np.log10(3.99)
        Evals = np.logspace(*logElims,myxvals);
        kavals = np.arccos((Evals-2*tl)/(-2*tl));
        jprimevals = Jval/(4*tl*kavals);
        jprimevals = jprimevals*2*np.sqrt(1/2); # renormalize J!!!
        menez_Tf = jprimevals*jprimevals/(1+(5/2)*jprimevals*jprimevals+(9/16)*np.power(jprimevals,4));
        menez_Tnf = (1+jprimevals*jprimevals/4)/(1+(5/2)*jprimevals*jprimevals+(9/16)*np.power(jprimevals,4));
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
            i1, i2 = [1], [1+N0];
            hblocks, tnn = wfm.utils.h_cicc_eff(Jval, tl, i1, i2, pair);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # get R, T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source);
            Rvals[Evali] = Rdum;
            Tvals[Evali] = Tdum;

        # plot T_+ vs N
        axes[0].plot(Evals,Tvals[:,pair[0]], color = mycolors[Jvali], marker=mymarkers[Jvali], markevery = 50, linewidth = mylinewidth);
        totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
        #axes[0].plot(Evals, totals, color="red");

        # plot differences
        axes[1].plot(Evals,abs(Tvals[:,pair[0]]-menez_Tf)/menez_Tf,color = mycolors[Jvali], marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth);
        #axes[3].plot(Evals,abs(Tvals[:,sourcei]-menez_Tnf)/menez_Tnf,color = mycolors[Jvali], marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth);
   

    # format
    axes[0].set_ylim(0,0.4);
    axes[0].set_ylabel('$T_{+}$', fontsize = myfontsize );
    axes[1].set_ylim(0,1.0);
    axes[1].set_ylabel('$|T_{+}-T_{+,c}|/T_{+,c}$', fontsize = myfontsize );
    axes[0].axhline(0.25, color="gray");
    
    # show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    plt.savefig('figs/Jlimit.pdf');
    plt.show();

##################################################################################
#### overlapping non-contact

if False: # compare T+ vs M_1 = N to see if T_+=8/9 can be realized
    num_plots = 1;
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);

    # sweep over energy
    Jval = 0.01;
    Nmaxvals = np.array([2,5,50,100]);
    velvals = Nmaxvals*3*Jval/(2*np.pi);
    kavals = np.arcsin(velvals/2);
    Evals = 2*tl - 2*tl*np.cos(kavals)
    for Evali in range(len(Evals)):
        
        # energy
        Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
        Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper
        print(">>> ",Eval, kavals[Evali]);
        # sweep over NSR
        NSRlims = 1,100
        NSRvals = np.linspace(*NSRlims,50, dtype = int);
        Rvals = np.empty((len(NSRvals),len(source)), dtype = float);
        Tvals = np.empty((len(NSRvals),len(source)), dtype = float);
        for NSRvali in range(len(NSRvals)):
            NSRval = NSRvals[NSRvali];

            # construct hams
            # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
            hblocks, tnn = wfm.utils.h_cicc_hacked(Jval, tl, NSRval, NSRval+2, pair);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # get R, T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source);
            Rvals[NSRvali] = Rdum;
            Tvals[NSRvali] = Tdum;

        # plot T_+ vs N
        axes[0].plot(NSRvals,Tvals[:,pair[0]], color = mycolors[Evali], marker=mymarkers[Evali], markevery = 10, linewidth = mylinewidth);
        #axes[1].plot(NSRvals,Tvals[:,pair[1]], color = mycolors[Evali], marker=mymarkers[Evali], markevery = 10, linewidth = mylinewidth);
        totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
        #axes[0].plot(Evals, totals, color="red");

    # format
    axes[0].axhline(8/9, color = "grey");
    axes[0].set_xlim(0,100);
    axes[0].set_ylim(0,1.0);
    axes[0].set_ylabel('$T_{+}$', fontsize = myfontsize );
    axes[-1].set_xlabel('$M_1$',fontsize = myfontsize);
    plt.tight_layout();
    plt.savefig('figs/switzer_overlap.pdf');
    plt.show();

if False: # compare T+ vs K_i, with M_1 fixed small by J to try to see T_+=8/9 again
    num_plots = 2
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);

    # sweep over Nmax
    Nmaxvals = [2,5,50,100];
    for Nvali in range(len(Nmaxvals)):
        Nmax = Nmaxvals[Nvali];

        # sweep over energy
        logElims = -4,np.log10(3.99)
        Evals = np.logspace(*logElims,myxvals);
        Rvals = np.empty((len(Evals),len(source)), dtype = float);
        Tvals = np.empty((len(Evals),len(source)), dtype = float);
        kavals = np.arccos((Evals - 2*tl)/(-2*tl));
        velvals = 2*np.sin(kavals);
        Jvals = 2*np.pi*velvals/(3*Nmax);
        for Jvali in range(len(Jvals)):
            
            # energy
            Eval = Evals[Jvali]; # Eval > 0 always, what I call K in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper
            #print(Eval, Jvals[Jvali]);

            # construct hams
            # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
            hblocks, tnn = wfm.utils.h_cicc_hacked(Jvals[Jvali], tl, Nmax, Nmax+2, pair);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # get R, T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source);
            Rvals[Jvali] = Rdum;
            Tvals[Jvali] = Tdum;

        # plot vs E
        axes[0].plot(Evals,Tvals[:,pair[0]], color = mycolors[Nvali], marker=mymarkers[Nvali], markevery = 50, linewidth = mylinewidth);
        totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
        #axes[0].plot(Evals, totals, color="red");
        axes[1].plot(Evals, Jvals,color = mycolors[Nvali], marker=mymarkers[Nvali], markevery = 50, linewidth = mylinewidth);

    # format
    axes[0].axhline(8/9, color = "grey");
    axes[0].axhline(0.25, color = "grey");
    axes[0].set_ylim(0,1);
    axes[0].set_ylabel('$T_{+}$', fontsize = myfontsize );
    axes[1].set_ylabel('$J_{max}$', fontsize = myfontsize);

    # show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    plt.savefig('figs/switzer_Jmax.pdf');
    plt.show();

##################################################################################
#### non-overlapping non-contact

if False: # compare T+ vs M_1 = N/2 to see if T_+=8/9 can be realized
    num_plots = 1;
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);

    # sweep over energy
    Jval = 0.01;
    Nmaxvals = np.array([2,5,50,100]);
    velvals = Nmaxvals*3*Jval/(2*np.pi);
    kavals = np.arcsin(velvals/2);
    Evals = 2*tl - 2*tl*np.cos(kavals)
    for Evali in range(len(Evals)):
        
        # energy
        Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
        Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper
        print(">>> ",Eval, kavals[Evali]);
        # sweep over M1
        Mlims = 1,100
        Mvals = np.linspace(*Mlims,40, dtype = int);
        Rvals = np.empty((len(Mvals),len(source)), dtype = float);
        Tvals = np.empty((len(Mvals),len(source)), dtype = float);
        for Mvali in range(len(Mvals)):
            Mval = Mvals[Mvali];

            # construct hams
            # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
            i1, i2 = list(range(1,Mval+1)), list(range(Mval+1,2*Mval+1));
            hblocks, tnn = wfm.utils.h_cicc_eff(Jval, tl, i1, i2, pair);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # get R, T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source);
            Rvals[Mvali] = Rdum;
            Tvals[Mvali] = Tdum;

        # plot T_+ vs N
        axes[0].plot(Mvals,Tvals[:,pair[0]], color = mycolors[Evali], marker=mymarkers[Evali], markevery = 10, linewidth = mylinewidth);
        #axes[1].plot(Mvals,Tvals[:,pair[1]], color = mycolors[Evali], marker=mymarkers[Evali], markevery = 10, linewidth = mylinewidth);
        totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
        #axes[0].plot(Evals, totals, color="red");

    # format
    axes[0].axhline(8/9, color = "grey");
    axes[0].set_ylim(0,1.0);
    axes[0].set_ylabel('$T_{+}$', fontsize = myfontsize );

    # show
    axes[0].set_xlim(*Mlims);
    axes[-1].set_xlabel('$M_1$',fontsize = myfontsize);
    plt.tight_layout();
    plt.savefig('figs/switzer_nonoverlap.pdf');
    plt.show();

if True: # compare T+ vs K_i, with M_1 fixed small by J to try to see T_+=8/9 again
    num_plots = 2;
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);

    # sweep over Nmax
    Nmaxvals = [2,5,50,100];
    for Nvali in range(len(Nmaxvals)):
        Nmax = Nmaxvals[Nvali];

        # sweep over energy
        logElims = -4,np.log10(3.99);
        Evals = np.logspace(*logElims,myxvals);
        Rvals = np.empty((len(Evals),len(source)), dtype = float);
        Tvals = np.empty((len(Evals),len(source)), dtype = float);
        kavals = np.arccos((Evals - 2*tl)/(-2*tl));
        velvals = 2*np.sin(kavals);
        Jvals = 2*np.pi*velvals/(3*Nmax);
        for Jvali in range(len(Jvals)):
            
            # energy
            Eval = Evals[Jvali]; # Eval > 0 always, what I call K in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper
            #print(Eval, Jvals[Jvali]);

            # construct hams
            # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
            i1, i2 = list(range(1,Nmax+1)), list(range(Nmax+1,2*Nmax+1));
            hblocks, tnn = wfm.utils.h_cicc_eff(Jvals[Jvali], tl, i1, i2, pair);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # get R, T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source);
            Rvals[Jvali] = Rdum;
            Tvals[Jvali] = Tdum;

        # plot vs E
        axes[0].plot(Evals,Tvals[:,pair[0]], color = mycolors[Nvali], marker=mymarkers[Nvali], markevery = 50, linewidth = mylinewidth);
        totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
        #axes[0].plot(Evals, totals, color="red");
        axes[1].plot(Evals, Jvals,color = mycolors[Nvali], marker=mymarkers[Nvali], markevery = 50, linewidth = mylinewidth);

    # format
    axes[0].axhline(8/9, color = "grey");
    axes[0].axhline(0.25, color = "grey");
    axes[0].set_ylim(0,1);
    axes[0].set_ylabel('$T_{+}$', fontsize = myfontsize );
    axes[1].set_ylabel('$J_{max}$', fontsize = myfontsize );

    # show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    plt.savefig('figs/switzer_Jmax_nonoverlap.pdf');
    plt.show();

    
