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
pair = (1,2); # pair[0] is the + state
sourcei = 4;

# fig standardizing
myfontsize = 14;
mycolors = ["black","darkblue","darkgreen","darkred", "darkmagenta","darkgray","darkcyan"];
mystyles = ["solid", "dashed","dotted","dashdot"];
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

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


if False: # compare T vs rhoJa for N fixed

    # siam inputs
    tl = 1.0;
    Jeff = 0.1;

    # choose boundary condition
    source = np.zeros(8); # incident up, imps = down, down
    source[4] = 1;
    spinstate = "baa";

    for DeltaVg in [2*tl]: #tl*np.array([1,1.5,2,2.5,3]):

        # iter over E, getting T
        logElims = -5,-1
        Evals = np.logspace(*logElims,199);
        Rvals = np.empty((len(Evals),len(source)), dtype = float);
        Tvals = np.empty((len(Evals),len(source)), dtype = float);
        for Evali in range(len(Evals)):

            # energy and K fixed by J, rhoJ
            #E_rho = Jeff*Jeff/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                                    # this E is measured from bottom of band !!!

            # energy
            Eval = Evals[Evali];
            Energy = Eval - 2*tl;
            
            # optical distances, N = 2 fixed
            ka = np.arccos((Energy)/(-2*tl));
            #Vg = Energy + 2*tl; # gate voltage
            Vg = Energy + DeltaVg;
            N0 = 2;

            # construct hams
            # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
            i1, i2 = 1, 1+N0;
            hblocks, tnn = wfm.utils.h_cicc_eff(Jeff, tl, i1, i2, i2+2, pair);
            hblocks[1] += Vg*np.eye(len(source)); # Vg shift in SR
            hblocks[2] += Vg*np.eye(len(source));
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping
            if(verbose > 3 and Eval == Evals[0]): print(hblocks);

            # get T from this setup
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source);
            Rvals[Evali] = Rdum;
            Tvals[Evali] = Tdum;

        # save data to .npy
        data = np.zeros((2+2*len(source),len(Evals)));
        data[0,0] = tl;
        data[0,1] = Jeff;
        data[1,:] = Evals;
        data[2:10,:] = Tvals.T; # 8 spin dofs
        data[10:,:] = Rvals.T;
        fname = "data/model12/N"+str(N0)+"_"+str(DeltaVg);
        print("Saving data to "+fname);
        np.save(fname, data);


########################################################################

# figure of merit
def FOM(Ti,Tp, grid=100000):

    thetavals = np.linspace(0,np.pi,grid);
    p2vals = Ti*Tp/(Tp*np.cos(thetavals)*np.cos(thetavals)+Ti*np.sin(thetavals)*np.sin(thetavals));
    fom = np.trapz(p2vals, thetavals)/np.pi;
    return fom;

#### plot cicc-like data
if True:
    fig = plt.figure();
    fig.set_size_inches(7/2,6/2);
    dataf = sys.argv[1];
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

    # plot 3 possibilities
    sigmas = [sourcei,pair[0], pair[1]];
    mainax = plt.subplot(3,1,(1,2));
    for sigmai in range(len(sigmas)):
        mainax.plot(xvals, Tvals[sigmas[sigmai]],color = mycolors[sigmai],linewidth = mylinewidth);
        mainax.set_xscale('log', subs = []);

    # inset zoom in
    insax = plt.subplot(3,1,3);
    insax.plot(xvals, Tvals[pair[1]],color = mycolors[2],linewidth = mylinewidth);
    insax.set_xscale('log', subs = []);
    insax.set_xlim(10**(-5), 10**(-1));
    insax.set_xticks([10**(-5),10**(-4),10**(-3),10**(-2),10**(-1)]);
    insax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));
    insax.set_xlabel('$(E+2t)/t$',fontsize = myfontsize);

    # format
    mainax.sharex(insax);
    mainax.label_outer();
    insax.label_outer();
    mainax.set_ylim(0,1.0);
    mainax.set_yticks([0,0.5,1]);
    mainax.set_ylabel('$T$', fontsize = myfontsize);
    mainax.set_title(mypanels[0], x=0.07, y = 0.8, fontsize = myfontsize);
    insax.set_ylim(0,2.0*10**(-3));
    insax.set_yticks([0,2.5*10**(-3)]);
    insax.set_ylabel('$T_-$', fontsize = myfontsize);  
    insax.set_title(mypanels[1], x=0.07, y = 0.45, fontsize = myfontsize);
    plt.tight_layout();
    plt.show();
    #plt.savefig('model12.pdf');

    # fom zoom in
    fig, fomax = plt.subplots();
    fomvals = np.empty_like(xvals);
    for xi in range(len(xvals)):
        fomvals[xi] = FOM(Tvals[sourcei,xi],Tvals[pair[0],xi]);
    fomax.plot(xvals, fomvals);
    fomax.set_xscale('log', subs = []);
    fomax.set_xlim(10**(-5), 10**(-1));
    fomax.set_xticks([10**(-5),10**(-4),10**(-3),10**(-2),10**(-1)]);
    fomax.set_xlabel('$(E+2t)/t$',fontsize = myfontsize);
    plt.show();
        






#### plot T_- / T_+ at different Vg
if False:
    fig, ax = plt.subplots();
    datafs = sys.argv[1:];
    for datafi in range(len(datafs)):
        dataf = datafs[datafi];
        print("Loading data from "+dataf);
        data = np.load(dataf);
        tl = data[0,0];
        Jeff = data[0,1];
        xvals = data[1];
        Tvals = data[2:10];
        Rvals = data[10:];
        #ax.plot(xvals, Tvals[pair[1]]/Tvals[pair[0]],color = mycolors[datafi],linewidth = mylinewidth);
        ax.plot(xvals, Tvals[pair[1]],color = mycolors[datafi],linewidth = mylinewidth);
        ax.set_xscale('log', subs = []);
        ax.set_xlim(10**(-5), 10**(-1));
        ax.set_xticks([10**(-5),10**(-4),10**(-3),10**(-2),10**(-1)]);

    # format
    ax.set_xlabel('$(E+2t)/t$',fontsize = myfontsize);
    ax.set_ylabel('$T$', fontsize = myfontsize);  
    plt.tight_layout();
    plt.show();

#### plot data side by side
if False:
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




    
