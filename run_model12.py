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
mypanels = ["(a)","(b)","(c)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"});

# tight binding params
tl = 1.0;

# choose boundary condition
source = np.zeros(8); 
source[sourcei] = 1; # down up up

##################################################################################
#### entanglement generation (cicc Fig 6)

if False: # compare T vs rhoJa for N not fixed

    # iter over E, getting T
    logElims = -4,-1
    Evals = np.logspace(*logElims,myxvals);
    Rvals = np.empty((len(Evals),len(source)), dtype = float);
    Tvals = np.empty((len(Evals),len(source)), dtype = float);
    for Evali in range(len(Evals)):

        # energy
        Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
        Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper
        
        # location of impurities, fixed by kx0 = pi
        k_rho = np.arccos(Energy/(-2*tl)); # = ka since \varepsilon_0ss = 0
        kx0 = 2.0*np.pi;
        N0 = max(1,int(kx0/(k_rho))); #N0 = (N-1)
        print(">>> N0 = ",N0);

        # construct hams
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
        i1, i2 = 1, 1+N0;
        hblocks, tnn = wfm.utils.h_cicc_eff(Jeff, tl, i1, i2, i2+2, pair);
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

        # get R, T coefs
        Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source);
        Rvals[Evali] = Rdum;
        Tvals[Evali] = Tdum;

    # save data to .npy
    data = np.zeros((2+2*len(source),len(Evals)));
    data[0,0] = tl;
    data[0,1] = Jeff;
    data[1,:] = Evals;
    data[2:10,:] = Tvals.T;
    data[10:,:] = Rvals.T;
    fname = "data/model12/Nx/"+str(int(kx0*100)/100);
    print("Saving data to "+fname);
    np.save(fname, data);


if False: # compare T vs rhoJa for N=2 fixed
    Jval = 0.1;

    # iter over E, getting T
    logElims = -4,-1;
    Evals = np.logspace(*logElims,myxvals);
    Rvals = np.empty((len(Evals),len(source)), dtype = float);
    Tvals = np.empty((len(Evals),len(source)), dtype = float);
    for Evali in range(len(Evals)):

        # energy
        Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
        Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper
        
        # optical distances, N = 2 fixed
        N0 = 1; # N0 = N - 1

        # construct hams
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
        i1, i2 = [1], [N0+1];
        hblocks, tnn = wfm.utils.h_cicc_eff(Jval, tl, i1, i2, pair);
        #hblocks[1] += Vg*np.eye(len(source)); # Vg shift in SR
        #hblocks[2] += Vg*np.eye(len(source));
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping
        if(verbose > 3 and Eval == Evals[0]): print(hblocks);

        # get R, T coefs
        Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source);
        Rvals[Evali] = Rdum;
        Tvals[Evali] = Tdum;

    # save data to .npy
    data = np.zeros((2+2*len(source),len(Evals)));
    data[0,0] = tl;
    data[0,1] = Jval;
    data[1,:] = Evals;
    data[2:10,:] = Tvals.T; # 8 spin dofs
    data[10:,:] = Rvals.T;
    fname = "data/model12/N2/"+str(int(Jval*10)/10);
    print("Saving data to "+fname);
    np.save(fname, data);


########################################################################
#### plot data

# load data
def load_data(fname):
    print("Loading data from "+fname);
    data = np.load(fname);
    tl = data[0,0];
    Jeff = data[0,1];
    myxvals = data[1];
    myTvals = data[2:10];
    myRvals = data[10:];
    mytotals = np.sum(myTvals, axis = 0) + np.sum(myRvals, axis = 0);
    print("- shape xvals = ", np.shape(myxvals));
    print("- shape Tvals = ", np.shape(myTvals));
    print("- shape Rvals = ", np.shape(myRvals));
    return myxvals, myRvals, myTvals, mytotals;

# p2
def p2(Ti,Tp,theta):
    assert isinstance(Ti,float) and isinstance(Tp,float); # vectorized in thetas only
    return Ti*Tp/(Tp*np.cos(theta/2)*np.cos(theta/2)+Ti*np.sin(theta/2)*np.sin(theta/2));

# figure of merit
def FOM(Ti,Tp, grid=100000):
    thetavals = np.linspace(0,np.pi,grid);
    p2vals = p2(Ti,Tp,thetavals);
    fom = np.trapz(p2vals, thetavals)/np.pi;
    return fom;

#### plot T+ like cicc figure
if True:
    num_plots = 2;
    fig, axes = plt.subplots(num_plots, sharex=True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);
    dataf = sys.argv[1];
    xvals, Rvals, Tvals, totals = load_data(dataf);
    logElims = np.log10(xvals[0]), np.log10(xvals[-1]);

    # plot Ti, T+, T-
    sigmas = [sourcei,pair[0], pair[1]];
    for sigmai in range(len(sigmas)):
        factor = 1;
        if sigmas[sigmai] == pair[1]: factor = 1000; # blow up T-
        axes[0].plot(xvals, factor*Tvals[sigmas[sigmai]],color = mycolors[sigmai],marker = mymarkers[sigmai],markevery=50,linewidth = mylinewidth);
    print(">>> T+ max = ",np.max(Tvals[pair[0]])," at Ki = ",xvals[np.argmax(Tvals[pair[0]])]);
    axes[0].set_ylim(0,1.0);
    axes[0].set_ylabel('$T_\sigma$', fontsize = myfontsize);
    
    # plot p2 at diff theta
    numtheta = 9;
    thetavals = np.linspace(0,np.pi,numtheta);
    for thetai in range(numtheta):
        cm_reds = matplotlib.cm.get_cmap("seismic");
        yvals = [];
        for xi in range(len(xvals)):
            yvals.append(p2(Tvals[sourcei,xi],Tvals[pair[0],xi],thetavals[thetai]));
        axes[1].plot(xvals, yvals,color = cm_reds((1+thetai)/(numtheta)));
    if False: # colorbar
        cb_reds = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cm_reds),location="right", ax=thetax,);
        cb_reds.set_label("$\\theta$",rotation = "horizontal");
        cb_reds.set_ticks([0,1],labels=["$\\tilde{\\theta} =$ 0","$\pi$"]);
    axes[1].set_ylim(0,1.0);
    axes[1].set_ylabel('$p^2(\\tilde{\\theta})$', fontsize = myfontsize);

    # plot analytical FOM
    axes[1].plot(xvals, np.sqrt(Tvals[sourcei]*Tvals[pair[0]]), color = mycolors[0], marker=mymarkers[0],markevery=50, linewidth = mylinewidth)
    print(">>> p2 max = ",np.max(np.sqrt(Tvals[sourcei]*Tvals[pair[0]]))," at Ki = ",xvals[np.argmax(np.sqrt(Tvals[sourcei]*Tvals[pair[0]]))]);
    # show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    plt.savefig('figs/model12.pdf');
    plt.show();
    
        
#### plot data at different vals of kx0
if False:
    folder = sys.argv[1];
    myrows, mycols = 3,2;
    fig, axes = plt.subplots(nrows = myrows, ncols = mycols, sharex = 'col', sharey = 'row'); # cols are T, R
    sigmas = [pair[0],pair[1],sourcei];
    nvals = np.array([0,1,1.25,1.5,1.75,2]);
    for ni in range(len(nvals)):
        pival = int(np.pi*nvals[ni]*100)/100
        dataf = "data/model12/"+folder+"/"+str(pival)+".npy";
        xvals, Rvals, Tvals, totals = load_data(dataf);

        # plot R+, R-, Ri
        for rowi in range(myrows):
            axes[rowi,0].plot(xvals, Rvals[sigmas[rowi]], color = mycolors[ni], label = nvals[ni]);
        #axes[1,0].plot(xvals, Rvals[pair[1]], color = mycolors[ni], label = nvals[ni]);
        #axes[2,0].plot(xvals, Rvals[sourcei], color = mycolors[ni], label = nvals[ni]);

        # plot T+, T-, Ti
        for rowi in range(myrows):
            axes[rowi,1].plot(xvals, Tvals[sigmas[rowi]], color = mycolors[ni], label = nvals[ni]);

        # plot totals
        axes[-1,-1].plot(xvals, totals, color='red');

    # format
    stems = ['$R','$T'];
    subscripts = ['_+$','_-$','_i$'];
    for coli in range(mycols):
        axes[-1,coli].set_xscale('log', subs = []);
        axes[-1,coli].set_xlim(10**(-5), 10**(-1));
        axes[-1,coli].set_xticks([10**(-5),10**(-4),10**(-3),10**(-2),10**(-1)]);
        axes[-1,coli].set_xlabel('$K_i/t$',fontsize = myfontsize);
        axes[-1,coli].legend();
        for rowi in range(myrows):
            axes[rowi,coli].set_ylabel(stems[coli]+subscripts[rowi],rotation = "horizontal");
    plt.tight_layout();
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




    
