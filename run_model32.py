'''
Christian Bunker
M^2QM at UF
October 2021

Quasi 1 body transmission through spin impurities project, part 4:
Cobalt dimer modeled as two spin-3/2 impurities mo
Spin interaction parameters calculated from dft, Jie-Xiang's Co dimer manuscript
'''

from code import fci_mod, wfm
from code.wfm import utils

import numpy as np
import matplotlib.pyplot as plt
import sys

#### top level
#np.set_printoptions(precision = 4, suppress = True);
verbose = 5;

# fig standardizing
myxvals = 199;
myfontsize = 14;
mycolors = ["black","darkblue","darkgreen","darkred", "darkmagenta","darkgray","darkcyan"];
mymarkers = ["o","^","s","d","X","P","*"];
mymarkevery = 50;
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

#### setup
peaks = np.array([ [   1/2 , 0.00 , 0.222490 , 0.298361 ],
    [3/2 , -0.08 , 0.071040 , 0.199658 ],
    [3/2 , -0.05 , 0.082684 , 0.211454 ],
    [3/2 , -0.01 , 0.113272 , 0.234906 ],
    [3/2 , 0.00  , 0.123194 , 0.242528 ],
    [3/2 , 0.01  , 0.123384 , 0.249863 ],
    [3/2 , 0.05  , 0.100259 , 0.257826 ],
    [3/2 , 0.08  , 0.087848 , 0.252942 ],
    [6   , 0.003 , 0.042640 , 0.167187 ] ]);
del peaks;

# def particles and their single particle states
species = np.array([1,1,1]); # num of each species, which are one e, elec, spin-3/2, spin-3/2
spec_strs = ["e","1","2"];
states = [[0,1],[2,3,4,5],[6,7,8,9]]; # e up, down, spin 1 mz, spin 2 mz
state_strs = ["0.5_","-0.5_","1.5_","0.5_","-0.5_","-1.5_","1.5_","0.5_","-0.5_","-1.5_"];
#dets = np.array([xi for xi in itertools.product(*tuple(states))]); # product states
dets52 = [[0,2,7],[0,3,6],[1,2,6]]; # total spin 5/2 subspace

# initialize source vector in down, 3/2, 3/2 state
sourcei = 2; # |down, 3/2, 3/2 >
assert(sourcei >= 0 and sourcei < len(dets52));
source = np.zeros(len(dets52));
source[sourcei] = 1;
source_str = "|";
for si in dets52[sourcei]: source_str += state_strs[si];
source_str += ">";
if(verbose): print("\nSource:\n"+source_str);

# entangle pair
pair = (0,1); # |up, 1/2, 3/2 > and |up, 3/2, 1/2 >
if(verbose):
    print("\nEntangled pair:");
    pair_strs = [];
    for pi in pair:
        pair_str = "|";
        for si in dets52[pi]: pair_str += state_strs[si];
        pair_str += ">";
        print(pair_str);
        pair_strs.append(pair_str);

tl = 1.0;
tp = 1.0;
JK = 0.1;
J12 = JK/10;
J12x, J12y, J12z = J12, J12, J12;
            
#########################################################
#### effects of Ki and Delta E

if False: # T+ at different Delta E by changing D
    
    Esplitvals = (1)*np.array([0.0,-0.01,-0.05,-0.08]);
    Dvals = -Esplitvals/2;
    for Dvali in range(len(Dvals)):
        Dval = Dvals[Dvali];

        # iter over E, getting T
        logElims = -4,-1
        Evals = np.logspace(*logElims,myxvals);
        Rvals = np.empty((len(Evals),len(source)), dtype = float);
        Tvals = np.empty((len(Evals),len(source)), dtype = float);
        for Evali in range(len(Evals)):

            # energy
            Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper
            
            # optical distances, N = 2 fixed
            N0 = 1; # N0 = N - 1

            # construct hblocks
            hblocks = [];
            impis = [1,2];
            for j in range(4): # LL, imp 1, imp 2, RL
                # define all physical params
                JK1, JK2 = 0, 0;
                if(j == impis[0]): JK1 = JK;
                elif(j == impis[1]): JK2 = JK;
                params = J12x, J12y, J12z, Dval, Dval, 0, JK1, JK2;
                h1e, g2e = wfm.utils.h_cobalt_2q(params); # construct ham
                # construct h_SR (determinant basis)
                hSR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);            
                # transform to eigenbasis
                hSR_diag = wfm.utils.entangle(hSR, *pair);
                hblocks.append(np.copy(hSR_diag));
                if(verbose > 3 and Eval == Evals[0]):
                    print("\nJK1, JK2 = ",JK1, JK2);
                    print(" - ham:\n", hSR);
                    print(" - transformed ham:\n", np.real(hSR_diag));
                    print(" - DeltaE = ",Esplitvals[Dvali])

            # finish hblocks
            hblocks = np.array(hblocks);
            E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
            for hb in hblocks:
                hb += -E_shift*np.eye(np.shape(hblocks[0])[0]);
            print("Delta E / t = ", (hblocks[0][0,0] - hblocks[0][2,2])/tl);

            # hopping
            tnn = np.array([-tl*np.eye(len(source)),-tp*np.eye(len(source)),-tl*np.eye(len(source))]);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # get R, T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source, all_debug = False);
            Rvals[Evali] = Rdum;
            Tvals[Evali] = Tdum;
         
        # save data to .npy
        data = np.zeros((2+2*len(source),len(Evals)));
        data[0,0] = tl;
        data[0,1] = JK;
        data[1,:] = Evals;
        data[2:2+len(source),:] = Tvals.T;
        data[2+len(source):2+2*len(source),:] = Rvals.T;
        fname = "data/model32/Esplit"+str(int(Esplitvals[Dvali]*100)/100);
        print("Saving data to "+fname);
        np.save(fname, data);

# load data
def load_data(fname):
    print("Loading data from "+fname);
    data = np.load(fname);
    tl = data[0,0];
    Jeff = data[0,1];
    myxvals = data[1];
    myTvals = data[2:5];
    myRvals = data[5:];
    mytotals = np.sum(myTvals, axis = 0) + np.sum(myRvals, axis = 0);
    print("- shape xvals = ", np.shape(myxvals));
    print("- shape Tvals = ", np.shape(myTvals));
    print("- shape Rvals = ", np.shape(myRvals));
    return myxvals, myRvals, myTvals, mytotals;

# p2
def p2(Ti,Tp,theta):
    assert isinstance(Ti,float) and isinstance(Tp,float); # vectorized in thetas only
    if Tp == 0.0: Tp = 1e-10;
    return Ti*Tp/(Tp*np.cos(theta/2)*np.cos(theta/2)+Ti*np.sin(theta/2)*np.sin(theta/2));

# figure of merit
def FOM(Ti,Tp, grid=100000):
    thetavals = np.linspace(0,np.pi,grid);
    p2vals = p2(Ti,Tp,thetavals);
    fom = np.trapz(p2vals, thetavals)/np.pi;
    return fom;

#### plot
if True:
    num_plots = 2;
    fig, axes = plt.subplots(num_plots, sharex=True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);
    datafs = sys.argv[1:];
    for fi in range(len(datafs)):
        xvals, Rvals, Tvals, totals = load_data(datafs[fi]);
        logElims = -2,0;

        # plot T+
        axes[0].plot(xvals, Tvals[pair[0]], color=mycolors[fi], marker=mymarkers[fi], markevery=mymarkevery, linewidth = mylinewidth); 
        #mainax.plot(xvals, totals, color="red");
        print(">>> T+ max = ",np.max(Tvals[pair[0]])," at Ki = ",xvals[np.argmax(Tvals[pair[0]])]);

        # plot analytical FOM
        axes[1].plot(xvals, np.sqrt(Tvals[sourcei]*Tvals[pair[0]]), color = mycolors[fi], marker=mymarkers[fi],markevery=mymarkevery, linewidth = mylinewidth)
        print(">>> p2 max = ",np.max(np.sqrt(Tvals[sourcei]*Tvals[pair[0]]))," at Ki = ",xvals[np.argmax(np.sqrt(Tvals[sourcei]*Tvals[pair[0]]))]);
        
    # format
    axes[0].set_ylim(0,0.2);
    axes[0].set_ylabel('$T_+$', fontsize = myfontsize);
    axes[1].set_ylim(0.2,0.3);
    axes[1].set_ylabel('$\overline{p^2}(\\tilde{\\theta})$', fontsize = myfontsize);

    # show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    #plt.savefig('figs/model32_detailed.pdf');
    plt.show();


if False: # T+ at different Delta E by changing J12z

    Dvals = np.array([1/10]);#  0,1/1000,1/100,2/100]);
    Dval = 0;
    DeltaEvals = -2*Dvals;
    DeltaJvals = (DeltaEvals+2*Dval)/(-3/2); # this is Jz - Jx
    J12zvals = J12x + DeltaJvals;
    for Di in range(len(J12zvals)):
        J12z = J12zvals[Di];

        # iter over E, getting T
        logElims = -5,-1
        Evals = np.logspace(*logElims,199);
        Rvals = np.empty((len(Evals),len(source)), dtype = float);
        Tvals = np.empty((len(Evals),len(source)), dtype = float);
        for Evali in range(len(Evals)):

            # energy
            Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper
            
            # optical distances, N = 2 fixed
            N0 = 1; # N0 = N - 1
            ka = np.arccos((Energy)/(-2*tl));

            # construct hblocks
            hblocks = [];
            impis = [1,2];
            for j in range(4): # LL, imp 1, imp 2, RL
                # define all physical params
                JK1, JK2 = 0, 0;
                if(j == impis[0]): JK1 = JK;
                elif(j == impis[1]): JK2 = JK;
                params = J12x, J12y, J12z, Dval, Dval, 0, JK1, JK2;
                h1e, g2e = wfm.utils.h_cobalt_2q(params); # construct ham
                # construct h_SR (determinant basis)
                hSR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);            
                # transform to eigenbasis
                hSR_diag = wfm.utils.entangle(hSR, *pair);
                hblocks.append(np.copy(hSR_diag));
                if(verbose > 3 and Eval == Evals[0]):
                    print("\nJK1, JK2 = ",JK1, JK2);
                    print(" - ham:\n", np.real(hSR));
                    print(" - transformed ham:\n", np.real(hSR_diag));
                    print(" - DeltaE = ",-Dval*(2*1.5-1))

            # finish hblocks
            hblocks = np.array(hblocks);
            hblocks[1] += Vg*np.eye(len(source)); # Vg shift in SR
            hblocks[2] += Vg*np.eye(len(source));
            E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
            for hb in hblocks:
                hb += -E_shift*np.eye(np.shape(hblocks[0])[0]);
            print("Delta E / t = ", (hblocks[0][0,0] - hblocks[0][2,2])/tl)
            # hopping
            tnn = np.array([-tl*np.eye(len(source)),-tp*np.eye(len(source)),-tl*np.eye(len(source))]);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # get R, T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source, all_debug = False);
            Rvals[Evali] = Rdum;
            Tvals[Evali] = Tdum;
         
        # save data to .npy
        data = np.zeros((2+2*len(source),len(Evals)));
        data[0,0] = tl;
        data[0,1] = JK;
        data[1,:] = Evals;
        data[2:2+len(source),:] = Tvals.T;
        data[2+len(source):2+2*len(source),:] = Rvals.T;
        fname = "data/model32/J12z"+str(int(J12z*1000)/1000);
        print("Saving data to "+fname);
        np.save(fname, data);

#### plot
if False:
    num_subplots = 2
    fig, axes = plt.subplots(num_subplots, sharex = True);
    fig.set_size_inches(7/2,3*num_subplots/2);
    datadir = "data/model32/";
    datafs_neg = ["J12z0.01.npy","J12z0.008.npy","J12z-0.003.npy","J12z-0.016.npy"];
    datafs_pos = ["J12z0.01_copy.npy","J12z0.011.npy","J12z0.023.npy","J12z0.143.npy"];
    datafs = datafs_neg[:]; datafs.extend(datafs_pos);
    for fi in range(len(datafs)):
        xvals, Rvals, Tvals, totals = load_data(datadir+datafs[fi]);
        mymarkevery = (fi*10,50);

        if datafs[fi] in datafs_neg:
            # plot T+ for negative case
            axes[1].plot(xvals, Tvals[pair[0]], color=mycolors[fi], marker=mymarkers[fi], markevery=mymarkevery, linewidth = mylinewidth); 
            #axes[0].plot(xvals, totals, color="red");

        if datafs[fi] in datafs_pos:
            # plot T+ for positive case
            axes[0].plot(xvals, Tvals[pair[0]], color=mycolors[fi-len(datafs_neg)], marker=mymarkers[fi-len(datafs_neg)], markevery=(10*(fi-len(datafs_neg)),50), linewidth = mylinewidth); 
            #axes[1].plot(xvals, totals, color="red");

    # format
    for axi in range(len(axes)):
        axes[-1].set_xscale('log', subs = []);
        axes[-1].set_xlim(10**(-5),10**(-1));
        axes[-1].set_xticks([10**(-5),10**(-4),10**(-3),10**(-2),10**(-1)])
        axes[-1].set_xlabel('$K_i/t$', fontsize = myfontsize);
        axes[axi].set_ylim(0,0.16);
        axes[axi].set_yticks([0.0,0.08,0.16]);
        axes[axi].set_ylabel('$T_+$', fontsize = myfontsize);
        axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    plt.show();



