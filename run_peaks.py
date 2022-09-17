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
def mymarkevery(fname,yvalues):
    if '-' in fname or '0.0.npy' in fname:
        return (40,40);
    else:
        return [np.argmax(yvalues)];
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

#### data

peaks12 = np.array([ [ 0.00 , 0.222490 , 0.298361 ]]);
peaks32 = np.array([
     [-0.12 , 0.061207 , 0.188055 ],
     [-0.08 , 0.071040 , 0.199658 ],
     [-0.05 , 0.082684 , 0.211454 ],
     [-0.01 , 0.113272 , 0.234906 ],
     [ 0.00 , 0.123194 , 0.242528 ],
     [ 0.01 , 0.123384 , 0.249863 ],
     [ 0.05 , 0.100259 , 0.257826 ],
     [ 0.08 , 0.087848 , 0.252942 ],
     [ 0.12 , 0.077284 , 0.244709 ] ]);
peaks6 = np.array([
     [-0.060, 0.043160, 0.165020 ],
     [-0.030, 0.043117, 0.166109 ],
     [-0.003, 0.042759, 0.167005 ],
     [ 0.003, 0.042640, 0.167187 ],
     [ 0.030, 0.041956, 0.167935 ],
     [ 0.060, 0.040989, 0.168579 ] ]); # [ 0.000, 0.042703, 0.167096 ],

#### plot T+ and p2 vs Delta E
if True:
    num_plots = 2;
    fig, axes = plt.subplots(num_plots, sharex=True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);
    indE, indT, indp = 0,1,2;


    # plot T+
    # for s=1/2
    axes[0].axhline(peaks12[0,indT], color=mycolors[0]);
    #for s=3/2
    axes[0].scatter(peaks32[:,indE], peaks32[:,indT], color=mycolors[1], marker=mymarkers[1], linewidth = mylinewidth);
    #for s=6
    axes[0].scatter(peaks6[:,indE], peaks6[:,indT], color=mycolors[2], marker=mymarkers[2], linewidth = mylinewidth);
       
    # plot analytical FOM
    # for s=1/2
    axes[1].axhline(peaks12[0,indp], color=mycolors[0]);
    # for s=3/2
    axes[1].scatter(peaks32[:,indE], peaks32[:,indp], color=mycolors[1], marker=mymarkers[1], linewidth = mylinewidth);
    # for s=6
    axes[1].scatter(peaks6[:,indE], peaks6[:,indp], color=mycolors[2], marker=mymarkers[2], linewidth = mylinewidth);

    # format
    axes[0].set_ylim(0,0.4);
    axes[0].set_ylabel('$T_+$', fontsize = myfontsize);
    axes[1].set_ylim(0.0,0.4);
    axes[1].set_ylabel('$\overline{p^2}(\\tilde{\\theta})$', fontsize = myfontsize);

    # show
    axes[-1].set_xlabel('$\Delta E/t$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
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



