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

#### setup

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

# constructing the hamiltonian
def reduced_ham(params, S):
    D1, D2, J12, JK1, JK2 = params;
    assert(D1 == D2);
    ham = np.array([[S*S*D1+(S-1)*(S-1)*D2+S*(S-1)*J12+(JK1/2)*S+(JK2/2)*(S-1), S*J12, np.sqrt(2*S)*(JK2/2) ], # up, s, s-1
                    [S*J12, (S-1)*(S-1)*D1+S*S*D2+S*(S-1)*J12+(JK1/2)*S + (JK2/2)*(S-1), np.sqrt(2*S)*(JK1/2) ], # up, s-1, s
                    [np.sqrt(2*S)*(JK2/2), np.sqrt(2*S)*(JK1/2),S*S*D1+S*S*D2+S*S*J12+(-JK1/2)*S +(-JK2/2)*S]], # down, s, s
                   dtype = complex);

    return ham;
            
#########################################################
#### effects of J

if True:
    Jvals = np.array([-0.01,-0.1]);
    num_plots = len(Jvals);
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);

    # params
    myspinS = 3/2;
    J12 = 0.0;
    Esplitvals = np.array([-0.05,-0.01,0.0,0.01,0.05]);
    Dvals = Esplitvals/(1-2*myspinS);
    for Dvali in range(len(Dvals)):
        Dval = Dvals[Dvali];

        # see effects of J
        for Jvali in range(len(Jvals)):
            Jval = Jvals[Jvali];

            # iter over E, getting T
            logElims = -4,0
            Evals = np.logspace(*logElims,myxvals, dtype = complex);
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
                    if(j == impis[0]): JK1 = Jval;
                    elif(j == impis[1]): JK2 = Jval;
                    params = Dval, Dval, J12, JK1, JK2;
                    # construct h_SR (determinant basis)
                    hSR = reduced_ham(params,S=myspinS);           
                    # transform to eigenbasis
                    hSR_diag = wfm.utils.entangle(hSR, *pair);
                    hblocks.append(np.copy(hSR_diag));
                    if(verbose > 3 and Eval == Evals[0]):
                        print("\nJK1, JK2 = ",JK1, JK2);
                        print(" - ham:\n", hSR);
                        print(" - transformed ham:\n", np.real(hSR_diag));

                # finish hblocks
                hblocks = np.array(hblocks);
                E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
                for hb in hblocks:
                    hb += -E_shift*np.eye(np.shape(hblocks[0])[0]);
                if(verbose > 3 and Eval == Evals[0]): print("Delta E / t = ", (hblocks[0][0,0] - hblocks[0][2,2])/tl);

                # hopping
                tnn = np.array([-tl*np.eye(len(source)),-tp*np.eye(len(source)),-tl*np.eye(len(source))]);
                tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

                # get R, T coefs
                Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source, all_debug = False);
                Rvals[Evali] = Rdum;
                Tvals[Evali] = Tdum;

            # plot p2 vs E
            axes[Jvali].plot(Evals,np.sqrt(Tvals[:,sourcei]*Tvals[:,pair[0]]),color = mycolors[Dvali], marker = mymarkers[Dvali], markevery = (40,40), linewidth = mylinewidth);

    # format
    axes[0].set_ylabel('$\overline{p^2}$', fontsize = myfontsize );
    axes[1].set_ylabel('$\overline{p^2}$', fontsize = myfontsize );
    
    # show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    plt.savefig("figs/Jlimit_MM.pdf");
    plt.show();


