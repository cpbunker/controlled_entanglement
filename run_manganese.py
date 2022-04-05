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
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#### top level
#np.set_printoptions(precision = 4, suppress = True);
colors = ["darkblue","darkgreen","darkred", "darkmagenta"]
verbose = 5;

#### setup

# def particles and their single particle states
species = np.array([1,1,1]); # num of each species, which are one e, elec, spin-3/2, spin-3/2
#spec_strs = ["e","1","2"];
states = [[0,1],[2,3,4,5,6,7,8,9,10,11,12,13,14],[15,16,17,18,19,20,21,22,23,24,25,26,27]]; # e up, down, spin 1 mz, spin 2 mz
#state_strs = ["0.5_","-0.5_","1.5_","0.5_","-0.5_","-1.5_","1.5_","0.5_","-0.5_","-1.5_"];
dets_int = [[0,2,16],[0,3,15],[1,2,15]]; # total spin 5/2 subspace

# initialize source vector in down, 3/2, 3/2 state
sourcei = 2; # |down, 6, 6 >
source = np.zeros(len(dets_int));
source[sourcei] = 1;

# entangled pair
pair = (0,1); # |up, 1/2, 3/2 > and |up, 3/2, 1/2 >

# tight binding params in cm^-1
tl = 1; # lead hopping, in Hartree
tp = 1;  # hopping between imps
convert = 8
J = 0.8/convert
J12 = 0.025/convert;
Di = 0.22/convert;

# constructing the hamiltonian
def reduced_ham(params, S=6):
    J12, D1, D2, JK1, JK2 = params;

    ham = np.array([[S*S*D1+(S-1)*(S-1)*D2+S*(S-1)*J12+(JK1/2)*S+(JK2/2)*(S-1), S*J12, np.sqrt(2*S)*(JK2/2) ], # up, 6, 5
                    [S*J12, (S-1)*(S-1)*D1+S*S*D2+S*(S-1)*J12+(JK1/2)*S + (JK2/2)*(S-1), np.sqrt(2*S)*(JK1/2) ], # up, 5, 6
                    [np.sqrt(2*S)*(JK2/2), np.sqrt(2*S)*(JK1/2),S*S*D1+S*S*D2+S*S*J12+(-JK1/2)*S +(-JK2/2)*S]]); # down, 6, 6

    return ham;

            
#########################################################
#### generation

if True: # T vs E

    # main plot T vs E
    fig, ax = plt.subplots();
    dummyvals = [0]
    for dummyi in dummyvals:

        # iter over Energy, getting T
        Tvals, Rvals = [], [];
        rhoJalims = np.array([0.05,4.0]);
        Elims = J*J/(rhoJalims*rhoJalims*np.pi*np.pi*tl) - 2*tl;
        Evals = np.linspace(Elims[-1], Elims[0], 99); # switch
        for Ei in range(len(Evals)):

            # energy
            Energy = Evals[Ei]
            
            # optical distances, N = 2 fixed
            ka = np.arccos((Energy)/(-2*tl));
            Vg = Energy + 2*tl; # gate voltage
            kpa = np.arccos((Energy-Vg)/(-2*tl));

            # construct hblocks
            hblocks = [];
            impis = [1,2];
            for j in range(4): # LL, imp 1, imp 2, RL
                # define all physical params
                JK1, JK2 = 0, 0;
                if(j == impis[0]): JK1 = J;
                elif(j == impis[1]): JK2 = J;
                params = J12, Di, Di, JK1, JK2;
                # construct h_SR (determinant basis)
                hSR = reduced_ham(params); 
                # transform to eigenbasis
                hSR_diag = wfm.utils.entangle(hSR, *pair);
                hblocks.append(np.copy(hSR_diag));

            # finish hblocks
            hblocks = np.array(hblocks);
            hblocks[1] += Vg*np.eye(len(source)); # Vg shift in SR
            hblocks[2] += Vg*np.eye(len(source));
            E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
            for hb in hblocks:
                hb += -E_shift*np.eye(np.shape(hblocks[0])[0]);

            # hopping
            tnn = np.array([-tl*np.eye(len(source)),-tp*np.eye(len(source)),-tl*np.eye(len(source))]);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # T
            Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, verbose = 0));
            Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, reflect = True));
         
        # plot T vs E
        Tvals, Rvals = np.array(Tvals), np.array(Rvals);
        ax.plot(Evals, Tvals[:,sourcei], label = "$|i\,>$", color = "black", linewidth = 2);
        ax.plot(Evals, Tvals[:,pair[0]], label = "$|+>$", color = colors[dummyi], linestyle = "dashed", linewidth = 2);
        ax.plot(Evals, Tvals[:,0]+Tvals[:,1]+Tvals[:,2]+Rvals[:,0]+Rvals[:,1]+Rvals[:,2], color = "red")
        
    # now do T vs rhoJa inset plot
    if True:
        axins = inset_axes(ax, width="50%", height="50%");
    else:
        dummyvals = [];
    for dummyi in dummyvals:

        # iter over rhoJ, getting T
        Tvals, Rvals = [], [];

        rhoJavals = np.linspace(rhoJalims[-1], rhoJalims[0], len(Evals)); # switched !
        for rhoi in range(len(rhoJavals)):

            # energy
            Energy = J*J/(rhoJavals[rhoi]*rhoJavals[rhoi]*np.pi*np.pi*tl) - 2*tl;

            # optical distances, N = 2 fixed
            ka = np.arccos((Energy)/(-2*tl));
            Vg = Energy + 2*tl; # gate voltage
            kpa = np.arccos((Energy-Vg)/(-2*tl));

            # construct hblocks
            hblocks = [];
            impis = [1,2];
            for j in range(4): # LL, imp 1, imp 2, RL
                # define all physical params
                JK1, JK2 = 0, 0;
                if(j == impis[0]): JK1 = J;
                elif(j == impis[1]): JK2 = J;
                params = J12, Di, Di, JK1, JK2;
                # construct h_SR (determinant basis)
                hSR = reduced_ham(params);
                # transform to eigenbasis
                hSR_diag = wfm.utils.entangle(hSR, *pair);
                hblocks.append(np.copy(hSR_diag));

            # finish hblocks
            hblocks = np.array(hblocks);
            hblocks[1] += Vg*np.eye(len(source)); # Vg shift in SR
            hblocks[2] += Vg*np.eye(len(source));
            E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
            for hb in hblocks:
                hb += -E_shift*np.eye(np.shape(hblocks[0])[0]);

            # hopping
            tnn = np.array([-tl*np.eye(len(source)),-tp*np.eye(len(source)),-tl*np.eye(len(source))]);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # T
            Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, verbose = 0));
            Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, reflect = True));

        # plot T vs rhoJa in inset
        Tvals, Rvals = np.array(Tvals), np.array(Rvals);
        axins.plot(rhoJavals, Tvals[:,pair[0]], label = "$|+>$", color = colors[dummyi], linestyle = "dashed", linewidth = 2);
        axins.plot(rhoJavals, Tvals[:,sourcei], label = "", color = "black", linestyle = "solid", linewidth = 2);
        axins.plot(rhoJavals, Tvals[:,pair[0]], label = "$|+>$", color = colors[dummyi], linestyle = "dashed", linewidth = 2);

        #axins.set_xlim(0,4);
        #axins.set_xticks([0,2,4]);
        axins.set_xlabel("$J/\pi \sqrt{t(E+2t)}$", fontsize = "x-large");
        #axins.set_ylim(0,0.25);
        #axins.set_yticks([0,0.25]);
        axins.set_ylabel("$T$", fontsize = "x-large");

    # format and show
    #ax.set_xlim(-2,-1.6);
    #ax.set_xticks([-2,-1.8,-1.6]);
    ax.set_xlabel("$E/t$", fontsize = "x-large");
    #ax.set_ylim(0,0.25);
    #ax.set_yticks([0,0.25]);
    ax.set_ylabel("$T$", fontsize = "x-large");
    plt.show();

