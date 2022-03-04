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
plt.style.use("seaborn-dark-palette");
verbose = 5;

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

# tight binding params
tl = 1; # lead hopping, in Hartree
tp = 1;  # hopping between imps
J = 0.1;
D = 0.05;

            
#########################################################
#### generation

if True: # fig 6 ie T vs rho J a

    # plot at diff DeltaK
    for dummy in [1]:
        
        # 2 site SR
        fig, ax = plt.subplots();
        hblocks = [];
        impis = [1,2];
        for j in range(4): # LL, imp 1, imp 2, RL

            # define all physical params
            JK1, JK2 = 0, 0;
            if(j == impis[0]): JK1 = J;
            elif(j == impis[1]): JK2 = J;
            params = 0, 0, 0, D, D, 0, JK1, JK2;
            h1e, g2e = wfm.utils.h_dimer_2q(params); # construct ham

            # construct h_SR (determinant basis)
            hSR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);
                
            # transform to eigenbasis
            hSR_diag = wfm.utils.entangle(hSR, *pair);
            
            if(verbose):
                print("\nJK1, JK2 = ",JK1, JK2);
                print(" - ham:\n", np.real(hSR));
                print(" - transformed ham:\n", np.real(hSR_diag));
            
            # add to blocks list
            hblocks.append(np.copy(hSR_diag));

        # finish hblocks
        hblocks.append(hSR_JK0_diag);
        hblocks = np.array(hblocks);
        E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
        for hb in hblocks:
            hb += -E_shift*np.eye(np.shape(hblocks[0])[0]);

        # hopping
        tnn = np.array([-th*np.eye(len(source)),-tp*np.eye(len(source)),-th*np.eye(len(source))]);
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

        # iter over rhoJ, getting T
        Tvals, Rvals = [], [];
        rhoJvals = np.linspace(0.01,1.0,99);
        Erhovals = DO*DO/(rhoJvals*rhoJvals*np.pi*np.pi*tl); # measured from bottom of band
        for rhoi in range(len(rhoJvals)):

            # energy
            rhoJa = rhoJvals[rhoi];
            Energy = Erhovals[rhoi] - 2*tl; # measure from mu
            k_rho = np.arccos(Energy/(-2*tl));
            if(False):
                print("\nCiccarello inputs");
                print("E/t, JK/t, Erho/JK1 = ",Energy/tl + 2, JK/tl, (Energy + 2*tl)/JK);
                print("ka = ",k_rho);
                print("rhoJa = ", abs(JK/np.pi)/np.sqrt((Energy+2*tl)*tl));

            # T (Energy from 0)
            Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, verbose = 0));
            Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, reflect = True));
            
        # plot
        Tvals, Rvals = np.array(Tvals), np.array(Rvals);
        ax.plot(rhoJvals, Tvals[:,sourcei], label = "$|i\,>$", color = "black", linewidth = 2);
        ax.plot(rhoJvals, Tvals[:,pair[0]], label = "$|+>$", color = "black", linestyle = "dashed", linewidth = 2);
        ax.plot(rhoJvals, Tvals[:,pair[1]], label = "$|->$", color = "black", linestyle = "dashdot", linewidth = 2);
        ax.plot(rhoJvals, Tvals[:,0]+Tvals[:,1]+Tvals[:,2]+Rvals[:,0]+Rvals[:,1]+Rvals[:,2], color = "red")

        # inset
        if False:
            Evals = JK*JK/(rhoJvals*rhoJvals*np.pi*np.pi*tl)-2*tl;
            axins = inset_axes(ax, width="50%", height="50%");
            axins.plot(Evals,Tvals[:,pair[0]], color = "darkgreen", linestyle = "dashed", linewidth = 2); # + state
            axins.set_xlabel("$E/t$", fontsize = "x-large");
            axins.set_ylim(0,0.2);
            axins.set_yticks([0,0.2]);

        # format and show
        ax.set_xlim(min(rhoJvals),max(rhoJvals));
        ax.set_xticks([0,1]);
        ax.set_xlabel("$D_O/\pi \sqrt{tE}$", fontsize = "x-large");
        ax.set_ylim(0,1.0);
        #ax.set_yticks([0,0.2]);
        ax.set_ylabel("$T$", fontsize = "x-large");
        #plt.legend();
        plt.show();

    # end sweep over JK
    raise(Exception);


