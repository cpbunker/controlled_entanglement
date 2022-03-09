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

            
#########################################################
#### generation

if False: # fig 6 ie T vs rho J a
    
    fig, ax = plt.subplots();
    axins = inset_axes(ax, width="50%", height="50%");
    Dvals = J*np.array([0,0.1,0.2,0.4])
    for Di in range(len(Dvals)):
        D = Dvals[Di];

        # iter over rhoJ, getting T
        Tvals, Rvals = [], [];
        rhoJavals = np.linspace(0.05,4.0,19);
        for rhoi in range(len(rhoJavals)):

            # energy
            rhoJa = rhoJavals[rhoi];
            E_rho = J*J/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                                    # this E is measured from bottom of band !!!
            Energy = E_rho - 2*tl; # regular energy
            
            # optical distances, N = 2 fixed
            ka = np.arccos((Energy)/(-2*tl));
            Vg = Energy + 2*tl; # gate voltage
            kpa = np.arccos((Energy-Vg)/(-2*tl));
            print(ka, kpa, Vg)

            # construct hblocks
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
                hblocks.append(np.copy(hSR_diag));
                if(verbose > 3 and rhoJa == rhoJavals[0]):
                    print("\nJK1, JK2 = ",JK1, JK2);
                    print(" - ham:\n", np.real(hSR));
                    print(" - transformed ham:\n", np.real(hSR_diag));

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
         
        # plot
        Tvals, Rvals = np.array(Tvals), np.array(Rvals);
        #ax.plot(rhoJavals, Tvals[:,sourcei], label = "$|i\,>$", color = "black", linewidth = 2);
        ax.plot(rhoJavals, Tvals[:,pair[0]], label = "$|+>$", color = colors[Di], linestyle = "dashed", linewidth = 2);
        #ax.plot(rhoJavals, Tvals[:,pair[1]], label = "$|->$", color = "black", linestyle = "dashdot", linewidth = 2);
        ax.plot(rhoJavals, Tvals[:,0]+Tvals[:,1]+Tvals[:,2]+Rvals[:,0]+Rvals[:,1]+Rvals[:,2], color = "red")

        # inset
        if True:
            Evals = J*J/(rhoJavals*rhoJavals*np.pi*np.pi*tl) - 2*tl;
            axins.plot(Evals,Tvals[:,pair[0]], color = colors[Di], linestyle = "dashed", linewidth = 2); # + state
            axins.set_xlim(min(Evals)-0.01,max(Evals));
            axins.set_xticks([-2,-1.6]);
            axins.set_xlabel("$E/t$", fontsize = "x-large");
            axins.set_ylim(0,0.15);
            axins.set_yticks([0,0.15]);
            axins.set_ylabel("$T$");

    # format and show
    ax.set_xlim(min(rhoJavals),max(rhoJavals));
    ax.set_xticks([0,1,2,3,4]);
    ax.set_xlabel("$J/\pi \sqrt{tE_b}$", fontsize = "x-large");
    ax.set_ylim(0,0.15);
    ax.set_yticks([0,0.15]);
    ax.set_ylabel("$T$", fontsize = "x-large");
    plt.show();

if False: # T vs E
    
    fig, ax = plt.subplots();
    axins = inset_axes(ax, width="50%", height="50%");
    Dvals =-J*np.array([0.1,0.2])
    for Di in range(len(Dvals)):
        D = Dvals[Di];

        # iter over rhoJ, getting T
        Tvals, Rvals = [], [];
        Evals = np.linspace(-2,-2+0.1,19)
        for rhoi in range(len(Evals)):

            # energy
            Energy = Evals[rhoi]
            
            # optical distances, N = 2 fixed
            ka = np.arccos((Energy)/(-2*tl));
            Vg = Energy + 2*tl; # gate voltage
            kpa = np.arccos((Energy-Vg)/(-2*tl));
            print(ka, kpa, Vg)

            # construct hblocks
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
         
        # plot
        Tvals, Rvals = np.array(Tvals), np.array(Rvals);
        #ax.plot(rhoJavals, Tvals[:,sourcei], label = "$|i\,>$", color = "black", linewidth = 2);
        ax.plot(Evals, Tvals[:,pair[0]], label = "$|+>$", color = colors[Di], linestyle = "dashed", linewidth = 2);
        #ax.plot(rhoJavals, Tvals[:,pair[1]], label = "$|->$", color = "black", linestyle = "dashdot", linewidth = 2);
        ax.plot(Evals, Tvals[:,0]+Tvals[:,1]+Tvals[:,2]+Rvals[:,0]+Rvals[:,1]+Rvals[:,2], color = "red")

    # format and show
    ax.set_xlim(min(Evals),max(Evals));
    ax.set_xticks([0,1,2,3,4]);
    ax.set_xlabel("$J/\pi \sqrt{tE_b}$", fontsize = "x-large");
    ax.set_ylim(0,0.15);
    ax.set_yticks([0,0.15]);
    ax.set_ylabel("$T$", fontsize = "x-large");
    #plt.legend();
    plt.show();


            
#########################################################
#### symmetry breaking

if True:
    
    DeltaVvals = -J*np.array([20]);
    for DeltaV in DeltaVvals:

        # symmetry breaking
        D = 0.5*J;
        DeltaD = 0.1*J;
        D1 = D + DeltaD/2;
        D2 = D - DeltaD/2;
        del D;

        # iter over rhoJ, getting T
        Tvals, Rvals = [], [];
        rhoJavals = np.linspace(0.05,0.1,19);
        for rhoi in range(len(rhoJavals)):

            # energy
            rhoJa = rhoJavals[rhoi];
            E_rho = J*J/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                                    # this E is measured from bottom of band !!!
            Energy = E_rho - 2*tl; # regular energy

            # JK=0 matrix for ref
            h1e_0, g2e_0 = wfm.utils.h_dimer_2q((0,0,0,0, 0, 0, 1, 1));
            hSR_0 = fci_mod.single_to_det(h1e_0, g2e_0, species, states, dets_interest = dets52);
            hSR_0 = wfm.utils.entangle(hSR_0, *pair);
            print(hSR_0);
            assert False;
            _, Udiag = np.linalg.eigh(hSR_0);
            del h1e_0, g2e_0, hSR_0;

            # construct hblocks
            hblocks = [];
            impis = [1,2];
            for j in range(4): # LL, imp 1, imp 2, RL
                # define all physical params
                JK1, JK2 = 0, 0;
                if(j == impis[0]): JK1 = J;
                elif(j == impis[1]): JK2 = J;
                params = 0, 0, 0, D1, D2, 0, JK1, JK2;
                h1e, g2e = wfm.utils.h_dimer_2q(params); # construct ham
                # construct h_SR (determinant basis)
                hSR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);
                # Vg splitting
                if(j == impis[0]): hSR += (DeltaV/2)*np.eye(len(source));    
                if(j == impis[1]): hSR += (-DeltaV/2)*np.eye(len(source));     
                # transform to eigenbasis
                hSR_ent = wfm.utils.entangle(hSR, *pair);
                hSR_diag = np.dot( np.linalg.inv(Udiag), np.dot(hSR_ent, Udiag));
                hblocks.append(np.copy(hSR_diag));
                if(verbose > 3 and rhoJa == rhoJavals[0]):
                    print("\nJK1, JK2 = ",JK1, JK2);
                    print(" - ham:\n", np.real(hSR));
                    print(" - transformed ham:\n", np.real(hSR_diag));

            # finish hblocks
            hblocks = np.array(hblocks);
            E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
            for hb in hblocks:
                hb += -E_shift*np.eye(np.shape(hblocks[0])[0]);

            # hopping
            tnn = np.array([-tl*np.eye(len(source)),-tp*np.eye(len(source)),-tl*np.eye(len(source))]);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # T
            Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, verbose = 0));
            Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, reflect = True));
         
        # plot
        Tvals, Rvals = np.array(Tvals), np.array(Rvals);
        fig, ax = plt.subplots();
        axins = inset_axes(ax, width="50%", height="50%");
        ax.plot(rhoJavals, Tvals[:,sourcei], label = "$|i\,>$", color = "black", linewidth = 2);
        ax.plot(rhoJavals, Tvals[:,pair[0]], label = "$|+>$", color = "black", linestyle = "dashed", linewidth = 2);
        ax.plot(rhoJavals, Tvals[:,pair[1]], label = "$|->$", color = "black", linestyle = "dashdot", linewidth = 2);
        ax.plot(rhoJavals, Tvals[:,0]+Tvals[:,1]+Tvals[:,2]+Rvals[:,0]+Rvals[:,1]+Rvals[:,2], color = "red")

        # inset
        if True:
            Evals = J*J/(rhoJavals*rhoJavals*np.pi*np.pi*tl) - 2*tl;
            axins.plot(Evals,Tvals[:,pair[0]], color = "black", linestyle = "dashed", linewidth = 2); # + state
            #axins.set_xlim(min(Evals)-0.01,max(Evals));
            #axins.set_xticks([-2,-1.6]);
            axins.set_xlabel("$E/t$", fontsize = "x-large");
            axins.set_ylim(0,0.15);
            axins.set_yticks([0,0.15]);
            axins.set_ylabel("$T$");

        # format and show
        #ax.set_xlim(min(rhoJavals),max(rhoJavals));
        #ax.set_xticks([0,1,2,3,4]);
        ax.set_xlabel("$J/\pi \sqrt{tE_b}$", fontsize = "x-large");
        ax.set_ylim(0,0.15);
        ax.set_yticks([0,0.15]);
        ax.set_ylabel("$T$", fontsize = "x-large");
        plt.show();


