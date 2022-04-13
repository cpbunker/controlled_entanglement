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
colors = ["black","darkblue","darkgreen","darkred", "darkmagenta","darkgray","darkcyan"]
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

# tight binding params, in meV
if False:
    tl = 100; # lead hopping
    tp = 100;  # hopping between imps
    th = tl/5;
    Ucharge = 1000;
    JK = 8*th*th/Ucharge;
    J12 = JK/50; # rough cobalt order of magnitude
    D0 = JK/10;
    print("\n>>>params, in meV:\n",tl, tp, JK,J12, D0); 
    del th, Ucharge;
    Ha2meV = 27.211386*1000;
    tl, tp, JK, J12, D0 = tl/Ha2meV, tp/Ha2meV, JK/Ha2meV, J12/Ha2meV, D0/Ha2meV; # convert all to Ha

else:
    tl = 1.0;
    tp = 1.0;
    JK = 0.1;
    J12 = JK/10;
            
#########################################################
#### generation

if False: # T/Tvs rho J a at diff D
    
    fig, ax = plt.subplots();
    Dvals = JK*np.array([-1/100,0,1/100,1/10,1]);
    for Di in range(len(Dvals)):
        D = Dvals[Di];

        # iter over rhoJ, getting T
        Tvals, Rvals = [], [];
        rhoJavals = np.linspace(0.05,4.0,9);
        for rhoi in range(len(rhoJavals)):

            # energy
            rhoJa = rhoJavals[rhoi];
            E_rho = JK*JK/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                                    # this E is measured from bottom of band !!!
            Energy = E_rho - 2*tl; # regular energy
            
            # optical distances, N = 2 fixed
            ka = np.arccos((Energy)/(-2*tl));
            Vg = Energy + 2*tl; # gate voltage

            # construct hblocks
            hblocks = [];
            impis = [1,2];
            for j in range(4): # LL, imp 1, imp 2, RL
                # define all physical params
                JK1, JK2 = 0, 0;
                if(j == impis[0]): JK1 = JK;
                elif(j == impis[1]): JK2 = JK;
                params = J12, J12, J12, D, D, 0, JK1, JK2;
                h1e, g2e = wfm.utils.h_cobalt_2q(params); # construct ham
                # construct h_SR (determinant basis)
                hSR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);            
                # transform to eigenbasis
                hSR_diag = wfm.utils.entangle(hSR, *pair);
                hblocks.append(np.copy(hSR_diag));
                if(verbose > 5 and rhoJa == rhoJavals[0]):
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
            print("Delta E / J = ", (hblocks[0][0,0] - hblocks[0][2,2])/JK)
            # hopping
            tnn = np.array([-tl*np.eye(len(source)),-tp*np.eye(len(source)),-tl*np.eye(len(source))]);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # T
            Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, verbose = 0));
            Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, reflect = True));
         
        # plot
        Tvals, Rvals = np.array(Tvals), np.array(Rvals);
        #ax.plot(rhoJavals, Tvals[:,sourcei], label = "$|i\,>$", color = colors[Di], linestyle = "solid", linewidth = 2);
        #ax.plot(rhoJavals, Tvals[:,pair[0]], label = "$|+>$", color = colors[Di], linestyle = "dashed", linewidth = 2);
        ax.plot(rhoJavals, Tvals[:,pair[0]]/Tvals[:,sourcei], label = "", color = colors[Di], linestyle = "solid", linewidth = 2);
        #ax.plot(rhoJavals, Tvals[:,0]+Tvals[:,1]+Tvals[:,2]+Rvals[:,0]+Rvals[:,1]+Rvals[:,2], color = "red");

    # format and show
    ax.set_xlim(0,4);
    ax.set_xticks([0,2,4]);
    ax.set_xlabel("$J/\pi \sqrt{t(E+2t)}$", fontsize = "x-large");
    ax.set_ylim(0,4);
    ax.set_yticks([0,2,4]);
    ax.set_ylabel("$T_+/T_0$", fontsize = "x-large");
    plt.show();

    # now do T vs E inset plot
    if False:
        axins = inset_axes(ax, width="50%", height="50%");

        rhoJalims = np.array([rhoJavals[0], rhoJavals[-1]]);
        Elims = J*J/(rhoJalims*rhoJalims*np.pi*np.pi*tl) - 2*tl;
        Evals = np.linspace(Elims[-1], Elims[0], len(rhoJavals)); # switched !
        #print(">>>", rhoJalims, "\n>>>", Evals); assert False;
        D = Dvals[0];

        # iter over E, getting T
        Tvals, Rvals = [], [];
        for Ei in range(len(Evals)):

            # energy
            Energy = Evals[Ei]
            
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
                h1e, g2e = wfm.utils.h_cobalt_2q(params); # construct ham
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
        axins.plot(Evals,Tvals[:,pair[0]], color = colors[Di], linestyle = "solid", linewidth = 2); # + state
    
        axins.set_xlim(-2,-1.6);
        axins.set_xticks([-2,-1.8,-1.6]);
        axins.set_xlabel("$E/t$", fontsize = "x-large");
        axins.set_ylim(0,0.15);
        axins.set_yticks([0,0.15]);
        axins.set_ylabel("$T_+$", fontsize = "x-large");

if False: # T/T vs rho J a at diff J12x
    
    fig, ax = plt.subplots();
    J12vals = J12*np.array([2,2.2,2.4,-100]);
    for Di in range(len(J12vals)):
        J12x = J12vals[Di];
        J12z = J12;

        # iter over rhoJ, getting T
        Tvals, Rvals = [], [];
        rhoJavals = np.linspace(0.05,4.0,9);
        for rhoi in range(len(rhoJavals)):

            # energy
            rhoJa = rhoJavals[rhoi];
            E_rho = JK*JK/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                                    # this E is measured from bottom of band !!!
            Energy = E_rho - 2*tl; # regular energy
            
            # optical distances, N = 2 fixed
            ka = np.arccos((Energy)/(-2*tl));
            Vg = Energy + 2*tl; # gate voltage

            # construct hblocks
            hblocks = [];
            impis = [1,2];
            for j in range(4): # LL, imp 1, imp 2, RL
                # define all physical params
                JK1, JK2 = 0, 0;
                if(j == impis[0]): JK1 = JK;
                elif(j == impis[1]): JK2 = JK;
                params = J12x, J12x, J12z, D0, D0, 0, JK1, JK2;
                h1e, g2e = wfm.utils.h_cobalt_2q(params); # construct ham
                # construct h_SR (determinant basis)
                hSR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);            
                # transform to eigenbasis
                hSR_diag = wfm.utils.entangle(hSR, *pair);
                hblocks.append(np.copy(hSR_diag));
                if(verbose > 5 and rhoJa == rhoJavals[0]):
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
            if(rhoi == 0): print("E_+ - E_sigma0 : ",hblocks[0][0,0] - hblocks[0][2,2]);
            
            # hopping
            tnn = np.array([-tl*np.eye(len(source)),-tp*np.eye(len(source)),-tl*np.eye(len(source))]);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # T
            Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, verbose = 0));
            Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, reflect = True));
         
        # plot
        Tvals, Rvals = np.array(Tvals), np.array(Rvals);
        #ax.plot(rhoJavals, Tvals[:,sourcei], label = "$|i\,>$", color = colors[Di], linestyle = "solid", linewidth = 2);
        #ax.plot(rhoJavals, Tvals[:,pair[0]], label = "$|+>$", color = colors[Di], linestyle = "dashed", linewidth = 2);
        ax.plot(rhoJavals, Tvals[:,pair[0]]/Tvals[:,sourcei], label = "", color = colors[Di], linestyle = "solid", linewidth = 2);
        #ax.plot(rhoJavals, Tvals[:,0]+Tvals[:,1]+Tvals[:,2]+Rvals[:,0]+Rvals[:,1]+Rvals[:,2], color = "red");

    # format and show
    ax.set_xlim(0,4);
    ax.set_xticks([0,2,4]);
    ax.set_xlabel("$J/\pi \sqrt{t(E+2t)}$", fontsize = "x-large");
    ax.set_ylim(0,4);
    ax.set_yticks([0,2,4]);
    ax.set_ylabel("$T_+/T_{\sigma_0}$", fontsize = "x-large");
    plt.show();
    

if True: # T vs E

    # main plot T vs E
    fig, ax = plt.subplots();
    Dvals = JK*np.array([-1/100,0,1/100,1/10,1]);
    for Di in range(len(Dvals)):
        D = Dvals[Di];

        # iter over Energy, getting T
        Tvals, Rvals = [], [];
        rhoJalims = np.array([0.05,4.0]);
        Elims = JK*JK/(rhoJalims*rhoJalims*np.pi*np.pi*tl) - 2*tl;
        Evals = np.linspace(Elims[-1], Elims[0], 499); # switched !
        for Ei in range(len(Evals)):

            # energy
            Energy = Evals[Ei]
            
            # optical distances, N = 2 fixed
            ka = np.arccos((Energy)/(-2*tl));
            Vg = Energy + 2*tl; # gate voltage

            # construct hblocks
            hblocks = [];
            impis = [1,2];
            for j in range(4): # LL, imp 1, imp 2, RL
                # define all physical params
                JK1, JK2 = 0, 0;
                if(j == impis[0]): JK1 = JK;
                elif(j == impis[1]): JK2 = JK;
                params = J12, J12, J12, D, D, 0, JK1, JK2;
                h1e, g2e = wfm.utils.h_cobalt_2q(params); # construct ham
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
            print("Delta E / J = ", (hblocks[0][0,0] - hblocks[0][2,2])/JK)
            # hopping
            tnn = np.array([-tl*np.eye(len(source)),-tp*np.eye(len(source)),-tl*np.eye(len(source))]);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # T
            Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, verbose = 0));
            Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, reflect = True));
         
        # plot T vs E
        Tvals, Rvals = np.array(Tvals), np.array(Rvals);
        #ax.plot(rhoJavals, Tvals[:,sourcei], label = "$|i\,>$", color = "black", linewidth = 2);
        ax.plot(Evals, Tvals[:,pair[0]], label = "$|+>$", color = colors[Di], linestyle = "solid", linewidth = 2);
        #ax.plot(rhoJavals, Tvals[:,pair[1]], label = "$|->$", color = "black", linestyle = "dashdot", linewidth = 2);
        ax.plot(Evals, Tvals[:,0]+Tvals[:,1]+Tvals[:,2]+Rvals[:,0]+Rvals[:,1]+Rvals[:,2], color = "red")


    # format and show
    ax.set_xlim(-2,-1.6);
    ax.set_xticks([-2,-1.8,-1.6]);
    ax.set_xlabel("$E/t$", fontsize = "x-large");
    ax.set_ylim(0,0.2);
    ax.set_yticks([0,0.1,0.2]);
    ax.set_ylabel("$T_+$", fontsize = "x-large");
    plt.show();


            
#########################################################
#### symmetry breaking

if False:

    fig, ax = plt.subplots();
    axins = inset_axes(ax, width="50%", height="50%");
    
    DeltaVvals = -J*np.array([0]);
    DeltaV = 0
    # symmetry breaking
    D = 0.5*J;
    DeltaD = 0.1*J;
    D1 = D + DeltaD/2;
    D2 = D - DeltaD/2;
    #J12 = J # -2*D/3;
    J12vals = DeltaD*np.array([0.2,1,5]);
    colors = ["darkblue","darkgreen","darkred","darkmagenta"];
    for J12i in range(len(J12vals)):
        J12 = J12vals[J12i];

        # iter over rhoJ, getting T
        Tvals, Rvals = [], [];
        rhoJavals = np.linspace(0.05,2.0,99);
        for rhoi in range(len(rhoJavals)):

            # energy
            rhoJa = rhoJavals[rhoi];
            E_rho = J*J/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                                    # this E is measured from bottom of band !!!
            Energy = E_rho - 2*tl; # regular energy
            Vg = Energy + 2*tl; # gate voltage

            # JK=0 matrix for ref
            h1e_0, g2e_0 = wfm.utils.h_dimer_2q((J12,J12,J12,D1,D2, 0, 0, 0));
            hSR_0 = fci_mod.single_to_det(h1e_0, g2e_0, species, states, dets_interest = dets52);
            hSR_0 = wfm.utils.entangle(hSR_0, *pair);
            #print(hSR_0);
            #assert False;
            _, Udiag = np.linalg.eigh(hSR_0);
            #del h1e_0, g2e_0, hSR_0;

            # von Neumann entropy
            if False:
                dummy = 2*DeltaD/(3*J12)
                alpha = (1+np.sqrt(1+dummy*dummy));
                beta = dummy;
                aval = (alpha + beta)/np.sqrt(2)/np.sqrt(alpha*np.conj(alpha) + beta*np.conj(beta));
                bval = (alpha - beta)/np.sqrt(2)/np.sqrt(alpha*np.conj(alpha) + beta*np.conj(beta));
                # project onto imp 1
                rho1 = np.array([[aval*np.conj(aval),0],[0,bval*np.conj(bval)]]);
                rho1_log2 = np.diagflat(np.log2(np.diagonal(rho1))); # since it is diagonal, log operation can be vectorized
                VNE = -np.trace(np.dot(rho1,rho1_log2));
                print(">>> VNE = ",VNE);
                #assert False 

            # construct hblocks
            hblocks = [];
            impis = [1,2];
            for j in range(4): # LL, imp 1, imp 2, RL
                # define all physical params
                JK1, JK2 = 0, 0;
                if(j == impis[0]): JK1 = J;
                elif(j == impis[1]): JK2 = J;
                params = J12, J12, J12, D1, D2, 0, JK1, JK2;
                h1e, g2e = wfm.utils.h_dimer_2q(params); # construct ham
                # construct h_SR (determinant basis)
                hSR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);
                # Vg splitting
                if(j == impis[0]): hSR += (DeltaV/2)*np.eye(len(source));    
                if(j == impis[1]): hSR += (-DeltaV/2)*np.eye(len(source));     
                # transform to eigenbasis
                hSR_ent = wfm.utils.entangle(hSR, *pair);
                hSR_diag = np.dot( np.linalg.inv(Udiag), np.dot(hSR_ent, Udiag));
                # force diagonal
                if((j not in impis) and True):
                    hSR_diag = np.diagflat(np.diagonal(hSR_diag));
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
        ax.plot(rhoJavals, Tvals[:,pair[0]], label = "$|+>$", color = colors[J12i], linestyle = "dashed", linewidth = 2);
        ax.plot(rhoJavals, Tvals[:,pair[1]], label = "$|->$", color = colors[J12i], linestyle = "dashdot", linewidth = 2);
        ax.plot(rhoJavals, Tvals[:,0]+Tvals[:,1]+Tvals[:,2]+Rvals[:,0]+Rvals[:,1]+Rvals[:,2], color = "red")

        # inset
        if True:           
            axins.plot(rhoJavals,Tvals[:,pair[0]]/Tvals[:,pair[1]], color = colors[J12i], linestyle = "solid", linewidth = 2); # + state
            
    # format and show
    ax.set_xlim(min(rhoJavals),max(rhoJavals));
    ax.set_xticks([0,1,2]);
    ax.set_xlabel("$J/\pi \sqrt{t(E+2t)}$", fontsize = "x-large");
    ax.set_ylim(0,0.15);
    ax.set_yticks([0,0.15]);
    ax.set_ylabel("$T$", fontsize = "x-large");
    axins.set_xlim(min(rhoJavals),max(rhoJavals));
    axins.set_xticks([0,1,2]);
    axins.set_xlabel("$J/\pi \sqrt{tE_b}$", fontsize = "x-large");
    #axins.set_ylim(min(Tvals[:,pair[0]]/Tvals[:,pair[1]]), 1.2*max(Tvals[:,pair[0]]/Tvals[:,pair[1]]) );
    axins.set_ylabel("$T_+/T_-$", fontsize = "x-large");
    plt.show();

    fig, ax = plt.subplots();
    ax.plot(rhoJavals,Tvals[:,pair[0]]/Tvals[:,pair[1]]);
    plt.show();


