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

# tight binding params
#tl = 0.0056; # lead hopping, in Hartree
#tp = 0.0056;  # hopping between imps

# params, all in units meV
tl = 100;
tp = 100;
th = tl/5;
Ucharge = 1000;
JK = 8*th*th/Ucharge;

# Ab initio params, in meV:
Ha2meV = 27.211386*1000; # 1 hartree is 27 eV
Jx = 0.209; # convert to hartree
Jz = 0.124;
DO = 0.674;
DT = 0.370;
Jz = 0 # necessary, but why?

# convert to Ha
print("\n>>>params, in meV:\n",tl, tp, JK, Jx, DO, DT); 
del th, Ucharge;
Ha2meV = 27.211386*1000;
tl, tp, JK, Jx, DO, DT= tl/Ha2meV, tp/Ha2meV, JK/Ha2meV, Jx/Ha2meV, DO/Ha2meV, DT/Ha2meV;

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
            
#########################################################
#### generation

if True: # fig 6 ie T vs rho J a, with T vs E inset optional

    fig, ax = plt.subplots();
    dummyvals = [0];
    for dummyi in range(len(dummyvals)):

        # iter over rhoJ, getting T
        Tvals, Rvals = [], [];
        rhoJalims = np.array([0.05,16.0]);
        rhoJavals = np.linspace(rhoJalims[-1], rhoJalims[0], 99);
        Elims = JK*JK/(rhoJalims*rhoJalims*np.pi*np.pi*tl) - 2*tl;
        Evals = np.linspace(Elims[-1], Elims[0], 99); # switched !
        for rhoi in range(len(rhoJavals)):

            # energy
            rhoJa = rhoJavals[rhoi];
            Energy = JK*JK/(rhoJa*rhoJa*np.pi*np.pi*tl) - 2*tl; # measure from mu
            ka = np.arccos(Energy/(-2*tl));
            Vg = Energy + 2*tl;

            # construct hblocks
            hblocks = [];
            impis = [1,2];
            Udiag = 0; # dummy for later
            for j in range(4): # iter over imps
                # define all physical params
                JKO, JKT = 0, 0;
                if (j == impis[0]): JKO = JK # J S dot sigma is onsite only
                elif(j == impis[1]): JKT = JK
                params = Jx, Jx, Jz, DO, DT, 0, JKO, JKT;
                h1e, g2e = wfm.utils.h_cobalt_2q(params); # construct ham
                # construct h_SR (determinant basis)
                hSR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);               
                # entangle, ie basis {|+>, |->, |sigma0>}
                hSR_ent = wfm.utils.entangle(hSR, *pair);
                # make leads diagonal in this basis
                if(j==0): _, Udiag = np.linalg.eigh(hSR_ent); # diagonalization is in leads only
                hSR_diag = np.dot(np.linalg.inv(Udiag), np.dot(hSR_ent, Udiag));
                if(verbose > 3 and rhoi == 0 and j == 0):
                    print("\nJKO, JKT = ",JKO*Ha2meV, JKT*Ha2meV);
                    print(" - ham:\n", Ha2meV*np.real(hSR));
                    print(" - ent ham:\n", Ha2meV*np.real(hSR_ent));
                    print(" - ent hame should be: ",Ha2meV*np.real(DO-DT),Ha2meV*np.real((2*1.5*1.5-2*1.5+1)*(DO+DT)/2));
                    print(" - diag ham:\n", Ha2meV*np.real(hSR_diag));
                # add to blocks list
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

            # T (Energy from 0)
            Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, verbose = 0));
            Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, reflect = True));
            
        # plot
        Tvals, Rvals = np.array(Tvals), np.array(Rvals);
        ax.plot(rhoJavals, Tvals[:,sourcei], label = "$|i\,>$", color = "black", linewidth = 2);
        ax.plot(rhoJavals, Tvals[:,pair[0]], label = "$|+'>$", color = "black", linestyle = "dashed", linewidth = 2);
        ax.plot(rhoJavals, Tvals[:,pair[1]], label = "$|-'>$", color = "black", linestyle = "dashdot", linewidth = 2);
        ax.plot(rhoJavals, Tvals[:,0]+Tvals[:,1]+Tvals[:,2]+Rvals[:,0]+Rvals[:,1]+Rvals[:,2], color = "red")
        ax.plot(rhoJavals, Tvals[:,pair[0]]/Tvals[:,sourcei], label = "", color = "blue", linestyle = "solid", linewidth = 2);

        # inset
        if False:
            Evals = JK*JK/(rhoJvals*rhoJvals*np.pi*np.pi*tl)-2*tl;
            axins = inset_axes(ax, width="50%", height="50%");
            axins.plot(Evals,Tvals[:,pair[0]], color = "darkgreen", linestyle = "dashed", linewidth = 2); # + state
            axins.set_xlabel("$E/t$", fontsize = "x-large");
            axins.set_ylim(0,0.2);
            axins.set_yticks([0,0.2]);

        # format and show
        #ax.set_xticks([0,1]);
        ax.set_xlabel("$J_K/\pi \sqrt{t(E+2t)}$", fontsize = "x-large");
        #ax.set_ylim(0,1.0);
        #ax.set_yticks([0,0.2]);
        ax.set_ylabel("$T$", fontsize = "x-large");
        plt.legend();
        plt.show();


#cos(theta) vs DeltaK only
if False:

    # dependent var containers
    numxvals = 15;
    DeltaKvals = DO*np.linspace(-400,100,numxvals);
    rhoJa = 1
    Erho = DO*DO/(rhoJa*rhoJa*np.pi*np.pi*tl); 

    # independent var containers
    Tvals = np.zeros((len(pair),len(DeltaKvals)));
    Rvals = np.zeros_like(Tvals);

    # |+'> channel and |-'> channel
    for pairi in range(len(pair)):

        # iter over JK
        for DKi in range(len(DeltaKvals)):
            DeltaK = DeltaKvals[DKi];

            # 2 site SR
            hblocks = [np.copy(hSR_JK0_diag)];
            for Coi in range(2): # iter over imps

                # define all physical params
                JKO, JKT = 0, 0;
                if Coi == 0: JKO = DO#+DeltaK/2; # J S dot sigma is onsite only
                else: JKT = DO#-DeltaK/2;
                params = Jx, Jx, Jz, DO, DT, An, JKO, JKT;
                h1e, g2e = wfm.utils.h_dimer_2q(params); # construct ham

                # construct h_SR (determinant basis)
                hSR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);

                # diagonalize lead states
                hSR_diag = np.dot(np.linalg.inv(Udiag), np.dot(hSR, Udiag));
                if(True):
                    print("\nJKO, JKT = ",JKO*Ha2meV, JKT*Ha2meV);
                    print(" - ham:\n", Ha2meV*np.real(hSR));
                    print(" - transformed ham:\n", Ha2meV*np.real(hSR_diag));

                if(True):
                    if Coi == 0: hSR_diag += (DeltaK/2)*np.eye(len(hSR_JK0_diag));
                    elif Coi == 1: hSR_diag += (-DeltaK/2)*np.eye(len(hSR_JK0_diag));
                
                # add to blocks list
                hblocks.append(np.copy(hSR_diag));

            # finish hblocks
            hblocks.append(hSR_JK0_diag);
            hblocks = np.array(hblocks);
            E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
            for hb in hblocks:
                hb += -E_shift*np.eye(np.shape(hblocks[0])[0]);
            if(verbose): print("\nhblocks = \n",hblocks);

            # hopping
            tnn = np.array([-th*np.eye(len(source)),-tp*np.eye(len(source)),-th*np.eye(len(source))]);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # T and R for desired channel only
            Tvals[pairi, DKi] = wfm.kernel(hblocks, tnn, tnnn, tl, Erho - 2*tl, source, verbose = 0)[pair[pairi]];
            #Rvals[pairi, DKi] = wfm.kernel(hblocks, tnn, tnnn, tl, Erhovals[rhoi] - 2*tl, source, reflect = True)[pair[pairi]];    

    # put plotting arrays in right form
    DeltaKvals = DeltaKvals/DO; # convert
    thetavals = 2*np.arctan(Tvals[pair[0]]/Tvals[pair[1]])/np.pi;
    
    # plot (To do)
    fig, ax = plt.subplots();
    ax.plot(DeltaKvals, thetavals, color = "darkblue", linewidth = 2);
                      
    # format and show
    #ax.set_xlim(min(DeltaKvals),max(DeltaKvals));
    ax.set_xlabel("$\Delta_{K} /D_O$", fontsize = "x-large");
    #ax.set_ylim(0,1);
    #ax.set_yticks([0,1]);
    ax.set_ylabel("$\\theta/\pi$", fontsize = "x-large");
    plt.show();

    # end sweep over JK
    raise(Exception);


#cos(theta) vs energy and DeltaK 
if False:

    # dependent var containers
    numxvals = 15;
    DeltaKvals = DO*np.linspace(-100,100,numxvals);
    rhoJavals = np.linspace(0.01,4.0,numxvals);
    Erhovals = DO*DO/(rhoJavals*rhoJavals*np.pi*np.pi*tl); # measured from bottom of band

    # independent var containers
    Tvals = np.zeros((len(pair),len(DeltaKvals),len(rhoJavals)));
    Rvals = np.zeros_like(Tvals);

    # |+'> channel and |-'> channel
    for pairi in range(len(pair)):

        # iter over JK
        for DKi in range(len(DeltaKvals)):
            DeltaK = DeltaKvals[DKi];

            # 2 site SR
            hblocks = [np.copy(hSR_JK0_diag)];
            for Coi in range(2): # iter over imps

                # define all physical params
                JKO, JKT = 0, 0;
                if Coi == 0: JKO = DO#+DeltaK/2; # J S dot sigma is onsite only
                else: JKT = DO#-DeltaK/2;
                params = Jx, Jx, Jz, DO, DT, An, JKO, JKT;
                h1e, g2e = wfm.utils.h_dimer_2q(params); # construct ham

                # construct h_SR (determinant basis)
                hSR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);

                # diagonalize lead states
                hSR_diag = np.dot(np.linalg.inv(Udiag), np.dot(hSR, Udiag));
                if(True):
                    print("\nJKO, JKT = ",JKO*Ha2meV, JKT*Ha2meV);
                    print(" - ham:\n", Ha2meV*np.real(hSR));
                    print(" - transformed ham:\n", Ha2meV*np.real(hSR_diag));

                if(True):
                    if Coi == 0: hSR_diag += (DeltaK/2)*np.eye(len(hSR_JK0_diag));
                    elif Coi == 1: hSR_diag += (-DeltaK/2)*np.eye(len(hSR_JK0_diag));
                
                # add to blocks list
                hblocks.append(np.copy(hSR_diag));

            # finish hblocks
            hblocks.append(hSR_JK0_diag);
            hblocks = np.array(hblocks);
            E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
            for hb in hblocks:
                hb += -E_shift*np.eye(np.shape(hblocks[0])[0]);
            if(verbose): print("\nhblocks = \n",hblocks);

            # hopping
            tnn = np.array([-th*np.eye(len(source)),-tp*np.eye(len(source)),-th*np.eye(len(source))]);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # iter over rhoJ (1/k)
            for rhoi in range(len(rhoJavals)):

                # T and R for desired channel only
                Tvals[pairi, DKi, rhoi] = wfm.kernel(hblocks, tnn, tnnn, tl, Erhovals[rhoi] - 2*tl, source, verbose = 0)[pair[pairi]];
                #Rvals[pairi, DKi, rhoi] = wfm.kernel(hblocks, tnn, tnnn, tl, Erhovals[rhoi] - 2*tl, source, reflect = True)[pair[pairi]];    

    # put plotting arrays in right form
    DeltaKvals = DeltaKvals/DO; # convert
    DeltaKvals, rhoJavals = np.meshgrid(DeltaKvals, rhoJavals);
    thetavals = 2*np.arctan(Tvals[pair[0]].T/Tvals[pair[1]].T);
    
    # plot (To do)
    fig = plt.figure();   
    #ax.plot(DeltaKvals, thetavals/np.pi, color = "darkblue", linewidth = 2);
    ax3d = fig.add_subplot(projection = "3d");
    ax3d.plot_surface(rhoJavals, DeltaKvals, thetavals, cmap = cm.coolwarm);
                      
    # format and show
    #ax.set_xlim(min(DeltaKvals),max(DeltaKvals));
    ax3d.set_xlabel("$\Delta_{K} /D_O$", fontsize = "x-large");
    #ax.set_ylim(0,1);
    #ax.set_yticks([0,1]);
    ax3d.set_ylabel("$\\theta/\pi$", fontsize = "x-large");
    plt.show();

    # end sweep over JK
    raise(Exception);   





