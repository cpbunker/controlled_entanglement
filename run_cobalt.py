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
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;

# fig standardizing
myxvals = 199;
myfontsize = 14;
mycolors = ["black","darkblue","darkgreen","darkred", "darkmagenta","darkgray","darkcyan"];
mymarkers = ["o","^","s","d","X","P","*"];
mystyles = ["solid","dashed"];
mymarkevery = (40,40)
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)"];
plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

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
JK = -1;

# Ab initio params, in meV:
Ha2meV = 27.211386*1000; # 1 hartree is 27 eV
Jx = 0.209; 
Jz = 0.124;
DO = 0.674;
DT = 0.370;
DO=DT;

# convert to Ha
print("\nParams, in meV:\n",tl, tp, JK, Jx, DO, DT); 
#Ha2meV = 27.211386*1000;
#tl, tp, JK, Jx, Jz, DO, DT= tl/Ha2meV, tp/Ha2meV, JK/Ha2meV, Jx/Ha2meV, Jz/Ha2meV, DO/Ha2meV, DT/Ha2meV;
tl, tp, JK, Jx, Jz, DO, DT= tl/tl, tp/tl, JK/tl, Jx/tl, Jz/tl, DO/tl, DT/tl;
print("\nParams, in tl:\n",tl, tp, JK, Jx, Jz, DO, DT);

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

if False: 

    dummyvals = [0];
    for dummyi in range(len(dummyvals)):

        # iter over rhoJ, getting T
        Tvals, Rvals = [], [];
        logElims = -6,-1
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
            kappaa = 0.0*np.pi;
            Vg = 0

            # construct hblocks
            hblocks = [];
            impis = [1,2];
            for j in range(4): # iter over imps
                # define all physical params
                JKO, JKT = 0, 0;
                if (j == impis[0]): JKO = JK # J S dot sigma is onsite only
                elif(j == impis[1]): JKT = JK
                params = Jx, Jx, Jz, DO, DT, 0, JKO, JKT;
                h1e, g2e = wfm.utils.h_spin32_2q(params); # construct ham
                # construct h_SR, basis = 1/2> |s>|s-1>, |1/2>|s-1>|s>, |-1/2>|s>|s> (determinant basis)
                hSR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);               
                if(dummyi == 0 and Evali == 0 and j==0):
                    # see eigenstates in the determinant basis
                    eigEs, Udiag = np.linalg.eigh(hSR); 
                    print("\nDeterminant basis:");
                    print(" - ham:\n", np.real(hSR));
                    print(" - |+'>: ",Udiag[:,1],"\n - |-'>: ", Udiag[:,0],"\n - |i'>: ", Udiag[:,2]);

                    # find the actual transmitted spin state in the determinant basis
                    beta = (2*1.5 - 1)*(DO-DT)/(2*1.5*Jx);
                    alpha11 = (1+np.sqrt(1+beta*beta)-beta)/(2*np.sqrt(1+beta*beta+np.sqrt(1+beta*beta)));
                    alpha12 = (1-np.sqrt(1+beta*beta)-beta)/(2*np.sqrt(1+beta*beta-np.sqrt(1+beta*beta)));
                    print("|1'>: ",);
                    
                    # find von neuman entropy
                    #for coli in [1,0]: get_VNE(Udiag[:,coli]);

                # change to the |+>, |->, |i> basis (entangling basis)
                hSR_ent = wfm.utils.entangle(hSR, *pair);
                if(dummyi == 0 and Evali == 0 and j==0):
                    # see eigenstates in the entangling basis
                    eigEs, Udiag = np.linalg.eigh(hSR_ent);
                    Dmid = (DO+DT)/2;
                    print("\nEntangling basis:");
                    print(" - ent ham:\n", np.real(hSR_ent));
                    const_term = 2*1.5*1.5*Dmid + (1.5*1.5-1.5)*Jz;
                    print(" - ent diagonal should be: ",const_term + np.array([1.5*Jx+(1-2*1.5)*Dmid,-1.5*Jx+(1-2*1.5)*Dmid,1.5*Jz]));
                    print(" - ent off-diagonal should be: ",DO-DT);
                    print(" - |+'>: ",Udiag[:,1],"\n - |-'>: ", Udiag[:,0],"\n - |i'>: ", Udiag[:,2]);

                # change to the |+'>, |-'>, |i> basis (diagonalizing basis)
                hSR_diag = np.dot(np.linalg.inv(Udiag), np.dot(hSR_ent, Udiag));
                if(dummyi == 0 and Evali == 0):
                    print("\nDiagonalizing basis:");
                    print(" - JKO, JKT = ",JKO, JKT);
                    print(" - diag ham:\n", np.real(hSR_diag));
                    
                # add to blocks list
                hblocks.append(np.copy(hSR_diag));
            #assert False;

            # finish hblocks
            hblocks = np.array(hblocks);
            hblocks[1] += Vg*np.eye(len(source)); # Vg shift in SR
            hblocks[2] += Vg*np.eye(len(source));
            E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
            for hb in hblocks:
                hb += -E_shift*np.eye(np.shape(hblocks[0])[0]);
            if(Eval == Evals[0]): print("E_1 - E_sigma0 : ",(hblocks[0][0,0] - hblocks[0][2,2])*tl);
            if(Eval == Evals[0]): print("E_2 - E_sigma0 : ",(hblocks[0][1,1] - hblocks[0][2,2])*tl);
            
            # hopping
            tnn = np.array([-tl*np.eye(len(source)),-tp*np.eye(len(source)),-tl*np.eye(len(source))]);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # get R, T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source, all_debug = False);
            Rvals[Evali] = Rdum;
            Tvals[Evali] = Tdum;
            
        # save data to .npy
        Tvals, Rvals = np.array(Tvals), np.array(Rvals);
        data = np.zeros((2+2*len(source),len(Evals)));
        data[0,0] = tl;
        data[0,1] = JK;
        data[1,:] = Evals;
        data[2:2+len(source),:] = Tvals.T;
        data[2+len(source):2+2*len(source),:] = Rvals.T;
        fname = "data/cobalt";
        print("Saving data to "+fname);
        np.save(fname, data);

if True: # plot

    # open command line file
    dataf = sys.argv[1];
    fig, axes = plt.subplots(2, sharex = True);
    fig.set_size_inches(7/2,6/2);
    print("Loading data from "+dataf);
    data = np.load(dataf);
    tl = data[0,0];
    Jeff = data[0,1];
    xvals = data[1];
    Tvals = data[2:2+len(source)];
    Rvals = data[2+len(source):2+2*len(source)];
    totals = np.sum(Tvals, axis = 0) + np.sum(Rvals, axis = 0);
    print("- shape xvals = ", np.shape(xvals));
    print("- shape Tvals = ", np.shape(Tvals));
    print("- shape Rvals = ", np.shape(Rvals));

    # plot T vs logE
    for pairi in range(len(pair)):
        axes[0].plot(xvals, Tvals[pair[pairi]], color=mycolors[0],linestyle=mystyles[pairi], marker=mymarkers[0],markevery=mymarkevery, linewidth = mylinewidth);      
    axes[0].set_ylabel("$T_\sigma$", fontsize = myfontsize);

    # plot T/T vs logE
    fomvals = np.sqrt(Tvals[sourcei]*(Tvals[pair[0]]+Tvals[pair[1]])); 
    axes[1].plot(xvals, fomvals, color = mycolors[0], linestyle = mystyles[0], marker=mymarkers[0],markevery=mymarkevery, linewidth = mylinewidth);  
    axes[1].set_ylabel("$\overline{p'^2}$", fontsize = myfontsize); 

    # show
    logElims = -6,-1;
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.065, y = 0.74, fontsize = myfontsize); 
    plt.tight_layout();
    #plt.savefig('figs/cobalt.pdf');
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





