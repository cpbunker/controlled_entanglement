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
myfontsize = 14;
mycolors = ["black","darkblue","darkgreen","darkred", "darkmagenta","darkgray","darkcyan"];
mystyles = ["solid", "dashed","dotted","dashdot"];
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

if False: # T/T vs rho J a at diff D
    
    Dvals = JK*np.array([-1/100,0,1/100,1/10,1]);
    for Di in range(len(Dvals)):
        D = Dvals[Di];

        # iter over rhoJ, getting T
        Tvals, Rvals = [], [];
        logElims = -5,-1
        Evals = np.logspace(*logElims,199);
        for Eval in Evals:

            # energy
            Energy = Eval - 2*tl;
            
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
                if(verbose > 3 and Eval == Evals[0]):
                    print("\nJK1, JK2 = ",JK1, JK2);
                    print(" - ham:\n", np.real(hSR));
                    print(" - transformed ham:\n", np.real(hSR_diag));
                    print(" - DeltaE = ",-D*(2*1.5-1))

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
         
        # save data to .npy
        Tvals, Rvals = np.array(Tvals), np.array(Rvals);
        data = np.zeros((2+2*len(source),len(Evals)));
        data[0,0] = tl;
        data[0,1] = JK;
        data[1,:] = Evals;
        data[2:2+len(source),:] = Tvals.T;
        data[2+len(source):2+2*len(source),:] = Rvals.T;
        fname = "data/model32/D"+str(int(D*1000)/1000);
        print("Saving data to "+fname);
        np.save(fname, data);

if False:

    # open command line file
    datafs = sys.argv[1:];
    fig, axes = plt.subplots(2, sharex = True);
    fig.set_size_inches(7/2,6/2);
    for fi in range(len(datafs)):
        dataf = datafs[fi];
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
        axes[0].plot(xvals, Tvals[pair[0]], color = mycolors[fi], linestyle = "solid", linewidth = mylinewidth);   
        axes[0].plot(xvals, totals, color="red");
        axes[0].set_ylim(0,0.2);
        axes[0].set_yticks([0,0.1,0.2]);
        axes[0].set_ylabel('$T_+$', fontsize = myfontsize);

        # plot T/T vs logE
        axes[1].plot(xvals, Tvals[pair[0]]/Tvals[sourcei], color = mycolors[fi], linestyle = "solid", linewidth = mylinewidth);   
        #axes[1].plot(xvals, totals, color="red");
        axes[1].set_ylim(0,8);
        axes[1].set_yticks([0,4,8]);
        axes[1].set_ylabel('$T_+/T_0$', fontsize = myfontsize); 

    # format
    axes[0].set_title(mypanels[0], x=0.93, y = 0.7, fontsize = myfontsize);
    axes[1].set_title(mypanels[1], x=0.93, y = 0.7, fontsize = myfontsize);
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(-5),10**(-1));
    axes[-1].set_xticks([10**(-5),10**(-4),10**(-3),10**(-2),10**(-1)])
    axes[-1].set_xlabel('$(E+2t)/t$', fontsize = myfontsize);
    plt.tight_layout();
    plt.savefig('model32.pdf');


#########################################################
#### symmetry breaking

def get_VNE(col_vec):
    a, b, c = col_vec[0], col_vec[1], col_vec[2];
    print("\n",40*"*");
    if(abs(c) > 1e-10): return;

    import qiskit.quantum_info as qi
    # create unperturbed +, - states
    plus_sv = qi.Statevector([0,1/np.sqrt(2),1/np.sqrt(2),0]);
    minus_sv = qi.Statevector([0,1/np.sqrt(2),-1/np.sqrt(2),0]);

    # create this state
    my_sv = a*plus_sv + b*minus_sv;
    my_sv = my_sv/my_sv.inner(my_sv); # normalize
    print(' -> ',my_sv.to_dict())

    # VNE
    rho0 = qi.partial_trace(my_sv,[1]);
    VNE = qi.entropy(rho0);
    print(' -> ',VNE);
    
if False: 

    # symmetry breaking
    Dmid = 0.5*JK;
    DeltaD = 0.1*JK;
    D1 = Dmid + DeltaD/2;
    D2 = Dmid - DeltaD/2;
    del J12;
    J12vals = DeltaD*np.array([0.1,1,10]); # ie DeltaD/J12 = 10, 1, 0.1
    for J12i in range(len(J12vals)):
        J12 = J12vals[J12i];

        # iter over rhoJ, getting T
        Tvals, Rvals = [], [];
        logElims = -5,-1
        Evals = np.logspace(*logElims,199);
        for Eval in Evals:

            # energy
            Energy = Eval - 2*tl;

            # optical distances, N = 2 fixed
            ka = np.arccos((Energy)/(-2*tl));
            Vg = Energy + 2*tl; # gate voltage

            # JK=0 matrix for ref
            h1e_0, g2e_0 = wfm.utils.h_cobalt_2q((J12,J12,J12,D1,D2, 0, 0, 0));
            hSR_0 = fci_mod.single_to_det(h1e_0, g2e_0, species, states, dets_interest = dets52);
            hSR_0 = wfm.utils.entangle(hSR_0, *pair);
            #print(hSR_0); assert False;
            _, Udiag = np.linalg.eigh(hSR_0);
            del h1e_0, g2e_0, hSR_0;

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
            if(Eval == Evals[0]):
                for coli in range(len(Udiag)):
                    get_VNE(Udiag[:,coli]);
                #assert False;

            # construct hblocks
            hblocks = [];
            impis = [1,2];
            for j in range(4): # LL, imp 1, imp 2, RL
                # define all physical params
                JK1, JK2 = 0, 0;
                if(j == impis[0]): JK1 = JK;
                elif(j == impis[1]): JK2 = JK;
                params = J12, J12, J12, D1, D2, 0, JK1, JK2;
                h1e, g2e = wfm.utils.h_cobalt_2q(params); # construct ham
                # construct h_SR (determinant basis)
                hSR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);  
                # transform to eigenbasis
                hSR_ent = wfm.utils.entangle(hSR, *pair);
                hSR_diag = np.dot( np.linalg.inv(Udiag), np.dot(hSR_ent, Udiag));
                # force diagonal
                if((j not in impis) and True):
                    hSR_diag = np.diagflat(np.diagonal(hSR_diag));
                hblocks.append(np.copy(hSR_diag));
                if(verbose > 3 and Eval == Evals[0]):
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
         
        # save data to .npy
        Tvals, Rvals = np.array(Tvals), np.array(Rvals);
        data = np.zeros((2+2*len(source),len(Evals)));
        data[0,0] = tl;
        data[0,1] = JK;
        data[1,:] = Evals;
        data[2:2+len(source),:] = Tvals.T;
        data[2+len(source):2+2*len(source),:] = Rvals.T;
        fname = "data/model32/J12"+str(int(DeltaD/J12));
        print("Saving data to "+fname);
        np.save(fname, data);

if True:

    # open command line file
    datafs = sys.argv[1:];
    fig, axes = plt.subplots(2, sharex = True);
    fig.set_size_inches(7/2,6/2);
    for fi in range(len(datafs)):
        dataf = datafs[fi];
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
        axes[0].plot(xvals, Tvals[pair[1]], color = mycolors[0], linestyle = mystyles[fi], linewidth = mylinewidth);   
        axes[0].plot(xvals, totals, color="red");
        axes[0].set_ylim(0,0.1);
        axes[0].set_yticks([0,0.05,0.1]);
        axes[0].set_ylabel("$T_{-'}$", fontsize = myfontsize);

        # plot T/T vs logE
        axes[1].plot(xvals, Tvals[pair[0]]/Tvals[pair[1]], color = mycolors[0], linestyle = mystyles[fi], linewidth = mylinewidth);   
        #axes[1].plot(xvals, totals, color="red");
        axes[1].set_ylim(0,1.0);
        axes[1].set_yticks([0,0.5,1.0]);
        axes[1].set_ylabel("$T_{+'}/T_{-'}$", fontsize = myfontsize); 

    # format
    axes[0].set_title(mypanels[0], x=0.93, y = 0.7, fontsize = myfontsize);
    axes[1].set_title(mypanels[1], x=0.93, y = 0.7, fontsize = myfontsize);
    axes[-1].set_xscale('log', subs = [2,3,4,5,6,7,8,9]);
    axes[-1].set_xlim(10**(-5), 10**(-1));
    axes[-1].set_xticks([10**(-5),10**(-4),10**(-3),10**(-2),10**(-1)])
    axes[-1].set_xlabel('$(E+2t)/t$', fontsize = myfontsize);
    plt.tight_layout();
    plt.savefig('model32_broken.pdf');






###########################################################################
#### misc

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
    

if False: # T vs E

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

        


