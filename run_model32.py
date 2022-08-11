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
mymarkers = ["o","^","s","d","*","X","P"];
mystyles = ["solid","dashed"];
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
#### generate data

if False: # T+ at different Delta E by changing D
    
    Dvals = np.array([2/100]) #0,1/1000,1/100,2/100]);
    for Di in range(len(Dvals)):
        Dval = Dvals[Di];

    #Dval = 0;
    #DeltaEvals = -2*Dvals;
    #DeltaJvals = (DeltaEvals+2*Dval)/(-3/2); # this is Jz - Jx
    #J12zvals = J12x + DeltaJvals;
    #for Di in range(len(J12zvals)):
        #J12z = J12zvals[Di];

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
            kappaa = 0.0*np.pi;
            Vg = Energy+2*tl*np.cos(kappaa);

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
        fname = "data/model32/D"+str(int(Dval*1000)/1000);
        print("Saving data to "+fname);
        np.save(fname, data);


########################################################################
#### plot data

# load data
def load_data(fname):
    print("Loading data from "+fname);
    data = np.load(fname);
    tl = data[0,0];
    Jeff = data[0,1];
    myxvals = data[1];
    myTvals = data[2:10];
    myRvals = data[10:];
    mytotals = np.sum(myTvals, axis = 0) + np.sum(myRvals, axis = 0);
    print("- shape xvals = ", np.shape(myxvals));
    print("- shape Tvals = ", np.shape(myTvals));
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
    num_subplots = 2
    fig, (mainax, fomax) = plt.subplots(num_subplots, sharex = True);
    fig.set_size_inches(7/2,3*num_subplots/2);
    datafs = sys.argv[1:];
    for fi in range(len(datafs)):
        xvals, Rvals, Tvals, totals = load_data(datafs[fi]);
        mymarkevery = (fi*10,50);

        # plot T+
        mainax.plot(xvals, Tvals[pair[0]], color=mycolors[fi], marker=mymarkers[fi], markevery=mymarkevery, linewidth = mylinewidth); 
        #mainax.plot(xvals, totals, color="red");

        # plot FOM
        fomvals = np.empty_like(xvals);
        for xi in range(len(xvals)):
            fomvals[xi] = FOM(Tvals[sourcei,xi],Tvals[pair[0],xi]);
        fomax.plot(xvals, fomvals, color = mycolors[fi], marker=mymarkers[fi],markevery=mymarkevery, linewidth = mylinewidth);

    # format
    mainax.set_ylim(0,0.16);
    mainax.set_yticks([0,0.08,0.16]);
    mainax.set_ylabel('$T_+$', fontsize = myfontsize);
    mainax.set_title(mypanels[0], x=0.07, y = 0.7, fontsize = myfontsize);
    fomax.set_xscale('log', subs = []);
    fomax.set_xlim(10**(-5),10**(-1));
    fomax.set_xticks([10**(-5),10**(-4),10**(-3),10**(-2),10**(-1)])
    fomax.set_xlabel('$K_i/t$', fontsize = myfontsize);
    fomax.set_ylim(0,0.32);
    fomax.set_yticks([0.0,0.16,0.32]);
    fomax.set_ylabel('$\overline{p^2}(\\tilde{\\theta})$', fontsize = myfontsize);
    fomax.set_title(mypanels[1], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    #plt.show();
    plt.savefig('model32_positive.pdf');




if False: # T+ at different Delta E by changing J12z

    Dvals = np.array([0,1/1000,1/100,2/100]);
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
            kappaa = 0.0*np.pi;
            Vg = Energy+2*tl*np.cos(kappaa);

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
    datafs_pos = ["J12z0.01_copy.npy","J12z0.011.npy","J12z0.023.npy","J12z0.036.npy"];
    datafs = datafs_neg[:]; datafs.extend(datafs_pos);
    for fi in range(len(datafs)):
        xvals, Rvals, Tvals, totals = load_data(datadir+datafs[fi]);
        mymarkevery = (fi*10,50);

        if datafs[fi] in datafs_neg:
            # plot T+ for negative case
            axes[0].plot(xvals, Tvals[pair[0]], color=mycolors[fi], marker=mymarkers[fi], markevery=mymarkevery, linewidth = mylinewidth); 
            #axes[0].plot(xvals, totals, color="red");

        if datafs[fi] in datafs_pos:
            # plot T+ for positive case
            axes[1].plot(xvals, Tvals[pair[0]], color=mycolors[fi-len(datafs_neg)], marker=mymarkers[fi-len(datafs_neg)], markevery=(10*(fi-len(datafs_neg)),50), linewidth = mylinewidth); 
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
    #plt.savefig('model32_positive.pdf');



