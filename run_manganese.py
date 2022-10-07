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
mymarkevery = (40,40);
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

#### setup

# def particles and their single particle states
#species = np.array([1,1,1]); # num of each species, which are one e, elec, spin-3/2, spin-3/2
#states = [[0,1],[2,3,4,5,6,7,8,9,10,11,12,13,14],[15,16,17,18,19,20,21,22,23,24,25,26,27]]; # e up, down, spin 1 mz, spin 2 m#dets = np.array([xi for xi in itertools.product(*tuple(states))]); # product states
#dets_int = [[0,2,16],[0,3,15],[1,2,15]]; # tMSQ subspace

# initialize source vector in down, up, up state
sourcei = 2; # |down, 6, 6 > in 3 state reduced space
source = np.zeros(3);
source[sourcei] = 1;

# entangled pair
pair = (0,1); # |up, s, s-1 > and |up, s-1, s >

# constructing the hamiltonian
def reduced_ham(params, S):
    D1, D2, J12, JK1, JK2 = params;
    assert(D1 == D2);
    ham = np.array([[S*S*D1+(S-1)*(S-1)*D2+S*(S-1)*J12+(JK1/2)*S+(JK2/2)*(S-1), S*J12, np.sqrt(2*S)*(JK2/2) ], # up, s, s-1
                    [S*J12, (S-1)*(S-1)*D1+S*S*D2+S*(S-1)*J12+(JK1/2)*S + (JK2/2)*(S-1), np.sqrt(2*S)*(JK1/2) ], # up, s-1, s
                    [np.sqrt(2*S)*(JK2/2), np.sqrt(2*S)*(JK1/2),S*S*D1+S*S*D2+S*S*J12+(-JK1/2)*S +(-JK2/2)*S]], # down, s, s
                   dtype = complex);

    return ham;

################################################################################        
#### material data
material = "MnPc";

# universal
tl = 100; # in meV
tp = 100; # in meV
JK = tl/100; 

if material == "Mn":
    myspinS = 6;

    # Jie Xiang paper results in cm^-1. Convert immediately to meV
    cm2meV = 1/8.06;
    D1 = -0.22*cm2meV; # converted from cm^-1 to meV
    D2 = D1;
    J12 = -2*0.025*cm2meV; # converted from cm^-1 to meV

elif material == "MnPc":
    myspinS = 3/2;

    # Jie Xiang paper results in cm^-1. Convert immediately to meV
    cm2meV = 1/8.06;
    D1 = 1.0;
    D2 = D1;
    J12 = 20.0;

elif material == "Co":
    myspinS = 3/2;

    # Jie-Xiang manuscript results in meV
    D1 = 0.674;
    D2 = 0.370;
    D1 = D2;
    Jx = 0.209; 
    Jz = 0.124;
    J12 = 0;
    
else: raise Exception("Material "+material+" not supported");

# convert to units of tl
tl, tp, D1, D2, J12, JK = tl/tl, tp/tl, D1/tl, D2/tl, J12/tl, JK/tl;

print("*"*50);
print("Material = "+material);
print("params = ",tl, tp, D1, D2, J12, JK);

################################################################################        
#### run code

if True: # T+ at different Delta E by changing D

        
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
            if(j == impis[0]): JK1 = JK;
            elif(j == impis[1]): JK2 = JK;
            params = D1, D2, J12, JK1, JK2;
            # construct h_SR (determinant basis)
            hSR = reduced_ham(params, myspinS);
            if( Evali == 0 and j == 0):
                # see eigenstates in the determinant basis
                eigEs, Udiag = np.linalg.eigh(hSR); 
                print("\nDeterminant basis:");
                print(" - ham:\n", np.real(hSR));
                print(" - |+'>: ",Udiag[:,1],"\n - |-'>: ", Udiag[:,0],"\n - |i'>: ", Udiag[:,2]);

            # transform the |+>, |->, |i> basis (entangling basis)
            hSR_ent = wfm.utils.entangle(hSR, *pair);
            if( Evali == 0 and j == 0):
                # see eigenstates in the entangling basis
                eigEs, Udiag = np.linalg.eigh(hSR_ent);
                print("\nEntangling basis:");
                print(" - ent ham:\n", np.real(hSR_ent));
                const_term = 2*myspinS*myspinS*(D1+D2)/2 + (myspinS*myspinS-myspinS)*J12;
                print(" - ent diagonal should be: ",const_term + np.array([myspinS*J12+(1-2*myspinS)*(D1+D2)/2,-1.5*J12+(1-2*myspinS)*(D1+D2)/2,myspinS*J12]));
                print(" - ent off-diagonal should be: ",0);
                print(" - |+'>: ",Udiag[:,1],"\n - |-'>: ", Udiag[:,0],"\n - |i'>: ", Udiag[:,2]);
                print(" - Delta E / t = ", hSR_ent[0,0]-hSR_ent[2,2]);

            # add to blocks list
            hblocks.append(np.copy(hSR_ent));

        # finish hblocks
        hblocks = np.array(hblocks);
        E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
        for hb in hblocks:
            hb += -E_shift*np.eye(np.shape(hblocks[0])[0]);
        Esplit = (hblocks[0][0,0] - hblocks[0][2,2])/tl;
        if(Evali == 0): print("Delta E / t = ", Esplit);
            
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
    fname = "data/"+material
    print("Saving data to "+fname);
    np.save(fname, data);

################################################################################        
#### plot

# load data
def load_data(fname):
    print("Loading data from "+fname);
    data = np.load(fname);
    tl = data[0,0];
    Jeff = data[0,1];
    myxvals = data[1];
    myTvals = data[2:5];
    myRvals = data[5:];
    mytotals = np.sum(myTvals, axis = 0) + np.sum(myRvals, axis = 0);
    print("- shape xvals = ", np.shape(myxvals));
    print("- shape Tvals = ", np.shape(myTvals));
    print("- shape Rvals = ", np.shape(myRvals));
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
    num_plots = 2;
    fig, axes = plt.subplots(num_plots, sharex=True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);
    datafs = sys.argv[1:];
    for fi in range(len(datafs)):
        xvals, Rvals, Tvals, totals = load_data(datafs[fi]);
        logElims = -4,0;

        # plot T+
        axes[0].plot(xvals, Tvals[pair[0]], color=mycolors[fi], marker=mymarkers[fi], markevery=mymarkevery, linewidth = mylinewidth); 
        #axes[0].plot(xvals, totals, color="red");
        print(">>> T+ max = ",np.max(Tvals[pair[0]])," at Ki = ",xvals[np.argmax(Tvals[pair[0]])]);

        # plot analytical FOM
        axes[1].plot(xvals, np.sqrt(Tvals[sourcei]*Tvals[pair[0]]), color = mycolors[fi], marker=mymarkers[fi],markevery=mymarkevery, linewidth = mylinewidth)
        print(">>> p2 max = ",np.max(np.sqrt(Tvals[sourcei]*Tvals[pair[0]]))," at Ki = ",xvals[np.argmax(np.sqrt(Tvals[sourcei]*Tvals[pair[0]]))]);

    # format
    axes[0].set_ylim(0,0.1);
    axes[0].set_ylabel('$T_+$', fontsize = myfontsize);
    axes[1].set_ylim(0.0,0.2);
    axes[1].set_ylabel('$\overline{p^2}(\\tilde{\\theta})$', fontsize = myfontsize);

    # show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    plt.show();

