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
mycolors = ["black","darkblue","darkgreen","darkred", "darkcyan", "darkmagenta","darkgray"];
mymarkers = ["o","^","s","d","*","X","P"];
mycolors, mymarkers = np.append(mycolors,mycolors), np.append(mymarkers, mymarkers);
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
JK = 10*-0.5*tl/100; ##############################################################################################
######################################################################################
######################################################################################
J12 = tl/100;

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
#### effects of Ki and Delta E

if False: # T+ at different Delta E by changing D
    myspinS = 6;
    # Evals should be order of D (0.1 meV for Mn to 1 meV for MnPc)
    Esplitvals = (10)*np.array([-0.004,-0.003,-0.002,-0.001,0.0,0.001,0.002,0.003,0.004,0.02]);
    #Esplitvals = (1)*np.array([-0.004,-0.003,-0.002,-0.001,0.0]);
    Dvals = Esplitvals/(1-2*myspinS);
    for Dvali in range(len(Dvals)):
        Dval = Dvals[Dvali];

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
            params = Dval, Dval, J12, JK1, JK2;
            # construct h_SR (determinant basis)
            hSR = reduced_ham(params,S=myspinS);           
            # transform to eigenbasis
            hSR_diag = wfm.utils.entangle(hSR, *pair);
            hblocks.append(np.copy(hSR_diag));
            if(verbose > 3 ):
                print("\nJK1, JK2 = ",JK1, JK2);
                print(" - ham:\n", hSR);
                print(" - transformed ham:\n", np.real(hSR_diag));
                print(" - DeltaE = ",Esplitvals[Dvali])

        # finish hblocks
        hblocks = np.array(hblocks);
        E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
        for hb in hblocks:
            hb += -E_shift*np.eye(np.shape(hblocks[0])[0]);
        if(verbose > 3 ): print("Delta E / t = ", (hblocks[0][0,0] - hblocks[0][2,2])/tl);

        # constant shift in right lead to bring transmitted |+> on resonance
        #hblocks[-1] += -1*Esplitvals[Dvali]*np.eye(np.shape(hblocks[0])[0]);

        # hopping
        tnn = np.array([-tl*np.eye(len(source)),-tp*np.eye(len(source)),-tl*np.eye(len(source))]);
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

        # iter over E, getting T
        logElims = -6,-2
        Evals = np.logspace(*logElims,myxvals, dtype = complex);
        Rvals = np.empty((len(Evals),len(source)), dtype = float);
        Tvals = np.empty((len(Evals),len(source)), dtype = float);
        for Evali in range(len(Evals)):

            # energy
            Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper

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
        fname = "data/model"+str(myspinS)+"/Esplit"+str(int(Esplitvals[Dvali]*100000)/100000);
        print("Saving data to "+fname);
        np.save(fname, data);

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
    Eindex = fname.find("Esplit")+5;
    dotindex = fname.find(".npy");
    Esplit = float(fname[Eindex+1:dotindex]);
    folder = fname[:Eindex-5];
    print("- Esplit = ",Esplit);
    print("- shape xvals = ", np.shape(myxvals));
    print("- shape Tvals = ", np.shape(myTvals));
    print("- shape Rvals = ", np.shape(myRvals));
    return myxvals, myRvals, myTvals, mytotals, Esplit, folder;

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

#### plot T+ and p2 vs Ki at different Delta E
if True:
    num_plots = 2;
    fig, axes = plt.subplots(num_plots, sharex=True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);
    datafs = sys.argv[1:];
    peaks = np.zeros((len(datafs),3));
    for fi in range(len(datafs)):
        xvals, Rvals, Tvals, totals, Esplit, folder = load_data(datafs[fi]);
        logElims = np.log10(xvals[0]), np.log10(xvals[-1]);

        # plot T+
        axes[0].plot(xvals, Tvals[pair[0]], color=mycolors[fi], marker=mymarkers[fi], markevery=mymarkevery(datafs[fi],Tvals[pair[0]]), linewidth = mylinewidth); 
        #mainax.plot(xvals, totals, color="red");
        Tpmax = np.max(Tvals[pair[0]])
        print(">>> T+ max = ",Tpmax," at Ki = ",xvals[np.argmax(Tvals[pair[0]])]);

        # plot analytical FOM
        axes[1].plot(xvals, np.sqrt(Tvals[sourcei]*Tvals[pair[0]]), color = mycolors[fi], marker=mymarkers[fi],markevery=mymarkevery(datafs[fi], np.sqrt(Tvals[sourcei]*Tvals[pair[0]])), linewidth = mylinewidth)
        p2max = np.max(np.sqrt(Tvals[sourcei]*Tvals[pair[0]]))
        print(">>> p2 max = ",p2max," at Ki = ",xvals[np.argmax(np.sqrt(Tvals[sourcei]*Tvals[pair[0]]))]);

        # replot with markers as needed
        if("-" not in datafs[fi] and "0.0.npy" not in datafs[fi]):
            pass;

        # record peaks
        peaks[fi,:] = [Esplit,Tpmax,p2max];
    # sort and save peaks
    peaks = peaks[peaks[:,0].argsort()]
    peaksfname = folder+"peaks.npy";
    print("Saving peaks data to "+peaksfname);
    np.save(peaksfname, peaks);    
        
    # format
    lower_y = 0.08;
    axes[0].set_ylim(-lower_y*0.2,0.2);
    axes[0].set_ylabel('$T_+$', fontsize = myfontsize);
    axes[1].set_ylim(0.0,0.3);
    axes[1].set_ylabel('$\overline{p^2}$', fontsize = myfontsize);

    # show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    #plt.savefig('figs/model1.pdf');
    plt.show();

