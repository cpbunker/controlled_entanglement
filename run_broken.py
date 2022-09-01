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
mycolors = ["black","darkblue","darkgreen","darkred", "darkmagenta","darkgray","darkcyan"];
mymarkers = ["o","^","s","d","X","P","*"];
mystyles = ["solid","dashed"];
mymarkevery = 50;
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

# tight binding params
tl = 1.0;
tp = 1.0;
JK = 0.1;
    
def get_VNE(eigvec,eigval):
    a, b, c = eigvec[0], eigvec[1], eigvec[2]; # coefs in the basis \{ |+>, |->, |i> \}
    print("\n",40*"*");

    import qiskit.quantum_info as qi
    # create unperturbed +, - states
    plus_sv = qi.Statevector([0,1/np.sqrt(2),1/np.sqrt(2),0]); # create basis statevectors
    minus_sv = qi.Statevector([0,1/np.sqrt(2),-1/np.sqrt(2),0]);
    i_sv = qi.Statevector([1,0,0,0]);

    # create Statevector as linear combination in this basis
    my_sv = a*plus_sv + b*minus_sv + c*i_sv;
    my_sv = my_sv/my_sv.inner(my_sv); # has to be normalized !!!
    print('Qiskit eigenstate = ',my_sv.to_dict());
    print('Eigenenergy = ',eigval);

    # VNE
    rho0 = qi.partial_trace(my_sv,[1]);
    VNE = qi.entropy(rho0);
    print('VNE = ',VNE,'\n');

#########################################################
#### generate data
    
if False: 
    Dmid = 0.5*JK;
    DeltaD = 0.1*JK;
    D1 = Dmid + DeltaD/2;
    D2 = Dmid - DeltaD/2;
    J12z = 0;
    betavals = np.array([10,1,0.1]); # beta = (2s-1)*DeltaD/(2s*J12x) = 2*DeltaD/3*J12x for s=3/2
    J12xvals = 2*DeltaD/(3*betavals)
    for J12xi in range(len(J12xvals)):
        J12x = J12xvals[J12xi]; # reassign J12x to get desired beta value
        beta = betavals[J12xi]

        # iter over E, getting T
        logElims = -3,0
        Evals = np.logspace(*logElims,myxvals);
        Rvals = np.empty((len(Evals),len(source)), dtype = float);
        Tvals = np.empty((len(Evals),len(source)), dtype = float);
        for Evali in range(len(Evals)):

            # energy
            Eval = Evals[Evali]; # Eval > 0 always, what I call Ki in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper
            
            # optical distances, N = 2 fixed
            N0 = 1; # N0 = N - 1
            ka = np.arccos((Energy)/(-2*tl));

            # JK=0 matrix for ref
            h1e_0, g2e_0 = wfm.utils.h_cobalt_2q((J12x,J12x,J12z,D1,D2, 0, 0, 0));
            hSR_0 = fci_mod.single_to_det(h1e_0, g2e_0, species, states, dets_interest = dets52);
            hSR_0 = wfm.utils.entangle(hSR_0, *pair);
            #print(hSR_0); assert False;
            epsilons, Udiag = np.linalg.eigh(hSR_0);
            del h1e_0, g2e_0, hSR_0;

            # construct hblocks
            hblocks = [];
            impis = [1,2];
            for j in range(4): # LL, imp 1, imp 2, RL
                # define all physical params
                JK1, JK2 = 0, 0;
                if(j == impis[0]): JK1 = JK;
                elif(j == impis[1]): JK2 = JK;
                params = J12x, J12x, J12z, D1, D2, 0, JK1, JK2;
                h1e, g2e = wfm.utils.h_cobalt_2q(params); # construct ham
                # construct h_SR (determinant basis)
                hSR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);  
                # transform to eigenbasis
                hSR_ent = wfm.utils.entangle(hSR, *pair);
                hSR_diag = np.dot( np.linalg.inv(Udiag), np.dot(hSR_ent, Udiag));
                # force diagonal
                if((j not in impis)):
                    hSR_diag = np.diagflat(np.diagonal(hSR_diag));
                hblocks.append(np.copy(hSR_diag));
                if(verbose > 3 and Eval == Evals[0]):
                    print("\nJK1, JK2 = ",JK1, JK2);
                    print(" - ham:\n", np.real(hSR));
                    print(" - transformed ham:\n", np.real(hSR_diag));

            # finish hblocks
            hblocks = np.array(hblocks);
            #hblocks[1] += Vg*np.eye(len(source)); # Vg shift in SR
            #hblocks[2] += Vg*np.eye(len(source));
            E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
            for hb in hblocks:
                hb += -E_shift*np.eye(np.shape(hblocks[0])[0]);

            # hopping
            tnn = np.array([-tl*np.eye(len(source)),-tp*np.eye(len(source)),-tl*np.eye(len(source))]);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # get R, T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source);
            Rvals[Evali] = Rdum;
            Tvals[Evali] = Tdum;

            # von Neumann entropy (should be same for |+'> and |-'> states)
            if(Eval == Evals[0]):
                for coli in range(len(Udiag)):
                    get_VNE(Udiag[:,coli],epsilons[coli]);

                # manually check |-'> state
                # coefs for |-'> in the plus, minus basis
                plus_coef = 1-np.sqrt(1+beta*beta);
                minus_coef = beta;
                normalization = np.sqrt(2*(1+beta*beta-np.sqrt(1+beta*beta))) #np.sqrt( plus_coef*np.conj(plus_coef)+minus_coef*np.conj(minus_coef))
                plus_coef, minus_coef = plus_coef/normalization, minus_coef/normalization;

                # now convert into the up down, down up basis
                updown_coef = (1/np.sqrt(2))*(plus_coef + minus_coef);
                downup_coef = (1/np.sqrt(2))*(plus_coef - minus_coef);
                print("\n",40*"*");
                print("Manual eigenstate |-'> = ",{'01':updown_coef,'10':downup_coef});

                # eigenenergy
                print("Eigenenergy = ",(1.5*1.5-1.5)*(2*Dmid - J12z)+Dmid - 1.5*J12x*np.sqrt(1+beta*beta));

                # project onto imp 1
                rho1 = np.array([[updown_coef*np.conj(updown_coef),0],[0,downup_coef*np.conj(downup_coef)]]);
                rho1_log2 = np.diagflat(np.log2(np.diagonal(rho1))); # since it is diagonal, log operation can be vectorized
                VNE = -np.trace(np.dot(rho1,rho1_log2));
                print('VNE = ',VNE,'\n');

                # coefs for |+'> in the plus, minus basis
                plus_coef_p = 1+np.sqrt(1+beta*beta);
                minus_coef_p = beta;
                normalization_p = np.sqrt(2*(1+beta*beta+np.sqrt(1+beta*beta))) #np.sqrt( plus_coef*np.conj(plus_coef)+minus_coef*np.conj(minus_coef))
                plus_coef_p, minus_coef_p = plus_coef_p/normalization_p, minus_coef_p/normalization_p;
                updown_coef_p = (1/np.sqrt(2))*(plus_coef_p + minus_coef_p);
                downup_coef_p = (1/np.sqrt(2))*(plus_coef_p - minus_coef_p);

                # find the general transmitted state (ignoring complex phases!!)
                Tpp, Tmp, Tip = Tvals[Evali,pair[0]],Tvals[Evali,pair[1]],Tvals[Evali,sourcei];
                transmitted_state_prime = {'100':np.sqrt(Tip),'001':np.sqrt(Tpp)*updown_coef_p + np.sqrt(Tmp)*updown_coef,
                                                            '010':np.sqrt(Tpp)*downup_coef_p + np.sqrt(Tmp)*downup_coef};
                print("\n",40*"*");
                print("Transmitted state = ",transmitted_state_prime,"\n")
         
        # save data to .npy for each DeltaD/J12 val
        data = np.zeros((2+2*len(source),len(Evals)));
        data[0,0] = tl;
        data[0,1] = JK;
        data[1,:] = Evals;
        data[2:2+len(source),:] = Tvals.T;
        data[2+len(source):2+2*len(source),:] = Rvals.T;
        fname = "data/broken/beta"+str(beta);
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
    myTvals = data[2:5];
    myRvals = data[5:];
    mytotals = np.sum(myTvals, axis = 0) + np.sum(myRvals, axis = 0);
    print("- shape xvals = ", np.shape(myxvals));
    print("- shape Tvals = ", np.shape(myTvals));
    print("- shape Rvals = ", np.shape(myRvals));
    return myxvals, myRvals, myTvals, mytotals;

num_subplots = 2
fig, (mainax, fomax) = plt.subplots(num_subplots, sharex = True);
fig.set_size_inches(7/2,3*num_subplots/2);
datafs = sys.argv[1:];
for fi in range(len(datafs)):
    xvals, Rvals, Tvals, totals = load_data(datafs[fi]);
    mymarkevery = (0,50);

    # plot T vs logE
    for pairi in range(len(pair)):
        mainax.plot(xvals, Tvals[pair[pairi]], color=mycolors[fi],linestyle=mystyles[pairi], marker=mymarkers[fi],markevery=mymarkevery, linewidth = mylinewidth);   
    #fomax.plot(xvals, Tvals[pair[0]]/Tvals[pair[pair[1]]], color=mycolors[fi]);

# format
logElims = -3,0
mainax.set_ylim(0,0.16);
mainax.set_yticks([0,0.08,0.16]);
mainax.set_ylabel('$T_\sigma$',fontsize = myfontsize);
mainax.set_title(mypanels[0], x=0.07, y = 0.7, fontsize = myfontsize);
fomax.set_xscale('log', subs = []);
fomax.set_xlim(10**(logElims[0]),10**(logElims[1]));
fomax.set_xticks([10**(logElims[0]),10**(logElims[1])]);
fomax.set_xlabel('$K_i/t$', fontsize = myfontsize);
fomax.set_ylim(0,0.16);
fomax.set_yticks([0,0.08,0.16]);
fomax.set_ylabel('$\overline{p^2}(\\tilde{\\theta})$', fontsize = myfontsize);
fomax.set_title(mypanels[1], x=0.07, y = 0.7, fontsize = myfontsize);
plt.tight_layout();
plt.savefig('figs/broken.pdf');
plt.show();