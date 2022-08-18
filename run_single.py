'''
Christian Bunker
M^2QM at UF
September 2021

Quasi 1 body transmission through spin impurities project, part 0:
Scattering of a single electron from a spin-1/2 impurity

wfm.py
- Green's function solution to transmission of incident plane wave
- left leads, right leads infinite chain of hopping tl treated with self energy
- in the middle is a scattering region, hop on/off with th usually = tl
- in SR the spin degrees of freedom of the incoming electron and spin impurities are coupled 
'''

from code import wfm, fci_mod, ops
from code.wfm import utils

import numpy as np
import matplotlib.pyplot as plt

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;

# fig standardizing
myxvals = 199;
myfontsize = 14;
mycolors = ["black","darkblue","darkgreen","darkred", "darkmagenta","darkgray","darkcyan"];
mymarkers = ["o","^","s","d","X","P","*"];
mymarkevery = 50;
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)"];
plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

# tight binding params
tl = 1.0;
th = 1.0;
Delta = 0.0; # inelastic splitting

#################################################################
#### replication of continuum solution

if True:
    num_plots = 3
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);

    # iter over effective J
    Jvals = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0];
    plot_Jvals = [0.1,0.2,0.4];
    criticalEs = np.full(len(Jvals),-1.0, dtype = float);
    critical_diff = 0.1;
    for Ji in range(len(Jvals)):
        Jeff = Jvals[Ji];
        
        # 2nd qu'd operator for S dot s
        h1e = np.zeros((4,4))
        g2e = wfm.utils.h_kondo_2e(Jeff, 0.5); # J, spin
        states_1p = [[0,1],[2,3]]; # [e up, down], [imp up, down]
        hSR = fci_mod.single_to_det(h1e, g2e, np.array([1,1]), states_1p); # to determinant form

        # zeeman splitting
        hzeeman = np.array([[Delta, 0, 0, 0],
                        [0,0, 0, 0],
                        [0, 0, Delta, 0], # spin flip gains PE delta
                        [0, 0, 0, 0]]);
        hSR += hzeeman;

        # truncate to coupled channels
        hSR = hSR[1:3,1:3];
        hzeeman = hzeeman[1:3,1:3];

        # leads
        hLL = np.copy(hzeeman);
        hRL = np.copy(hzeeman)

        # source = up electron, down impurity
        sourcei, flipi = 0,1
        source = np.zeros(np.shape(hSR)[0]);
        source[sourcei] = 1;

        # package together hamiltonian blocks
        hblocks = [hLL,hSR];
        hblocks.append(hRL);
        hblocks = np.array(hblocks);

        # hopping
        tnn = [-th*np.eye(*np.shape(hSR)),-th*np.eye(*np.shape(hSR))]; # on and off imp
        tnn = np.array(tnn);
        tnnn = np.zeros_like(tnn)[:-1];
        if(verbose and Jeff == 0.1): print("\nhblocks:\n", hblocks, "\ntnn:\n", tnn,"\ntnnn:\n",tnnn);

        # sweep over range of energies
        # def range
        logElims = -3,0
        Evals = np.logspace(*logElims,myxvals);
        Rvals = np.empty((len(Evals),len(source)), dtype = float);
        Tvals = np.empty((len(Evals),len(source)), dtype = float); 
        for Evali in range(len(Evals)):

            # energy
            Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper

            if(Evali < 1): # verbose
                Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, all_debug = False, verbose = verbose);
            else: # not verbose
                 Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, all_debug = False);
            Rvals[Evali] = Rdum;
            Tvals[Evali] = Tdum;

            # find the critical energy
            menez_T = Jeff*Jeff/(16*tl*Eval);
            my_T = Tdum[1];
            if abs(my_T - menez_T)/menez_T < 0.1 and criticalEs[Ji] == -1.0: # critical and not found yet
                criticalEs[Ji] = Eval; # > 0 always

        if Jvals[Ji] in plot_Jvals: # only plot some  
            # plot Tvals vs E
            colori = plot_Jvals.index(Jeff);
            axes[0].plot(Evals,Tvals[:,flipi], color = mycolors[colori], marker = mymarkers[colori], markevery = mymarkevery, linewidth = mylinewidth);
            axes[1].plot(Evals,Tvals[:,sourcei], color = mycolors[colori], marker = mymarkers[colori], markevery = mymarkevery, linewidth = mylinewidth);
            totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
            #axes[1].plot(Evals, totals, color="red", label = "total ");
            # menezes prediction for continuum
            axes[0].plot(Evals, Jeff*Jeff/(16*tl*Evals), color = mycolors[colori],linestyle = "dashed", marker = mymarkers[colori], markevery = mymarkevery, linewidth = mylinewidth);

    # critical E's
    axes[2].scatter(criticalEs, Jvals, marker = 'D', color = "red");
    # best fit
    slope, intercept = np.polyfit(criticalEs, Jvals, deg=1);
    axes[2].plot(criticalEs, intercept+slope*criticalEs, color = "red");
    print("slope = ",slope,", intercept = ", intercept);
    print("K slope = ",1/slope,", K intercept = ", -intercept/slope);

    # format
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    axes[0].set_ylim(0,0.4)
    axes[0].set_ylabel('$T_{flip}$', fontsize = myfontsize );
    axes[1].set_ylim(0,1.01);
    axes[1].set_yticks([0,0.5,1.0]);
    axes[1].set_ylabel('$T_{i}$', fontsize = myfontsize );
    axes[2].set_ylabel('$J$', fontsize = myfontsize );
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    plt.savefig('figs/continuum.pdf');
    plt.show();


#################################################################
#### physical origin

if False:
    num_plots = 2
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);

    # iter over effective J by changing epsilon
    epsilons = [-27.5,-11.3,-5.3];
    for epsi in range(len(epsilons)):
        epsilon = epsilons[epsi];

        # tight binding parameters
        th = 1.0;
        U1 = 0.0
        U2 = 100.0;
        Jeff = 2*th*th*U2/((U1-epsilon)*(U2+epsilon)); #exactly as in my paper
        print("Jeff = ",Jeff);

        # SR physics: site 1 is in chain, site 2 is imp with large U
        hSR = np.array([[0,-th,th,0], # up down, -
                        [-th,epsilon, 0,-th], # up, down (source)
                        [th, 0, epsilon, th], # down, up (flip)
                        [0,-th,th,U2+2*epsilon]]); # -, up down

        # source = up electron, down impurity
        source = np.zeros(np.shape(hSR)[0]);
        sourcei, flipi = 1,2;
        source[sourcei] = 1;

        # lead physics
        hLL = np.diagflat([0,epsilon, epsilon, 2*epsilon]);
        hRL = np.diagflat([0,epsilon, epsilon, 2*epsilon]);

        # package together hamiltonian blocks
        hblocks = np.array([hLL, hSR, hRL]);
        for hb in hblocks: hb += -epsilon*np.eye(len(source));  # constant shift so source is at zero
        tnn_mat = -tl*np.array([[0,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0]]);
        tnn = np.array([np.copy(tnn_mat), np.copy(tnn_mat)]);
        tnnn = np.zeros_like(tnn)[:-1];
        #if verbose: print("\nhblocks:\n", hblocks, "\ntnn:\n", tnn, "\ntnnn:", tnnn)

        if True: # do the downfolding explicitly
            matA = np.array([[0, 0],[0,0]]);
            matB = np.array([[-th,-th],[th,th]]);
            matC = np.array([[-th,th],[-th,th]]);
            matD = np.array([[-epsilon, 0],[0,U2+epsilon]]);
            mat_downfolded = matA - np.dot(matB, np.dot(np.linalg.inv(matD), matC))  
            #print("mat_df = \n",mat_downfolded);
            Jeff = 2*abs(mat_downfolded[0,0]);
            print(">>>Jeff = ",Jeff);
            mat_downfolded += np.eye(2)*Jeff/4
            #print("mat_df = \n",mat_downfolded);
        
        # sweep over range of energies
        # def range
        logElims = -3,0
        Evals = np.logspace(*logElims,myxvals);
        Rvals = np.empty((len(Evals),len(source)), dtype = float);
        Tvals = np.empty((len(Evals),len(source)), dtype = float);
        for Evali in range(len(Evals)):

            # energy
            Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper

            Rdum, Tdum =wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source);
            Rvals[Evali] = Rdum;
            Tvals[Evali] = Tdum;

        # plot Tvals vs E
        axes[0].plot(Evals,Tvals[:,flipi], color = mycolors[epsi], marker = mymarkers[epsi], markevery = mymarkevery, linewidth = mylinewidth);
        axes[1].plot(Evals,Tvals[:,sourcei], color = mycolors[epsi], marker = mymarkers[epsi], markevery = mymarkevery, linewidth = mylinewidth);
        totals = Tvals[:,sourcei] + Tvals[:,flipi] + Rvals[:,sourcei] + Rvals[:,flipi];
        #axes[1].plot(Evals, totals, color="red");

        # menezes prediction in the continuous case
        axes[0].plot(Evals, Jeff*Jeff/(16*np.real(Evals)), color = mycolors[epsi], marker = mymarkers[epsi], markevery = mymarkevery, linestyle = "dashed", linewidth = mylinewidth);

    # format
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    axes[0].set_ylim(0,0.4)
    axes[0].set_ylabel('$T_{flip}$', fontsize = myfontsize );
    axes[1].set_ylim(0,1.01);
    axes[1].set_yticks([0,0.5,1.0]);
    axes[1].set_ylabel('$T_{i}$', fontsize = myfontsize );
    plt.tight_layout();
    plt.savefig('figs/origin.pdf');
    plt.show();








