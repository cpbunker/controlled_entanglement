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
mymarkevery = (40, 40);
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

# tight binding params
tl = 1.0;
Jval = 0.1;
FMbarrier = 0.0;

#################################################################
#### QD to select write in state

if True:
    num_plots = 4;
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);

    # iter over effective J
    Deltavals = np.array([0.0,0.1,0.5,1.0]);
    for Deltai in range(len(Deltavals)):
        Delta = Deltavals[Deltai];

        # left lead
        hLL = np.array([[FMbarrier,0], # up, down
                        [0,0]]);       # down, up (source)

        # LQD with zeeman splitting to stop up reflection
        hLQD = np.array([[Delta, 0],   # up, down gains PE delta
                           [0, 0] ]); # down, up
        
        # MSQ just has S dot s 
        h1e = np.zeros((4,4))
        g2e = wfm.utils.h_kondo_2e(Jval, 0.5); # J, spin
        states_1p = [[0,1],[2,3]]; # [e up, down], [imp up, down]
        hMSQ = fci_mod.single_to_det(h1e, g2e, np.array([1,1]), states_1p); # to determinant form
        hMSQ = hMSQ[1:3,1:3]; # just need up, down and down, up channels

        # RQD with zeeman splitting to select resonant state
        hRQD = np.array([[Delta, 0],   # up, down gains PE delta
                           [0, 0] ]); # down, up
        hRQD += -Delta*np.eye(np.shape(hRQD)[0]);

        # right lead
        hRL = np.zeros_like(hLL);
        
        # package together hamiltonian blocks
        hblocks = np.array([np.copy(hLL), np.copy(hLQD), np.copy(hMSQ), np.copy(hRQD), np.copy(hRL)]);

        # source = up electron, down impurity
        sourcei, flipi = 1,0
        source = np.zeros(np.shape(hLL)[0]);
        source[sourcei] = 1;

        # hopping
        tnn = [];
        for _ in range(len(hblocks)-1): tnn.append(-tl*np.eye(*np.shape(hLL)));
        tnn = np.array(tnn);
        tnnn = np.zeros_like(tnn)[:-1];
        if(verbose and Deltai == 1): print("\nhblocks:\n", hblocks, "\ntnn:\n", tnn,"\ntnnn:\n",tnnn);

        # sweep over range of energies
        # def range
        logElims = -4,0
        Evals = np.logspace(*logElims,myxvals);
        kavals = np.arccos((Evals-2*tl)/(-2*tl));
        jprimevals = Jval/(4*tl*kavals);
        menez_Tf = jprimevals*jprimevals/(1+(5/2)*jprimevals*jprimevals+(9/16)*np.power(jprimevals,4));
        menez_Tnf = (1+jprimevals*jprimevals/4)/(1+(5/2)*jprimevals*jprimevals+(9/16)*np.power(jprimevals,4));
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

        # plot tight binding results
        axes[0].plot(Evals,Tvals[:,flipi], color = mycolors[Deltai], marker = mymarkers[Deltai], markevery = mymarkevery, linewidth = mylinewidth);
        axes[0].set_ylabel("$T_f$", fontsize = myfontsize);
        axes[1].plot(Evals,Rvals[:,flipi], color = mycolors[Deltai], marker = mymarkers[Deltai], markevery = mymarkevery, linewidth = mylinewidth);
        axes[1].set_ylabel("$R_f$", fontsize = myfontsize);
        axes[2].plot(Evals,Tvals[:,sourcei], color = mycolors[Deltai], marker = mymarkers[Deltai], markevery = mymarkevery, linewidth = mylinewidth);
        axes[2].set_ylabel("$T_{nf}$", fontsize = myfontsize);
        axes[3].plot(Evals,Rvals[:,sourcei], color = mycolors[Deltai], marker = mymarkers[Deltai], markevery = mymarkevery, linewidth = mylinewidth);
        axes[3].set_ylabel("$R_{nf}$", fontsize = myfontsize);
        totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
        axes[-1].plot(Evals, totals, color="red", label = "total ");
   
    # show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.065, y = 0.74, fontsize = myfontsize); 
    plt.tight_layout();
    plt.show();








