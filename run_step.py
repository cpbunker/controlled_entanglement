'''
Christian Bunker
M^2QM at UF
September 2021

Quasi 1 body transmission through spin impurities project, part 0:
Scattering of a single electron from a step potential

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

# tight binding params
tl = 1.0;
Vb = 0.1; # barrier

# blocks and inter block hopping
hLL = np.array([[0]]);
hSR = np.array([[Vb]]);
hRL = np.array([[Vb]]);
hblocks = np.array([hLL, hSR, hRL]);
tnn = -tl*np.array([[[1]],[[1]]]);
tnnn = np.zeros_like(tnn)[:-1];
if verbose: print("\nhblocks:\n", hblocks, "\ntnn:\n", tnn,"\ntnnn:\n",tnnn); 

# source
source = np.zeros(np.shape(hSR)[0]);
source[0] = 1;

# sweep over range of energies
Elims = np.array([-2*tl+0.001,-2*tl+3.0]);
Es = np.linspace(Elims[0], Elims[1], 100, dtype = float);
Esplus = np.real(Es + 2*tl);

# test main wfm kernel
Tvals, Rvals = [], [];
for E in Es:
    if(E+2 > 0.4 and E+2 < 0.41): # verbose
        Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E, source, verbose = verbose)); 
        # analytical comparison
        dum1 = E/(-2*tl);
        Sigma = -tl*(dum1 + np.sqrt(dum1*dum1 - 1));
        Ginv = np.array([[E-Sigma, tl, 0],[tl, E-Vb, tl],[0,tl,E-Vb-Sigma]]);
        Gmat = np.linalg.inv(Ginv);
        print(">>> G = \n", Gmat);
    else:
        Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E, source));
    Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E, source, reflect = True));

# plot Tvals vs E
Tvals, Rvals = np.array(Tvals), np.array(Rvals);
fig, ax = plt.subplots();
ax.scatter(Esplus,np.real(Tvals[:,0]), marker = 's');
ax.plot(Esplus, 4*np.lib.scimath.sqrt(Esplus*(Esplus-Vb))/np.power(np.lib.scimath.sqrt(Esplus) + np.lib.scimath.sqrt(Esplus - Vb),2));

totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
ax.plot(Esplus, totals, color="red");
        
# format and show
ax.set_xlim(min(Esplus), max(Esplus));
#ax.set_ylim(0,1);
ax.set_xlabel("$E+2t$", fontsize = "x-large");
ax.set_ylabel("$T$", fontsize = "x-large");
plt.show();
   

if False:    # test wfm discrete
    Ts = [];
    for E in Es:
        if( E == Es[-1]): # verbose
            Ts.append(wfm_tight.Tcoef(np.array([hLL[0,0],hSR[0,0],hRL[0,0]]), tl, E, verbose =verbose));
        else:
            Ts.append(wfm_tight.Tcoef(np.array([hLL[0,0],hSR[0,0],hRL[0,0]]), tl, E));
    plt.scatter(Esplus, np.real(Ts), marker = 's');
    plt.plot(Esplus, 4*np.lib.scimath.sqrt(Esplus*(Esplus-Vb))/np.power(np.lib.scimath.sqrt(Esplus) + np.lib.scimath.sqrt(Esplus - Vb),2));
    plt.show();






