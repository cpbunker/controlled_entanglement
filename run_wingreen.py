'''

'''

import fcdmft

import numpy as np
import matplotlib.pyplot as plt

#### top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;
iE = 1e-3; # small imag part


##### 1: set up the impurity + leads system

# anderson dot
Vg = -0.5;
U = 2.0;
h1e = np.array([[[Vg,0],[0,-Vg]]]); # on site energy
g2e = np.zeros((1,2,2,2,2));
g2e[0][0,0,1,1] += U; # coulomb
g2e[0][1,1,0,0] += U;

# embed in semi infinite leads (noninteracting, nearest neighbor only)
tl = 1.0; # lead hopping
Vb = 0.001; # bias
th = 0.5; # coupling between imp, leads
coupling = np.array([[[-th, 0],[0,-th]]]); # ASU

# left lead
H_LL = np.array([[[Vb/2,0],[0,Vb/2]]]);
V_LL = np.array([[[-tl,0],[0,-tl]]]); # spin conserving hopping
LLphys = (H_LL, V_LL, np.copy(coupling), Vb/2); # pack

# right lead
H_RL = np.array([[[-Vb/2,0],[0,-Vb/2]]]);
V_RL = np.array([[[-tl,0],[0,-tl]]]); # spin conserving hopping
RLphys = (H_RL, V_RL, np.copy(coupling), -Vb/2); # pack

# energy spectrum
cutoff = 1e-5; # how far around 0 to avoid
Es = np.append(np.linspace(-Vb, -cutoff, 99), np.linspace(cutoff, Vb, 99));
iE = 1e-2;

# temp
kBT = 0.0;

#### 2: compute the many body green's function for imp + leads system
MBGF = fcdmft.kernel(Es, iE, h1e, g2e, LLphys, RLphys, solver = 'fci', n_bath_orbs = 4, verbose = verbose);

#### 3: use meir wingreen formula
jE = fcdmft.wingreen(Es, iE, kBT, MBGF, LLphys, RLphys, verbose = verbose);

# also try landauer formula
jE_land = fcdmft.landauer(Es, iE, kBT, MBGF, LLphys, RLphys, verbose = verbose);

plt.plot(Es, np.real(jE));
plt.plot(Es, np.real(jE_land));
plt.title((np.pi/Vb)*np.trapz(np.real(jE), Es) );
plt.show();

