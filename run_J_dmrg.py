'''
Christian Bunker
M^2QM at UF
September 2021

Runner file for simulating current through SIAM, weak biasing, using td-dmrg
'''

import siam_current

import numpy as np

import sys

##################################################################################
#### 

# top level
verbose = 5;
nleads = (2,2);
nelecs = (sum(nleads)+1,0); # half filling
ndots = 1;
get_data = int(sys.argv[1]); # whether to run computations, if not data already exists

# phys params, must be floats
tl = 1.0;
th = 0.1; 
td = 0.0; # because there is only one dot
Vb = 0.01;
mu = 0.1;
Vgs = [-1.0];
U = 2.0
B = 0.0;
theta = 0.0;

#time info
dt = 0.1;
tf = 1.0;

# dmrg info
bdims = [300, 400, 450, 500];
noises = [1e-3, 1e-4, 1e-5, 0];

if get_data: # must actually compute data

    for i in range(len(Vgs)): # iter over Vg vals;
        Vg = Vgs[i];
        params = tl, th, td, Vb, mu, Vg, U, B, theta;
        siam_current.DotDataDmrg(nleads, nelecs, ndots, tf, dt, params, bdims, noises, prefix = "", verbose = verbose);

else:

    import plot

    # plot results
    datafs = sys.argv[2:]
    labs = Vgs # one label for each Vg
    splots = ['Jup','Jdown','occ','delta_occ','Szleads','Sz']; # which subplots to plot
    title = "Current between impurity and left, right leads"
    plot.CompObservables(datafs, labs, splots = splots, mytitle = title, leg_title = "$V_g$", leg_ncol = 3);

    








