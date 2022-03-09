'''
Christian Bunker
M^2QM at UF
September 2021

Quasi 1 body transmission through spin impurities project, part 2:
Scattering of single electron off of two localized spin-1/2 impurities
Following cicc, imp spins are confined to single sites, separated by x0
    imp spins can flip
    e-imp interactions treated by (effective) J Se dot Si
    look for resonances in transmission as function of kx0 for fixed E, k
    ie as impurities are pulled further away from each other
    since this is discrete, separate by x0 = N0 a lattice spacings
'''

from code import wfm
from code.wfm import utils

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5

##################################################################################
#### transparency / detection
    
if False: # original version of 2b (varying x0 by varying N)

    # tight binding params
    tl = 10.0;
    Jeff = 0.1;

    # cicc inputs
    rhoJa = 2.0; # integer that cicc param rho*J is set equal to
    E_rho = Jeff*Jeff/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                            # this E is measured from bottom of band !!!
    k_rho = np.arccos((E_rho - 2*tl)/(-2*tl)); # input E measured from 0 by -2*tl
    assert(abs((E_rho - 2*tl) - -2*tl*np.cos(k_rho)) <= 1e-8 ); # check by getting back energy measured from bottom of band
    print("E, E - 2t, J, E/J = ",E_rho, E_rho -2*tl, Jeff, E_rho/Jeff);
    print("k*a = ",k_rho); # a =1
    print("rho*J = ", (Jeff/np.pi)/np.sqrt(E_rho*tl));
    E_rho = E_rho - 2*tl; # measure from mu

    # choose boundary condition
    source = np.zeros(8);
    source[1] = 1/np.sqrt(2);
    source[2] = -1/np.sqrt(2);
    spinstate = "psimin";
    
    # mesh of x0s (= N0s * alat)
    kx0max = 2.1*np.pi;
    N0max = 1+int(kx0max/(k_rho)); # a = 1
    if verbose: print("N0max = ",N0max);
    N0vals = np.linspace(2, N0max, 299, dtype = int); # always integer
    kx0vals = k_rho*(N0vals-1); # a = 1

    # iter over all the differen impurity spacings, get transmission
    Tvals = []
    for N0 in N0vals:

        # construct hams
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks also
        hblocks, tnn = wfm.utils.h_cicc_eff(Jeff, tl, 1, N0, N0+2);
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

        # get T from this setup
        Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E_rho , source));

    # package into one array
    Tvals = np.array(Tvals);
    if(verbose): print("shape(Tvals) = ",np.shape(Tvals));
    info = np.zeros_like(kx0vals);
    info[0], info[1], info[2], info[3] = tl, Jeff, rhoJa, k_rho; # save info we need
    data = [info, kx0vals];
    for Ti in range(np.shape(Tvals)[1]):
        data.append(Tvals[:,Ti]); # data has columns of N0val, k0val, corresponding T vals
    # save data
    fname = "dat/cicc/"+spinstate+"/";
    fname = "";
    fname +="N_rhoJa"+str(int(np.around(rhoJa)))+".npy";
    np.save(fname,np.array(data));
    if verbose: print("Saved data to "+fname);
    raise(Exception);


if False: # vary kx0 by varying Vgate
    
    # tight binding params
    tl = 1.0; # norm convention, -> a = a0/sqrt(2) = 0.37 angstrom
    Jeff = 0.1; # eff heisenberg

    # cicc quantitites
    N_SR = 100 #100, 198, 988;
    ka0 = np.pi/(N_SR - 1); # a' is length defined by hopping t' btwn imps
                            # ka' = ka'0 = ka0 when t' = t so a' = a
    E_rho = 2*tl-2*tl*np.cos(ka0); # energy of ka0 wavevector, which determines rhoJa
                                    # measured from bottom of the band!!
    rhoJa = Jeff/(np.pi*np.sqrt(tl*E_rho));

    # diagnostic
    if(verbose):
        print("\nCiccarello inputs")
        print(" - E, J, E/J = ",E_rho, Jeff, E_rho/Jeff);
        print(" - ka0 = ",ka0);
        print("- rho*J*a = ", rhoJa);
    # choose boundary condition
    source = np.zeros(8);
    source[1] = 1/np.sqrt(2);
    source[2] = -1/np.sqrt(2);
    spinstate = "psimin";

    # get data
    kalims = (0.0*ka0,2.1*ka0);
    kavals = np.linspace(*kalims, 99);
    Vgvals = -2*tl*np.cos(ka0) + 2*tl*np.cos(kavals);
    Tvals = [];
    for Vg in Vgvals:

        # construct blocks of hamiltonian
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks also
        hblocks, tnn = wfm.utils.h_cicc_eff(Jeff, tl, 1, N_SR, N_SR+2);
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

        # add gate voltage in SR
        for blocki in range(len(hblocks)): 
            if(blocki > 0 and blocki < N_SR + 1): # if in SR
                hblocks[blocki] += Vg*np.eye(np.shape(hblocks[0])[0])
                
        # get data
        Energy = -2*tl*np.cos(ka0);
        Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source));

    # package into one array
    Tvals = np.array(Tvals);
    if(verbose): print("shape(Tvals) = ",np.shape(Tvals));
    info = np.zeros_like(kavals);
    info[0], info[1], info[2], info[3] = tl, Jeff, rhoJa, ka0; # save info we need
    data = [info, kavals*(N_SR-1)];
    for Ti in range(np.shape(Tvals)[1]):
        data.append(Tvals[:,Ti]); # data has columns of kaval, corresponding T vals
    # save data
    fname = "dat/cicc/"+spinstate+"/";
    fname +="Vg_rhoJa"+str(int(np.around(rhoJa)))+".npy";
    np.save(fname,np.array(data));
    if verbose: print("Saved data to "+fname);
    raise(Exception);


if False: # vary kx0 by varing k at fixed N

    # tight binding params
    tl = 1.0; # norm convention, -> a = a0/sqrt(2) = 0.37 angstrom
    Jeff = 0.1;

    # choose boundary condition
    source = np.zeros(8);
    source[1] = 1/np.sqrt(2);
    source[2] = -1/np.sqrt(2);
    spinstate = "psimin";

    # cicc inputs
    N_SR = 100 #100,199,989; # num sites in SR
                # N_SR = 99, J = 0.1 gives rhoJa \approx 1, Na \approx 20 angstrom
    ka0 = np.pi/(N_SR-1); # val of ka (dimensionless) s.t. kx0 = ka*N_SR = pi
    E_rho = 2*tl-2*tl*np.cos(ka0); # energy of ka0 wavevector, which determines rhoJa
                                    # measured from bottom of the band!!
    rhoJa = Jeff/(np.pi*np.sqrt(tl*E_rho))

    # diagnostic
    if(verbose):
        print("\nCiccarello inputs")
        print(" - E, J, E/J = ",E_rho, Jeff, E_rho/Jeff);
        print(" - ka0 = ",ka0);
        print("- rho*J*a = ", rhoJa);

    # construct blocks of hamiltonian
    # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks also
    hblocks, tblocks = wfm.utils.h_cicc_eff(Jeff, tl, 1, N_SR, N_SR+2);
    raise Exception("switch from Data() to kernel()")
    
        
if False: # plot fig 2b data

    # plot each file given at command line
    fig, axes = plt.subplots();
    axes = [axes];
    datafs = sys.argv[1:];
    colors = ["black","black","black"]
    styles = ["dashdot","dashed","solid"];
    for fi in range(len(datafs)):

        # unpack
        print("Loading data from "+datafs[fi]);
        data = np.load(datafs[fi]);
        tl, Jeff, rhoJa, k_rho = data[0,0], data[0,1], data[0,2], data[0,3];
        kNavals = data[1];
        Tvals = data[2:];

        # convert T
        Ttotals = np.sum(Tvals, axis = 0);
        print(">>>",np.shape(Ttotals));

        # plot
        axes[0].plot(kNavals/np.pi, Ttotals, linewidth = 2, linestyle = styles[fi], color = colors[fi], label = "$\\rho  \, J a= $"+str(int(rhoJa)));

    # format and show
    axes[0].set_xlim(0.0,2.1);
    axes[0].set_xticks([0,1,2]);
    axes[0].set_xlabel("$ka(N-1)/\pi$", fontsize = "x-large");
    axes[0].set_ylim(0.0,1);
    axes[0].set_yticks([0,0.5,1]);
    axes[0].set_ylabel("$T$", fontsize = "x-large");
    plt.show();
    raise(Exception);


##################################################################################
#### molecular dimer regime (N = 2 fixed)

if True: # vary k'x0 by varying Vg for low energy detection, t', th != t;

    # incident state
    theta_param = 3*np.pi/4;
    phi_param = 0;
    source = np.zeros(8);
    source[1] = np.cos(theta_param);
    source[2] = np.sin(theta_param);
    spinstate = "psimin";

    # tight binding params
    tl = 1.0; # norm convention, -> a = a0/sqrt(2) = 0.37 angstrom
    tp = 1.0;
    Jeff = 0.1; # eff heisenberg
    N_SR = 2;

    factor = 988;
    ka0 =  np.pi/(N_SR - 1)/factor; # free space wavevector, should be << pi
                                    # increasing just broadens the Vg peak
    kpa0 = np.pi/(N_SR - 1)/factor; # wavevector in gated SR
    E_rho = 2*tl-2*tl*np.cos(ka0); # energy of ka0 wavevector, which determines rhoJa
                                    # measured from bottom of the band!!
    rhoJa = Jeff/(np.pi*np.sqrt(tl*E_rho));

    # diagnostic
    if(verbose):
        print("\nCiccarello inputs")
        print(" - E, J, E/J = ",E_rho, Jeff, E_rho/Jeff);
        print(" - ka0/pi = ",ka0/np.pi);
        print("- rho*J*a = ", rhoJa);

    # get data
    kpalims = (0.0*kpa0,(factor/2)*kpa0); # k'a in first 1/2 of the zone
    kpavals = np.linspace(*kpalims, 99);
    Vgvals =  -2*tl*np.cos(ka0) + 2*tp*np.cos(kpavals);
    Tvals, Rvals = [], []
    for Vg in Vgvals:

        # construct blocks of hamiltonian
        # t's vary, so have to construct hblocks, tnn list by hand
        hblocks, tnn = wfm.utils.h_cicc_eff(Jeff, tp, 0, N_SR-1, N_SR); # start with SR
        hblocks = np.append([np.zeros_like(hblocks[0])], hblocks, axis = 0); # LL block
        hblocks = np.append(hblocks, [np.zeros_like(hblocks[0])], axis = 0); # RL block
        tnn = np.append([-tl*np.eye(np.shape(hblocks)[1])], tnn, axis = 0); # coupling to LL
        tnn = np.append(tnn,[-tl*np.eye(np.shape(hblocks)[1])], axis = 0); # coupling to LL
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

        # add gate voltage in SR
        for blocki in range(len(hblocks)): 
            if(blocki > 0 and blocki < N_SR + 1): # gate voltage if in SR
                hblocks[blocki] += Vg*np.eye(np.shape(hblocks[0])[0])
                
        # get data
        Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, -2*tl*np.cos(ka0), source));
        Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, -2*tl*np.cos(ka0), source, reflect = True));
    
    Tvals, Rvals = np.array(Tvals), np.array(Rvals);
    Ttotals, Rtotals = np.sum(Tvals, axis = 1), np.sum(Rvals, axis = 1);

    # plot
    fig, axes = plt.subplots(2)
    axes[0].plot(kpavals/np.pi, Ttotals);
    axes[0].plot(kpavals/np.pi, Ttotals + Rtotals, color = "red");
    axes[1].plot(Vgvals, Ttotals);
    axes[0].set_xlabel("$k'a/\pi$", fontsize = "x-large");
    axes[0].set_ylabel("$T$", fontsize = "x-large");
    axes[1].axvline(-2*tl*np.cos(ka0) + 2*tp, color = "black");
    plt.show();
    del Vg, Tvals, Ttotals, fig, axes;

    #### vary theta, phi
    #### -> detection !
    myVg =0.0# -2*tp*np.cos(ka0); # Vg = E
    kpa = np.arccos((-2*tl*np.cos(ka0)-myVg)/(-2*tl));
    print(">>> myVg, k'a/pi = ",myVg, kpa/np.pi); 
    thetavals = np.linspace(0, np.pi, 49);
    phivals = np.linspace(0, np.pi, 49);
    Ttotals = np.zeros((len(thetavals), len(phivals)));

    # construct blocks of hamiltonian
    # have to construct hblocks, tnn list by hand
    hblocks, tnn = wfm.utils.h_cicc_eff(Jeff, tp, 0, N_SR-1, N_SR);
    hblocks = np.append([np.zeros_like(hblocks[0])], hblocks, axis = 0); # LL block
    hblocks = np.append(hblocks, [np.zeros_like(hblocks[0])], axis = 0); # RL block
    tnn = np.append([-tl*np.eye(np.shape(hblocks)[1])], tnn, axis = 0); # couple to LL
    tnn = np.append(tnn,[-tl*np.eye(np.shape(hblocks)[1])], axis = 0); # couple to RL
    tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

    # add gate voltage in SR
    for blocki in range(len(hblocks)):
        if(blocki > 0 and blocki < N_SR + 1): 
            hblocks[blocki] += myVg*np.eye(np.shape(hblocks[0])[0])

    # iter over entanglement space
    for ti in range(len(thetavals)):
        for pi in range(len(phivals)):

            thetaval = thetavals[ti];
            phival = phivals[pi];
	
            source = np.zeros(8, dtype = complex);
            source[1] = np.cos(thetaval);
            source[2] = np.sin(thetaval)*np.exp(complex(0,phival));

            # get data
            Ttotals[ti, pi] = sum(wfm.kernel(hblocks, tnn, tnnn, tl, -2*tl*np.cos(ka0), source));

    # plot
    fig = plt.figure();
    ax = fig.add_subplot(projection = "3d");
    thetavals, phivals = np.meshgrid(thetavals, phivals);
    ax.plot_surface(thetavals/np.pi, phivals/np.pi, Ttotals.T, cmap = cm.coolwarm);
    ax.set_xlim(0,1);
    ax.set_xticks([0,1/4,1/2,3/4,1]);
    ax.set_ylabel("$\phi/\pi$", fontsize = "x-large");
    ax.set_ylim(0,1);
    ax.set_yticks([0,1/4,1/2,3/4,1]);
    ax.set_xlabel("$\\theta/\pi$", fontsize = "x-large");
    ax.set_zlim(0,1);
    ax.set_zticks([0,1]);
    ax.set_zlabel("$T$", fontsize = "x-large");
    plt.show();
    raise(Exception);





    
