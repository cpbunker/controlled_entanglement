'''
Christian Bunker
M^2QM at UF
September 2021

Wave function matching (WFM) in 1D
general formalism: all sites map to all the different
degrees of freedom of the system
'''

from code import fci_mod

import numpy as np

##################################################################################
#### driver of transmission coefficient calculations

def kernel(h, tnn, tnnn, tl, E, Ajsigma, verbose = 0, all_debug = True):
    '''
    coefficient for a transmitted up and down electron
    Args
    -h, array, block hamiltonian matrices
    -tnn, array, nearest neighbor block hopping matrices
    -tnnn, array, next nearest neighbor block hopping matrices
    -tl, float, hopping in leads, not necessarily same as hopping on/off SR
        or within SR which is defined by th matrices
    -E, float, energy of the incident electron
    -Ajsigma, incident particle amplitude at site 0 in spin channel j
    Optional args
    -verbose, how much printing to do
    -all_debug, whether to enforce a bunch of extra assert statements

    Returns
    tuple of R coefs (vector of floats for each sigma) and T coefs (likewise)
    '''
    if(not isinstance(h, np.ndarray)): raise TypeError;
    if(not isinstance(tnn, np.ndarray)): raise TypeError;
    if(not isinstance(tnnn, np.ndarray)): raise TypeError;

    
    # check that lead hams are diagonal
    for hi in [0, -1]: # LL, RL
        isdiag = h[hi] - np.diagflat(np.diagonal(h[hi])); # subtract off diag
        if(all_debug and np.any(isdiag)): # True if there are nonzero off diag terms
            raise Exception("Not diagonal\n"+str(h[hi]))
    for i in range(len(Ajsigma)): # always set incident mu = 0
        if(Ajsigma[i] != 0):
            assert(h[0,i,i] == 0);

    # check incident amplitude
    assert( isinstance(Ajsigma, np.ndarray));
    assert( len(Ajsigma) == np.shape(h[0])[0] );
    sigma0 = -1; # incident spin channel
    for sigma in range(len(Ajsigma)): # ensure there is only one incident spin config
        if(Ajsigma[sigma] != 0):
            if( sigma0 != -1): # then there was already a nonzero element, bad
                raise(Exception("Ajsigma has too many nonzero elements:\n"+str(Ajsigma)));
            else: sigma0 = sigma;
    assert(sigma0 != -1);

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0];

    # determine velocities in the left, right leads
    ka_L = np.arccos((E-np.diagonal(h[0]))/(-2*tl)); # vector with sigma components
    ka_R = np.arccos((E-np.diagonal(h[-1]))/(-2*tl));
    v_L = 2*tl*np.sin(ka_L); # vector with sigma components
    v_R = 2*tl*np.sin(ka_R); # a, hbar defined as 1

    # green's function
    if(verbose): print("\nEnergy = {:.6f}".format(np.real(E+2*tl))); # start printouts
    G = Green(h, tnn, tnnn, tl, E, verbose = verbose);

    # contract G with source to pick out matrix elements we need
    Avector = np.zeros(np.shape(G)[0], dtype = complex); # go from spin space to spin+site space
    for sigma in range(n_loc_dof):
        Avector[sigma] = Ajsigma[sigma]; # fill from spin space
    
    G_0sigma0 = np.dot(G, Avector); # G contracted with incident amplitude
                                    # picks out matrix elements of incident
                                    # still has 1 free spatial, spin index for transmitted

    # compute reflection and transmission coeffs
    Rs = np.zeros(n_loc_dof, dtype = float); # force as float bc we check that imag part is tiny
    Ts = np.zeros(n_loc_dof, dtype = float); 
    for sigma in range(n_loc_dof): # iter over spin dofs

        # given in appendix A of manuscript, eq:Tcoef
        T = G_0sigma0[-n_loc_dof+sigma]*np.conj(G_0sigma0[-n_loc_dof+sigma])*v_R[sigma]*v_L[sigma0];
        
        # given in appendix A of manuscript, eq:Rcoef
        R = (complex(0,1)*G_0sigma0[0+sigma]*v_L[sigma0] - Ajsigma[sigma])*np.conj(complex(0,1)*G_0sigma0[0+sigma]*v_L[sigma0] - Ajsigma[sigma])*v_L[sigma]/v_L[sigma0];  

        # benchmarking
        if(verbose > 1): print(" - sigma = "+str(sigma)+",   T = {:.4f}+{:.4f}j, R = {:.4f}+{:.4f}j"
                               .format(np.real(T), np.imag(T), np.real(R), np.imag(R)));
        # check that the imag part is tiny
        # fails if E < barrier (evanescent)
        if(all_debug and abs(np.imag(T)) > 1e-10 ): raise(Exception("T = "+str(T)+" must be real")); 
        if(all_debug and abs(np.imag(R)) > 1e-10 ): raise(Exception("R = "+str(R)+" must be real"));

        # in view of passing the above check, can drop the imag part
        Rs[sigma] = R;
        Ts[sigma] = T;
    
    return Rs, Ts;


def Hmat(h, tnn, tnnn, verbose = 0):
    '''
    Make the hamiltonian H for reduced dimensional N+2 x N+2 system
    where there are N sites in the scattering region (SR), 1 LL site, 1 RL site
    Args
    -h, 2d array, on site blocks at each of the N+2 sites of the system
    -tnn, 2d array, nearest neighbor hopping btwn sites, N-1 blocks
    -tnnn, 2d array, next nearest neighbor hopping btwn sites, N-2 blocks
    '''
    if(not len(tnn) +1 == len(h)): raise ValueError;
    if(not len(tnnn)+2 == len(h)): raise ValueError;

    # unpack
    N = len(h) - 2; # num scattering region sites, ie N+2 = num spatial dof
    n_loc_dof = np.shape(h[0])[0]; # dofs that will be mapped onto row in H
    H =  np.zeros((n_loc_dof*(N+2), n_loc_dof*(N+2) ), dtype = complex);
    # outer shape: num sites x num sites (0 <= j <= N+1)
    # shape at each site: n_loc_dof, runs over all other degrees of freedom)

    # first construct matrix of matrices
    for sitei in range(0,N+2): # iter site dof only
        for sitej in range(0,N+2): # same
                
            for loci in range(np.shape(h[0])[0]): # iter over local dofs
                for locj in range(np.shape(h[0])[0]):
                    
                    # site, loc indices -> overall indices
                    ovi = sitei*n_loc_dof + loci;
                    ovj = sitej*n_loc_dof + locj;

                    if(sitei == sitej): # input from local h to main diag
                        H[ovi, ovj] = h[sitei][loci, locj];

                    elif(sitei == sitej+1): # input from tnn to lower diag
                        H[ovi, ovj] = tnn[sitej][loci, locj];

                    elif(sitei+1 == sitej): # input from tnn to upper diag
                        H[ovi, ovj] = tnn[sitei][loci, locj];

                    elif(sitei == sitej+2): # input from tnnn to 2nd lower diag
                        H[ovi, ovj] = tnnn[sitej][loci, locj];

                    elif(sitei+2 == sitej): # input from tnnn to 2nd upper diag
                        H[ovi, ovj] = tnnn[sitei][loci, locj];

    return H; # end Hmat

def Hprime(h, tnn, tnnn, tl, E, verbose = 0):
    '''
    Make H' (hamiltonian + self energy) for N+2 x N+2 system
    where there are N sites in the scattering region (SR).
    Args
    -h, array, on site blocks at each of the N+2 sites of the system
    -tnn, array, nearest neighbor hopping btwn sites, N-1 blocks
    -tnnn, array, next nearest neighbor hopping btwn sites, N-2 blocks
    -tl, float, hopping in leads, distinct from hopping within SR def'd by tnn, tnnn
    '''

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0];

    # base hamiltonian
    Hp = Hmat(h, tnn, tnnn, verbose = verbose); # H matrix from SR on site, hopping blocks
    
    # self energies in LL
    # need a self energy for all incoming/outgoing spin states (all local dof)
    SigmaLs = np.zeros(n_loc_dof, dtype = complex);
    for Vi in range(n_loc_dof): # iters over all local dof
        # scale the energy
        V = h[0][Vi,Vi];
        lamL = (E-V)/(-2*tl);
        # make sure sign of SigmaL is correctly assigned
        assert( abs(np.imag(lamL)) < 1e-10);
        lamL = np.real(lamL);
        # reflected self energy
        LambdaLminus = lamL - np.lib.scimath.sqrt(lamL*lamL - 1); 
        SigmaL = -tl/LambdaLminus; 
        Hp[Vi,Vi] += SigmaL;
        SigmaLs[Vi] = SigmaL
    del V, lamL, LambdaLminus, SigmaL

    # self energies in RL
    SigmaRs = np.zeros(n_loc_dof, dtype = complex);
    for Vi in range(n_loc_dof): # iters over all local dof
        # scale the energy
        V = h[-1][Vi,Vi];     
        lamR = (E-V)/(-2*tl);
        # make sure the sign of SigmaR is correctly assigned
        assert( abs(np.imag(lamR)) < 1e-10);
        lamR = np.real(lamR); # makes sure sign of SigmaL is correctly assigned
        # transmitted self energy
        LambdaRplus = lamR + np.lib.scimath.sqrt(lamR*lamR - 1);
        SigmaR = -tl*LambdaRplus;
        Hp[Vi-n_loc_dof,Vi-n_loc_dof] += SigmaR;
        SigmaRs[Vi] = SigmaR;
    del V, lamR, LambdaRplus, SigmaR;

    # check that modes with given energy are allowed in some LL channels
    SigmaLs, SigmaRs = np.array(SigmaLs), np.array(SigmaRs);
    assert(np.any(np.imag(SigmaLs)) );
    for sigmai in range(len(SigmaLs)):
        if(abs(np.imag(SigmaLs[sigmai])) > 1e-10 and abs(np.imag(SigmaRs[sigmai])) > 1e-10 ):
            assert(np.sign(np.imag(SigmaLs[sigmai])) == np.sign(np.imag(SigmaRs[sigmai])));
    if(verbose > 3):
        ka_L = np.arccos((E-np.diagonal(h[0]))/(-2*tl)); # vector running over sigma
        ka_R = np.arccos((E-np.diagonal(h[-1]))/(-2*tl));
        v_L = 2*tl*np.sin(ka_L); # a/hbar defined as 1
        v_R = 2*tl*np.sin(ka_R);
        for sigma in range(len(ka_L)):
            print(" - sigma = "+str(sigma)+", v_L = {:.4f}+{:.4f}j, Sigma_L = {:.4f}+{:.4f}j"
                  .format(np.real(v_L[sigma]), np.imag(v_L[sigma]), np.real(SigmaLs[sigma]), np.imag(SigmaLs[sigma])));
            print(" - sigma = "+str(sigma)+", v_R = {:.4f}+{:.4f}j, Sigma_R = {:.4f}+{:.4f}j"
                  .format(np.real(v_R[sigma]), np.imag(v_R[sigma]), np.real(SigmaRs[sigma]), np.imag(SigmaRs[sigma])));

    return Hp;


def Green(h, tnn, tnnn, tl, E, verbose = 0):
    '''
    Greens function for system described by
    Args
    -h, array, on site blocks at each of the N+2 sites of the system
    -tnn, array, nearest neighbor hopping btwn sites, N-1 blocks
    -tnnn, array, next nearest neighbor hopping btwn sites, N-2 blocks
    -tl, float, hopping in leads, distinct from hopping within SR def'd by above arrays
    -E, float, incident energy
    '''

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0];

    # get green's function matrix
    Hp = Hprime(h, tnn, tnnn, tl, E, verbose = verbose);
    #if(verbose): print(">>> H' = \n", Hp );
    #if(verbose): print(">>> EI - H' = \n", E*np.eye(np.shape(Hp)[0]) - Hp );
    G = np.linalg.inv( E*np.eye(np.shape(Hp)[0] ) - Hp );

    # of interest is the qith row which contracts with the source q
    return G;



##################################################################################
#### test code

if __name__ == "__main__":

    pass;





    
    


    








