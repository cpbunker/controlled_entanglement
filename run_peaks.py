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
def mymarkevery(fname,yvalues):
    if '-' in fname or '0.0.npy' in fname:
        return (40,40);
    else:
        return [np.argmax(yvalues)];
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

#### data
real = True;

peaks12 = np.array([ [ 0.00 , 0.222 , 0.298 ]]);

#### real data
peaks12_real = np.zeros_like(peaks12); 
peaks1_real = np.zeros_like(peaks12);
peaks32_real = np.array([
    [ 0.020, 0.003, 0.052 ]]); # MnPc
peaks72_real = np.array([
    [ 0.006327, 0.017, 0.126 ]]); # Mn4_72
peaks4_real = np.array([
    [ 0.005645, 0.016, 0.122 ]]); # Mn2
peaks92_real = np.array([
    [ 0.004963, 0.018, 0.131 ]]); # Mn4
peaks6_real = np.array([        
    [ 0.003, 0.021, 0.138 ]]);  # Mn3

if real: peaks12, peaks1, peaks32, peaks72, peaks4, peaks92, peaks6 = peaks12_real, peaks1_real, peaks32_real, peaks72_real, peaks4_real, peaks92_real, peaks6_real;

#### plot T+ and p2 vs Delta E
if True:
    num_plots = 2;
    fig, axes = plt.subplots(num_plots, sharex=True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);
    indE, indT, indp = 0,1,2;

    # convert to meV
    convert = 100;

    # plot
    dataindex = [indT, indp];
    for axi in range(len(axes)):
        # for s=1/2
        axes[axi].scatter(convert*peaks12[:,indE], peaks12[:,dataindex[axi]], color=mycolors[0], marker = mymarkers[0], linewidth = mylinewidth);
        # for s=1
        axes[axi].scatter(convert*peaks1[:,indE], peaks1[:,dataindex[axi]], color=mycolors[1], marker=mymarkers[1], linewidth = mylinewidth);
        # for s=3/2
        axes[axi].scatter(convert*peaks32[:,indE], peaks32[:,dataindex[axi]], color=mycolors[2], marker=mymarkers[2], linewidth = mylinewidth);
        # for s=7/2
        axes[axi].scatter(convert*peaks72[:,indE], peaks72[:,dataindex[axi]], color=mycolors[3], marker=mymarkers[2], linewidth = mylinewidth);
        # for s=4
        axes[axi].scatter(convert*peaks4[:,indE], peaks4[:,dataindex[axi]], color=mycolors[4], marker=mymarkers[3], linewidth = mylinewidth);
        # for s=9/2
        axes[axi].scatter(convert*peaks92[:,indE], peaks92[:,dataindex[axi]], color=mycolors[5], marker=mymarkers[2], linewidth = mylinewidth);
        # for s=6
        axes[axi].scatter(convert*peaks6[:,indE], peaks6[:,dataindex[axi]], color=mycolors[6], marker=mymarkers[4], linewidth = mylinewidth);
           
    # format
    lower_y = 0.08
    axes[0].set_ylim(-lower_y*0.24,0.24);
    axes[0].set_ylabel('max($T_+$)', fontsize = myfontsize);
    axes[1].set_ylim(0.0,0.32);
    axes[1].set_ylabel('max($\overline{p^2}$)', fontsize = myfontsize);

    # show
    axes[-1].set_xlabel('$\Delta E/t$',fontsize = myfontsize);
    if real: axes[-1].set_xlabel('$\Delta E$ (meV)',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    fname = 'figs/peaks.pdf';
    if real: fname = 'figs/peaks_real.pdf'
    plt.savefig(fname);
    plt.show();



