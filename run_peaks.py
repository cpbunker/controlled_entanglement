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
real = False;

peaks12 = np.array([ [ 0.00 , 0.222 , 0.298 ]]);
peaks1 = np.array([
    [-0.02 , 0.017 , 0.103 ],
    [-0.01 , 0.024 , 0.122 ],
    [ 0.00 , 0.160 , 0.264 ],
    [ 0.01 , 0.015 , 0.121 ],
    [ 0.02 , 0.007 , 0.084 ]]);
peaks32 = np.array([
    [-0.02 , 0.017 , 0.103 ],
    [-0.01 , 0.023 , 0.121 ],
    [ 0.00 , 0.122 , 0.239 ],
    [ 0.01 , 0.021 , 0.141 ],
    [ 0.02 , 0.010 , 0.100 ]]);
peaks92 = np.array([
    [-0.02 , 0.016 , 0.101 ],
    [-0.01 , 0.022 , 0.115 ],
    [ 0.00 , 0.050 , 0.163 ],
    [ 0.01 , 0.024 , 0.147 ],
    [ 0.02 , 0.018 , 0.131 ]]);
peaks6 = np.array([
    [-0.02 , 0.016 , 0.099 ],
    [-0.01 , 0.021 , 0.112 ],
    [ 0.00 , 0.038 , 0.145 ],
    [ 0.01 , 0.023 , 0.141 ],
    [ 0.02 , 0.018 , 0.127 ]]);

#### real data
peaks12_real = np.zeros_like(peaks12);
peaks1_real = np.zeros_like(peaks1);
peaks32_real = np.array([
    [-0.020, 0.017, 0.104 ]]);
peaks92_real = np.array([
    [ 0.004963, 0.032, 0.163 ]]); # Mn4
#   [ 0.003762, 0.035, 0.168 ]]); # F SMM, single (no exchange)
peaks6_real = np.array([
    [ 0.003, 0.032, 0.153 ]]);

if real: peaks12, peaks1, peaks32, peaks92, peaks6 = peaks12_real, peaks1_real, peaks32_real, peaks92_real, peaks6_real;

#### plot T+ and p2 vs Delta E
if True:
    num_plots = 2;
    fig, axes = plt.subplots(num_plots, sharex=True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);
    indE, indT, indp = 0,1,2;

    # plot T+
    # for s=1/2
    axes[0].scatter(peaks12[:,indE], peaks12[:,indT], color=mycolors[0], marker = mymarkers[0], linewidth = mylinewidth);
    # for s=1
    axes[0].scatter(peaks1[:,indE], peaks1[:,indT], color=mycolors[1], marker=mymarkers[1], linewidth = mylinewidth);
    # for s=3/2
    axes[0].scatter(peaks32[:,indE], peaks32[:,indT], color=mycolors[2], marker=mymarkers[2], linewidth = mylinewidth);
    # for s=9/2
    axes[0].scatter(peaks92[:,indE], peaks92[:,indT], color=mycolors[3], marker=mymarkers[3], linewidth = mylinewidth);
    # for s=6
    axes[0].scatter(peaks6[:,indE], peaks6[:,indT], color=mycolors[4], marker=mymarkers[4], linewidth = mylinewidth);
       
    # plot analytical FOM
    # for s=1/2
    axes[1].scatter(peaks12[:,indE], peaks12[:,indp], color=mycolors[0], marker = mymarkers[0], linewidth = mylinewidth);
    # for s=1
    axes[1].scatter(peaks1[:,indE], peaks1[:,indp], color=mycolors[1], marker=mymarkers[1], linewidth = mylinewidth);
    # for s=3/2
    axes[1].scatter(peaks32[:,indE], peaks32[:,indp], color=mycolors[2], marker=mymarkers[2], linewidth = mylinewidth);
    # for s=9/2
    axes[1].scatter(peaks92[:,indE], peaks92[:,indp], color=mycolors[3], marker=mymarkers[3], linewidth = mylinewidth);
    # for s=6
    axes[1].scatter(peaks6[:,indE], peaks6[:,indp], color=mycolors[4], marker=mymarkers[4], linewidth = mylinewidth);

    # try arrow
    lowest_x, lowest_y = peaks32[np.argmin(peaks32[:,indp]),indE], peaks32[np.argmin(peaks32[:,indp]),indp];
    highest_x, highest_y = peaks32[np.argmax(peaks32[:,indp]),indE], peaks32[np.argmax(peaks32[:,indp]),indp];
    #plt.arrow(lowest_x, highest_y, lowest_x - lowest_x, lowest_y - highest_y, color = "darkslategray", length_includes_head = True);

    # format
    #axes[0].set_ylim(0.0,0.24);
    axes[0].set_ylabel('max($T_+$)', fontsize = myfontsize);
    #axes[1].set_ylim(0.08,0.32);
    axes[1].set_ylabel('max($\overline{p^2}$)', fontsize = myfontsize);

    # show
    axes[-1].set_xlim(-0.025,0.025);
    axes[-1].set_xlabel('$\Delta E/t$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    fname = 'figs/peaks.pdf';
    if real: fname = 'figs/peaks_real.pdf'
    #plt.savefig(fname);
    plt.show();



