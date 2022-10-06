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
plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

#### data

peaks12 = np.array([ [-0.05 , 0.089667 , 0.223979 ],
                     [-0.01 , 0.152587 , 0.272373 ],
                     [ 0.00 , 0.222490 , 0.298361 ],
                     [ 0.01 , 0.233674 , 0.314810 ],
                     [ 0.05 , 0.106960 , 0.285130 ]]);
peaks12 = np.array([ [ 0.00 , 0.222490 , 0.298361 ]]);
peaks1 = np.array([
     [-0.12 , 0.081899 , 0.213875 ],
     [-0.08 , 0.094447 , 0.225977 ],
     [-0.05 , 0.109244 , 0.238067 ],
     [-0.01 , 0.148408 , 0.261165 ],
     [ 0.00 , 0.162989 , 0.268366 ],
     [ 0.01 , 0.168571 , 0.268366 ],
     [ 0.05 , 0.148591 , 0.289001 ],
     [ 0.08 , 0.132247 , 0.289522 ],
     [ 0.12 , 0.122127 , 0.286590 ] ]);
peaks32 = np.array([
     [-0.12 , 0.061207 , 0.188055 ],
     [-0.08 , 0.071040 , 0.199658 ],
     [-0.05 , 0.082684 , 0.211454 ],
     [-0.01 , 0.113272 , 0.234906 ],
     [ 0.00 , 0.123194 , 0.242528 ],
     [ 0.01 , 0.123384 , 0.249863 ],
     [ 0.05 , 0.100259 , 0.257826 ],
     [ 0.08 , 0.087848 , 0.252942 ],
     [ 0.12 , 0.077284 , 0.244709 ] ]);
peaks6 = np.array([
     [-0.12 , 0.042421, 0.162669 ],
     [-0.08 , 0.043014, 0.164255 ],
     [-0.05 , 0.043184, 0.165394 ],
     [-0.01 , 0.042881, 0.166785 ],
     [ 0.003, 0.042640, 0.167187 ],
     [ 0.01 , 0.042486, 0.167398 ],
     [ 0.05 , 0.041329, 0.168386],
     [ 0.08 , 0.040290, 0.168872 ],
     [ 0.12 , 0.038848, 0.169136 ] ]);
peaks6_real = np.array([
     [ 0.003, 0.042640, 0.167187 ],
     [-0.060, 0.043160, 0.165020 ],
     [-0.030, 0.043117, 0.166109 ],
     [-0.003, 0.042759, 0.167005 ],
     [ 0.003, 0.042640, 0.167187 ],
     [ 0.030, 0.041956, 0.167935 ],
     [ 0.060, 0.040989, 0.168579 ] ]); 

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
    #for s=3/2
    axes[0].scatter(peaks32[:,indE], peaks32[:,indT], color=mycolors[1], marker=mymarkers[1], linewidth = mylinewidth);
    #for s=6
    axes[0].scatter(peaks6[:,indE], peaks6[:,indT], color=mycolors[2], marker=mymarkers[2], linewidth = mylinewidth);
    #for s=6 real
    axes[0].scatter(peaks6_real[:1,indE], peaks6_real[:1,indT], color='red', marker='x', linewidth = mylinewidth);
 
       
    # plot analytical FOM
    # for s=1/2
    axes[1].scatter(peaks12[:,indE], peaks12[:,indp], color=mycolors[0], marker = mymarkers[0], linewidth = mylinewidth);
    # for s=1
    axes[1].scatter(peaks1[:,indE], peaks1[:,indp], color=mycolors[1], marker=mymarkers[1], linewidth = mylinewidth);
    # for s=3/2
    axes[1].scatter(peaks32[:,indE], peaks32[:,indp], color=mycolors[2], marker=mymarkers[2], linewidth = mylinewidth);
    # for s=6
    axes[1].scatter(peaks6[:,indE], peaks6[:,indp], color=mycolors[3], marker=mymarkers[3], linewidth = mylinewidth);
    # for s=6
    axes[1].scatter(peaks6_real[:1,indE], peaks6_real[:1,indp], color='red', marker='x', linewidth = mylinewidth);

    # try arrow
    lowest_x, lowest_y = peaks32[np.argmin(peaks32[:,indp]),indE], peaks32[np.argmin(peaks32[:,indp]),indp];
    highest_x, highest_y = peaks32[np.argmax(peaks32[:,indp]),indE], peaks32[np.argmax(peaks32[:,indp]),indp];
    #plt.arrow(lowest_x, highest_y, lowest_x - lowest_x, lowest_y - highest_y, color = "darkslategray", length_includes_head = True);

    # format
    axes[0].set_ylim(0.0,0.24);
    axes[0].set_ylabel('max($T_+$)', fontsize = myfontsize);
    axes[1].set_ylim(0.15,0.31);
    axes[1].set_ylabel('max($\overline{p^2}$)', fontsize = myfontsize);

    # show
    axes[-1].set_xlabel('$\Delta E/t$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    plt.savefig('figs/peaks.pdf');
    plt.show();



