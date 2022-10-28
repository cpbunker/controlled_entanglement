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
mycolors = ["black","darkblue","darkgreen","darkred", "darkcyan", "darkmagenta","darkgray"];
mymarkers = ["o","^","s","d","*","X","P"];
def mymarkevery(fname,yvalues):
    if '-' in fname or '0.0.npy' in fname:
        return (40,40);
    else:
        return [np.argmax(yvalues)];
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)"];
plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

#### data
real = False;

peaks12 = np.array([ [ 0.00 , 0.222 , 0.298 ]]);
peaks1 = np.load("data/model1/peaks.npy");
peaks32 = np.load("data/model1.5/peaks.npy");
peaks72 = np.load("data/model3.5/peaks.npy"); # not plotted as of now
peaks4 = np.load("data/model4/peaks.npy");
peaks92 = np.load("data/model4.5/peaks.npy");
peaks6 = np.load("data/model6/peaks.npy");

#### real data
peaks12_real = np.array([ [ 0.00 , 0.222 , 0.298 ]]);
peaks1_real = np.zeros_like(peaks12);
peaks32_real = np.array([
    [ 0.020, 0.002, 0.047 ]]); # MnPc
peaks72_real = np.array([
    [ 0.006327, 0.017, 0.126 ]]); # Mn4_72 # not plotted as of now
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
    fig, axes = plt.subplots(nrows = num_plots, ncols = num_plots, sharex="col", sharey = "row", gridspec_kw={'width_ratios' : [9,1]});
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);
    indE, indT, indp = 0,1,2;

    # convert real data to meV
    convert = 100;

    # plot data on both x axes
    dataindex = [indT, indp];
    for yaxi in range(np.shape(axes)[0]):
        for xaxi in range(np.shape(axes)[1]):
            # for s=1/2
            axes[yaxi, xaxi].plot(convert*peaks12[:,indE], peaks12[:,dataindex[yaxi]], color=mycolors[0], marker = mymarkers[0], linewidth = mylinewidth);
            # for s=1
            if not real: axes[yaxi, xaxi].plot(convert*peaks1[:,indE], peaks1[:,dataindex[yaxi]], color=mycolors[1], marker=mymarkers[1], linewidth = mylinewidth);
            # for s=3/2
            axes[yaxi, xaxi].plot(convert*peaks32[:,indE], peaks32[:,dataindex[yaxi]], color=mycolors[2], marker=mymarkers[2], linewidth = mylinewidth);
            # for s=7/2
            #axes[yaxi, xaxi].plot(convert*peaks72[:,indE], peaks72[:,dataindex[yaxi]], color=mycolors[3], marker=mymarkers[3], linewidth = mylinewidth);
            # for s=4
            axes[yaxi, xaxi].plot(convert*peaks4[:,indE], peaks4[:,dataindex[yaxi]], color=mycolors[3], marker=mymarkers[3], linewidth = mylinewidth);
            # for s=9/2
            axes[yaxi, xaxi].plot(convert*peaks92[:,indE], peaks92[:,dataindex[yaxi]], color=mycolors[4], marker=mymarkers[4], linewidth = mylinewidth);
            # for s=6
            axes[yaxi, xaxi].plot(convert*peaks6[:,indE], peaks6[:,dataindex[yaxi]], color=mycolors[-1], marker=mymarkers[-1], linewidth = mylinewidth);
           
    # format
    lower_y = 0.08
    axes[0,0].set_ylim(-lower_y*0.24,0.24);
    axes[0,0].set_ylabel('max($T_+$)', fontsize = myfontsize);
    axes[1,0].set_ylim(-lower_y*0.32,0.32);
    axes[1,0].set_ylabel('max($\overline{p^2}$)', fontsize = myfontsize);

    # show
    if not real: xdatadelta = convert*abs(peaks1[0,indE]-peaks1[1,indE])/2;
    else: xdatadelta = convert*0.001/2;
    myxlabel = '$\Delta E$ (meV)';
    axes[-1,0].set_xlabel(myxlabel, fontsize = myfontsize);
    for yaxi in range(np.shape(axes)[0]): 
        if not real:
            axes[yaxi,0].set_title(mypanels[yaxi], x=0.07, y = 0.7, fontsize = myfontsize);
            axes[yaxi,0].set_xlim(convert*-0.004-xdatadelta,convert*0.004+xdatadelta);
            axes[yaxi,1].set_xlim(convert*0.02-xdatadelta, convert*0.02+xdatadelta);
            axes[yaxi,1].set_xticks([0.02*convert]);
        else:
            axes[yaxi,1].set_title(mypanels[yaxi], x=0.4, y = 0.7, fontsize = myfontsize);
            axes[yaxi,0].set_xlim(convert*0.0-xdatadelta,convert*0.008+xdatadelta);
            axes[yaxi,1].set_xlim(convert*0.02-xdatadelta, convert*0.02+xdatadelta);
            axes[yaxi,1].set_xticks([0.02*convert]);
        axes[yaxi,0].spines['right'].set_visible(False);
        axes[yaxi,1].spines['left'].set_visible(False);
        axes[yaxi,1].yaxis.set_visible(False);
        # break axes
        break_size = 0.12; # in display coordinates
        break_offset = (-6,-9)
        break_kw = dict(transform=axes[yaxi,1].transAxes, color='black', clip_on=False);
        axes[yaxi,1].plot((break_offset[0]*break_size-break_size,break_offset[0]*break_size+break_size),(-break_size,+break_size),linewidth = mylinewidth, **break_kw);
        axes[yaxi,1].plot((break_offset[1]*break_size-break_size,break_offset[1]*break_size+break_size),(-break_size,+break_size),linewidth = mylinewidth, **break_kw);
        axes[yaxi,1].plot((break_offset[0]*break_size-break_size,break_offset[0]*break_size+break_size),(1-break_size,1+break_size),linewidth = mylinewidth, **break_kw);
        axes[yaxi,1].plot((break_offset[1]*break_size-break_size,break_offset[1]*break_size+break_size),(1-break_size,1+break_size),linewidth = mylinewidth, **break_kw);
        # connect to breaks
        #axes[yaxi,1].plot((break_offset[0]*break_size-break_size,0),(0,0),linewidth = mylinewidth, **break_kw);
    plt.tight_layout();
    fname = 'figs/peaks.pdf';
    if real: fname = 'figs/peaks_real.pdf'
    #plt.savefig(fname);
    plt.show();



