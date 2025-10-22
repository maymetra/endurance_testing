
import numpy as np
import pandas as pd
from matplotlib.widgets import Button
import tkinter as tk
import sys
import csv
import os
import skrf as rf
from pathlib import Path
from collections import defaultdict
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from time import time_ns, sleep, localtime, strftime
import time 
import threading
import queue

import json
import hashlib
import logging
import pathlib
import datetime as dt
from typing import List, Dict, Any



k = keith

def where(*args):
    return np.where(*args)[0]

def list_files(directory, extension):
    return (f for f in listdir(directory) if f.endswith('.' + extension))

def setup_pcm_plots():
    def plot0(data, ax=None, **kwargs):
        ax.cla()
        ax.set_title('Answer')
        if data['t_scope']:
            ax.plot(data['t_scope'][-1], data['v_answer'][-1], **kwargs)    
        ax.set_ylabel('Voltage [V]')
        ax.set_xlabel('Time [s]')
        ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        
    def plot1(data, ax=None, **kwargs):
        ax.cla()
        ax.semilogy(data['t'], data['V'] / data['I'], **kwargs)
        #if data['t_event']:
        #    ax.vlines(data['t_event'],ax.get_ylim()[0]*1.2,ax.get_ylim()[1]*0.8, alpha = 0.5)
        ax.set_ylabel('Resistance [V/A]')
        ax.set_xlabel('Time [s]')
        ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        
    def plot2(data, ax=None, **kwargs):
        ax.cla()
        ax.set_title('Pulse')
        if data['t_scope']:
            ax.plot(data['t_scope'][-1], data['v_pulse'][-1], **kwargs)
        ax.set_ylabel('Voltage [V]')
        ax.set_xlabel('Time [s]')
        ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        
    def plot3(data, ax=None, **kwargs):
        ax.cla()
        ax.plot(data['t'], data['I'], **kwargs)
        # if data['t_event']:
        #    ax.vlines(data['t_event'],ax.get_ylim()[0]*1.2,ax.get_ylim()[1]*0.8, alpha = 0.5)
        ax.set_ylabel('Current [A]')
        ax.set_xlabel('Time [s]')
        ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        
    iplots.plotters = [[0, plot0],
                       [1, plot1],
                       [2, plot2],
                       [3, plot3]]
                 
    iplots.newline()

####### VCM 4 plotters ############
def setup_vcm_plots():

    def plot0(data, ax=None, **kwargs):
        line = data.iloc[-1]
        i=0
        if np.isnan(line.t_hrs).any():
            i+=1
            line = data.iloc[-2]
        ax.set_title('Pre Pulse Resistance State  #' + str(len(data)-i))
        if not np.isnan(line.t_hrs).any():
            ax.cla()
            ax.set_title('Pre Resistance State #' + str(len(data)-i))
            ax.plot(line.t_hrs,  line.V_hrs /  line.I_hrs - 50)
            ax.set_ylabel('Pre Resistance [V/A]')
            ax.set_xlabel('Time [s]')
            ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        
    def plot1(data, ax=None, **kwargs):
        line = data.iloc[-1]
        i=0
        if np.isnan(line.t_ttx).any():
            line = data.iloc[-2]
            i+=1
        ax.set_title('Pulse Answer #' + str(len(data)-i))
        if not np.isnan(line.t_ttx).any():
            ax.cla()
            ax.set_title('Pulse Answer #' + str(len(data)-i))
            ax.plot(line.t_ttx, line.V_ttx, **kwargs)    
            ax.set_ylabel('Voltage [V]')
            ax.set_xlabel('Time [s]')
            ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())

    def plot2(data, ax=None, **kwargs):
        line = data.iloc[-1]
        i=0
        if np.isnan(line.t_lrs).any():
            line = data.iloc[-2]
            i+=1
        ax.set_title('Post Pulse Resistance State #' + str(len(data)-i))
        if not np.isnan(line.t_lrs).any():
            ax.cla()
            ax.set_title('Post Resistance State #' + str(len(data)-i))
            ax.plot(line.t_lrs,  line.V_lrs /  line.I_lrs - 50)
            ax.set_ylabel('Post Resistance [V/A]')
            ax.set_xlabel('Time [s]')
            ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())

    def plot3(data, ax=None, **kwargs):
        line = data.iloc[-1]
        i=0
        if np.isnan(line.V_sweep).any():
            line = data.iloc[-2]
            i+=1
        ax.set_title('Sweep #' + str(len(data)-i))
        if not np.isnan(line.V_sweep).any():
            ax.cla()
            ax.set_title('Sweep #' + str(len(data)-i))
            ax.semilogy(line.V_sweep,  line.V_sweep /  line.I_sweep - 50)
            ax.set_ylabel('Resistance Sweep [V/A]')
            ax.set_xlabel('Voltage [V]')
            ax.set_ylim(bottom = 1e2)
            ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())   
        

        
    iplots.plotters = [[0, plot0],
                       [1, plot1],
                       [2, plot2]]

    #iplots.plotters = [[0, plot0],
    #                   [1, plot1],
    #                   [2, plot2],
    #                   [3, plot3]]

                 
    iplots.newline()


####### VCM 6 plotter for pick-off Tee ############
def setup_vcm_pick_off_plots():

    def plot0(data, ax=None, **kwargs):
        line = data.iloc[-1]
        i=0
        if np.isnan(line.t_hrs).any():
            i+=1
            line = data.iloc[-2]
        ax.set_title('Pre Pulse Resistance State  #' + str(len(data)-i))
        if not np.isnan(line.t_hrs).any():
            ax.cla()
            ax.set_title('Pre Resistance State #' + str(len(data)-i))
            ax.plot(line.t_hrs,  line.V_hrs /  line.I_hrs - 50)
            ax.set_ylabel('Pre Resistance [V/A]')
            ax.set_xlabel('Time [s]')
            ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        
    def plot1(data, ax=None, **kwargs):
        line = data.iloc[-1]
        i=0
        if np.isnan(line.t_ttx).any():
            line = data.iloc[-2]
            i+=1
        ax.set_title('Pulse Answer #' + str(len(data)-i))
        if not np.isnan(line.t_ttx).any():
            ax.cla()
            ax.set_title('Pulse Answer #' + str(len(data)-i))
            ax.plot(line.t_ttx, line.V_ttx, **kwargs)
            ax.plot(line.t_ttx_app, line.V_ttx_app, **kwargs)    
            ax.set_ylabel('Voltage [V]')
            ax.set_xlabel('Time [s]')
            ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())

    def plot2(data, ax=None, **kwargs):
        line = data.iloc[-1]
        i=0
        if np.isnan(line.t_lrs).any():
            line = data.iloc[-2]
            i+=1
        ax.set_title('Post Pulse Resistance State #' + str(len(data)-i))
        if not np.isnan(line.t_lrs).any():
            ax.cla()
            ax.set_title('Post Resistance State #' + str(len(data)-i))
            ax.plot(line.t_lrs,  line.V_lrs /  line.I_lrs - 50)
            ax.set_ylabel('Post Resistance [V/A]')
            ax.set_xlabel('Time [s]')
            ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())

    def plot3(data, ax=None, **kwargs):
        line = data.iloc[-1]
        i=0
        if np.isnan(line.V_sweep).any():
            line = data.iloc[-2]
            i+=1
        ax.set_title('Sweep #' + str(len(data)-i))
        if not np.isnan(line.V_sweep).any():
            ax.cla()
            ax.set_title('Sweep #' + str(len(data)-i))
            ax.semilogy(line.V_sweep,  line.V_sweep /  line.I_sweep - 50)
            ax.set_ylabel('Resistance Sweep [V/A]')
            ax.set_xlabel('Voltage [V]')
            ax.set_ylim(bottom = 1e2)
            ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())   
        

        
    iplots.plotters = [[0, plot0],
                       [1, plot1],
                       [2, plot2],
                       [3, plot3]]
                 
    iplots.newline()  

def setup_pcm_plots_2():

    def plot0(data, ax=None, **kwargs):
        line = data.iloc[-1]
        i=0
        if np.isnan(line.t_pre).any():
            i+=1
            line = data.iloc[-2]
        ax.set_title('Read PRE #' + str(len(data)-i))
        if not np.isnan(line.t_pre).any():
            ax.cla()
            ax.set_title('Read PRE #' + str(len(data)-i))
            ax.plot(line.t_pre,  line.V_pre /  line.I_pre)
            ax.set_ylabel('Resistance PRE [V/A]')
            ax.set_xlabel('Time [s]')
            ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
            ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter())
        
    def plot1(data, ax=None, **kwargs):
        line = data.iloc[-1]
        i=0
        if np.isnan(line.t_ttx).any():
            line = data.iloc[-2]
            i+=1
        ax.set_title('Answer #' + str(len(data)-i))
        if not np.isnan(line.t_ttx).any():
            ax.cla()
            ax.set_title('Answer #' + str(len(data)-i))
            ax.plot(line.t_ttx, line.V_ttx, **kwargs)    
            ax.set_ylabel('Voltage [V]')
            ax.set_xlabel('Time [s]')
            ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())

    def plot2(data, ax=None, **kwargs):
        line = data.iloc[-1]
        i=0
        if np.isnan(line.t_post).any():
            line = data.iloc[-2]
            i+=1
        ax.set_title('Read POST #' + str(len(data)-i))
        if not np.isnan(line.t_post).any():
            ax.cla()
            ax.set_title('Read POST #' + str(len(data)-i))
            ax.plot(line.t_post,  line.V_post /  line.I_post)
            ax.set_ylabel('Resistance Post [V/A]')
            ax.set_xlabel('Time [s]')
            ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
            ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter())

        
    iplots.plotters = [[0, plot0],
                       [1, plot1],
                       [2, plot2]]
                 
    iplots.newline()  
    
def set_keithley_plotters():
    iplots.plotters = keithley_plotters
    iplots.ax0.cla()
    iplots.ax1.cla()
    iplots.ax2.cla()
    iplots.ax3.cla()

def analog_measurement_series(
    # values for pandas file
    samplename,
    padname,
    attenuation = 0, 
    repetitions = 3,

    # values for sweeps in between analog measurements
    V_set = [0.9,1.,1.1],
    V_reset = [-1.0,-1.1,-1.2],
    number_sweeps = 10,

    # values for keithley
    V_read = 0.2,
    points = 7e3, # there is only 10 points in vcm_measurement. Why?
    interval = 1e-3, # is fixed to 0.1 in vcm_measurement
    range_read = 1e-3,
    limit_read = 1e-3,
    nplc = 1e-2,

    # values for tektronix
    trigger_level = 0.025,
    polarity = 1,
    recordlength = 5000,
    position = -2.5,
    scale = 0.04,
    transient_measurement = False,

    # values for sympuls
    pulse_widths = [],
    pulse_spacing = 100e-3
):
    
    data = {}
    data['padname'] = padname
    data['samplename'] = samplename
    timestamp = strftime("%Y.%m.%d-%H.%M.%S", localtime())
    data['timestamp'] = timestamp

    for i in range(repetitions):
        for V_set_cycle, V_reset_cycle in zip(V_set, V_reset):
            for pulse_width in pulse_widths:
                
                data[f'pulsewidth{pulse_width:.2e}s_{i+1}'.replace("+", "")] = analog_measurement(
                    # values for pandas file
                    samplename,
                    padname,
                    attenuation=attenuation,
                    # values for sweeps
                    V_set=V_set_cycle,
                    V_reset=V_reset_cycle,
                    number_sweeps=number_sweeps,
                    # values for keithley
                    V_read=V_read,
                    points=points,
                    interval=interval, # is fixed to 0.1 in vcm_measurement
                    range_read=range_read,
                    limit_read=limit_read,
                    nplc=nplc,
                    # values for tektronix
                    trigger_level=trigger_level,
                    polarity=polarity,
                    recordlength=recordlength,
                    position=position,
                    scale=scale,
                    transient_measurement=transient_measurement,
                    # values for sympuls
                    pulse_width = pulse_width,
                    pulse_spacing = pulse_spacing
                )

    """
    datafolder = os.path.join('C:\\Messdaten', padname, samplename, "series")
    # subfolder = datestr
    file_exits = True
    i=1
    # f"{timestamp}_pulsewidth={pulse_width:.2e}s_attenuation={attenuation}dB_points={points:.2e}_{i}"
    filepath = os.path.join(datafolder, f"{timestamp}_attenuation{attenuation}dB_series_{i}.s")
    while os.path.isfile(filepath + '.s'):
        i +=1
        filepath = os.path.join(datafolder, f"{timestamp}_attenuation{attenuation}dB_series_{i}.s")
    io.write_pandas_pickle(meta.attach(data), filepath)
    """

    return data

def analog_measurement(
    # values for pandas file
    samplename,
    padname,
    attenuation = 0, 

    # values for sweeps
    V_set = 1.,
    V_reset = -1.1,
    number_sweeps = 10,

    # values for keithley
    V_read = 0.2,
    points = 1e4, # there is only 10 points in vcm_measurement. Why?
    interval = 1e-3, # is fixed to 0.1 in vcm_measurement
    range_read = 1e-3,
    limit_read = 1e-3,
    nplc = 1e-2,

    # values for tektronix
    trigger_level = 0.025,
    polarity = 1,
    recordlength = 5000,
    position = -2.5,
    scale = 0.04,
    transient_measurement = False,

    # values for sympuls
    pulse_width = 10e-9,
    pulse_spacing = 100e-3
    # pg5_measurement = True,
    # continuous = False
):
    '''run a measurement during which the Keithley2600 applies a constants voltage and measures the current. 
    Pulses applied during this measurement are also recorded. '''
    number_of_events =0
    data = {}
    data['padname'] = padname
    data['samplename'] = samplename

    data['num_sweeps'] = number_sweeps
    data['V_set'] = V_set
    data['V_reset'] = V_reset 
    data['V_read'] = V_read
    data['points'] = points 
    data['interval'] = interval
    data['range_read'] = range_read 
    data['limit_read'] = limit_read
    data['nplc'] = nplc
    data['trigger_level'] = trigger_level
    data['polarity'] = polarity
    data['position'] = position
    data['scale'] = scale
    data['pulse_spacing'] = pulse_spacing

    data['t_scope'] = []
    data['v_pulse'] = []
    data['v_answer'] = []
    data['t_event'] = []
    
    data['attenuation'] = attenuation
    data['recordlength'] = recordlength
    data['pulse_width'] = pulse_width

    timestamp = strftime("%Y.%m.%d-%H.%M.%S", localtime())
    data['timestamp'] = timestamp

    # functions for sweeps
    def reset():
        k._iv_lua(
            tri(V_reset, 0.05), Irange=1e-2, Ilimit=1e-2,
            Plimit=V_reset*1e-3, nplc=nplc, Vrange=V_reset
        )
        while not k.done():
            sleep(0.01)
        return k.get_data()
    def set():
        k._iv_lua(
            tri(V_set, 0.05), Irange=1e-3, Ilimit=3e-4, 
            Plimit=V_set*1e-3, nplc=nplc, Vrange=V_set
        )
        while not k.done():
            sleep(0.01)
        return k.get_data()
    def read():
        return kiv(tri(v1 = V_read, step = 0.02), measure_range = 1e-3, i_limit = 1e-3)
    def get_current_resistance ():
        data = read()
        I = data["I"]
        V = data["Vmeasured"]
        return V[len(V)//2]/I[len(I)//2]

    # start doing a few sweeps to improve reproducibility
    set_keithley_plotters()
    iplots.show()

    # get initial resistance state and switch to HRS if necessary 
    data['initial_state'] = get_current_resistance()
    if data['initial_state'] <= 5000:
        data['initial_set'] = reset()

    # create list for sets and resets
    data['sets'] = []
    data['resets'] = []
    
    # now in HRS we do {number_sweeps}
    for i in range(number_sweeps-1):
        data['sets'].append(set())
        # data[f'set_{i+1}_state'] = get_current_resistance()
        data['resets'].append(reset())
        # data[f'reset_{i+1}_state'] = get_current_resistance()
    data['sets'].append(kiv(tri(v1=V_set,step=0.05),measure_range=1e-3,i_limit=3e-4))
    data['resets'].appen(kiv(tri(v1=V_reset,step=0.05),measure_range=1e-2,i_limit=1e-2))

    # get initial HRS after sweeps
    data['initial_HRS'] = get_current_resistance()

    # then do analog measurement
    setup_pcm_plots()
    iplots.show()  
    num_pulses = 0  

    # recordlength = (pulse_width * 100e9) + 500
    # read resistance state with keithley
    k.source_output(ch = 'A', state = True)
    k.source_level(source_val= V_read, source_func='v', ch='A')
    plt.pause(1)
    k._it_lua(sourceVA = V_read , sourceVB = 0, points = points, interval = interval, rangeI = range_read , limitI = limit_read, nplc = nplc)
    data['t_begin'] = time_ns()

    # set up tektronix
    ttx.inputstate(1, False)
    ttx.inputstate(2, False)
    ttx.inputstate(3, True)    
    ttx.inputstate(4, False)
    ttx.scale(3, scale)
    ttx.position(3, position*polarity)
    ttx.change_samplerate_and_recordlength(100e9, recordlength)
    trigger_level = trigger_level*polarity

    # set up sympuls
    sympuls.set_pulse_width(pulse_width)

    # have python wait for keithley to get ready
    plt.pause(0.5)
    
    # middle measurements, where keithey just reads and sympuls sends pulses
    while not k.done():
        if transient_measurement:
            ttx.arm(source = 3, level = trigger_level, edge = 'r') 
            plt.pause(0.1)
        sympuls.trigger()
        data['t_event'].append(time_ns())
        num_pulses += 1
        # sleep at least 10ms between pulses
        if transient_measurement:
            plt.pause(0.2)
            data.update(k.get_data())
            if ttx.triggerstate():
                plt.pause(0.1)
                ttx.disarm()
                # padname+="_no_last_pulse_detected_"
            else:
                number_of_events +=1
                data_scope2 = ttx.get_curve(3)
                # time_array = data['t']
                data['t_scope'].append(data_scope2['t_ttx'])
                data['v_answer'].append(data_scope2['V_ttx'])
        else:
            sleep(pulse_spacing)

    # read all the measured data from keithley
    data.update(k.get_data())
    k.source_output(ch = 'A', state = False)
    k.source_output(ch = 'B', state = False)
    data["num_pulses"] = num_pulses

    """
    # last measurement where tektronix reads pulse
    ttx.arm(source = 3, level = trigger_level, edge = 'r') 
    plt.pause(0.1)
    sympuls.trigger()
    plt.pause(0.2)
    if ttx.triggerstate():
        plt.pause(0.1)
        ttx.disarm()
    else:
        number_of_events +=1
        data_scope2 = ttx.get_curve(3)
        data['t_scope'].append(data_scope2['t_ttx'])
        data['v_answer'].append(data_scope2['V_ttx'])
    iplots.updateline(data)
    ttx.disarm()
    """

    # save results
    datafolder = os.path.join('C:\\Messdaten', padname, samplename)
    i=1
    # f"{timestamp}_pulsewidth={pulse_width:.2e}s_attenuation={attenuation}dB_points={points:.2e}_{i}"
    filepath = os.path.join(datafolder, f"{timestamp}_pulsewidth{pulse_width:.2e}s_attenuation{attenuation}dB_points{points:.2e}_{i}.s".replace("+", ""))
    while os.path.isfile(filepath + '.s'):
        i +=1
        filepath = os.path.join(datafolder, f"{timestamp}_pulsewidth{pulse_width:.2e}s_attenuation{attenuation}dB_points{points:.2e}_{i}.s".replace("+", ""))
    io.write_pandas_pickle(meta.attach(data), filepath)
    # print(len(data))
    print(f"{num_pulses=}")
    return data    


def test_measurement_single(
    # values for pandas file
    samplename,
    padname,
    attenuation =0, 

    # values for keithley
    V_read = -0.2,
    points = 250, # there is only 10  points in vcm_measurement. Why?
    interval = 0.1, # is fixed to 0.1 in vcm_measurement
    range_read = 1e-3,
    limit_read = 1e-3,
    nplc = 1,

    # values for tektronix
    trigger_level = 0.1,
    polarity = 1,
    recordlength = 250,
    position = -2.5,
    scale = 0.12,

    # values for sympuls
    pulse_width = 50e-12,
    pg5_measurement = True,
    continuous = False
):
    '''run a measurement during which the Keithley2600 applies a constants voltage and measures the current. 
    Pulses applied during this measurement are also recorded. '''
    setup_pcm_plots()

    number_of_events =0
    data = {}
    data['padname'] = padname
    data['samplename'] = samplename

    data['V_read'] = V_read
    data['points'] = points 
    data['interval'] = interval
    data['range_read'] = range_read 
    data['limit_read'] = limit_read
    data['nplc'] = nplc
    data['trigger_level'] = trigger_level
    data['polarity'] = polarity
    data['position'] = position
    data['scale'] = scale
    data['pg5_measurement'] = pg5_measurement
    data['continuous'] = continuous

    data['t_scope'] = []
    data['v_pulse'] = []
    data['v_answer'] = []
    data['t_event'] = []
    
    data['attenuation'] = attenuation
    data['recordlength'] = recordlength
    data['pulse_width'] = pulse_width

    num_pulses = 0

    iplots.show()    

    # recordlength = (pulse_width * 100e9) + 500
    # read resistance state with keithley
    k.source_output(ch = 'A', state = True)
    k.source_level(source_val= V_read, source_func='v', ch='A')
    plt.pause(1)
    k._it_lua(sourceVA = V_read , sourceVB = 0, points = points, interval = interval, rangeI = range_read , limitI = limit_read, nplc = nplc)

    # set up tektronix
    ttx.inputstate(1, False)
    ttx.inputstate(2, False)
    ttx.inputstate(3, True)    
    ttx.inputstate(4, False)
    ttx.scale(3, scale)
    ttx.position(3, position*polarity)
    ttx.change_samplerate_and_recordlength(100e9, recordlength)
    trigger_level = trigger_level*polarity

    # set up sympuls
    sympuls.set_pulse_width(pulse_width)
    
    # if pg5_measurement:
    #     sympuls.set_pulse_width(pulse_width)
    #     plt.pause(1)
    #     sympuls.trigger()
    
    while not k.done():
        ttx.arm(source = 3, level = trigger_level, edge = 'r') 
        plt.pause(0.1)

        if pg5_measurement and continuous:
            sympuls.trigger()
            num_pulses += 1
            print('trigger'+str(trigger_level))
            plt.pause(0.2)
        data.update(k.get_data())
        if ttx.triggerstate():
            plt.pause(0.1)
        else:
            number_of_events +=1
            data_scope2 = ttx.get_curve(3)
            time_array = data['t']
            data['t_scope'].append(data_scope2['t_ttx'])
            data['v_answer'].append(data_scope2['V_ttx'])
            '''Moritz: last current data point measured after last trigger event so the entry one before
             will be used as time reference (-2 instead of -1, which is the last entry)'''
            data['t_event'].append(time_array[len(time_array)-2])
            print(time_array[len(time_array)-2])
        iplots.updateline(data)
    data.update(k.get_data())
    iplots.updateline(data)
#    k.set_channel_state('A', False)
#    k.set_channel_state('B', False)
    k.source_output(ch = 'A', state = False)
    k.source_output(ch = 'B', state = False)
    ttx.disarm()
    datafolder = os.path.join('C:\Messdaten', samplename, padname)
    subfolder = datestr
    file_exits = True
    i=1
    filepath = os.path.join(datafolder, subfolder, 'test_measurement_'+str(int(pulse_width*1e12)) + 'ps_' +str(int(attenuation)) + 'dB_'+str(int(points/10)) +'secs_' +str(i))
    while os.path.isfile(filepath + '.s'):
        i +=1
        filepath = os.path.join(datafolder, subfolder, 'test_measurement_'+str(int(pulse_width*1e12)) + 'ps_' +str(int(attenuation)) + 'dB_'+str(int(points/10)) +'secs_' +str(i))
    io.write_pandas_pickle(meta.attach(data), filepath)
    # print(len(data))
    print(f"{num_pulses=}")
    return data    

def test_measurement(samplename,
padname,
amplitude = np.nan,
bits = np.nan,
sourceVA = -0.2,
points = 250,
interval = 0.1,
trigger = 0.1,
two_channel = False,
rangeI = 0,
nplc = 1,
pulse_width = 50e-12,
attenuation =0,
pg5_measurement = True,
polarity = 1,
recordlength = 250,
answer_position = -2.5,
pulse_postition = -4,
answer_scale = 0.12,
pulse_scale = 0.12,
continuous = False):
    '''run a measurement during which the Keithley2600 applies a constants voltage and measures the current. 
    Pulses applied during this measurement are also recorded. '''
    setup_pcm_plots()

    number_of_events =0
    data = {}
    data['t_scope'] = []
    data['v_pulse'] = []
    data['v_answer'] = []
    data['t_event'] = []
    data['amplitude'] = amplitude
    data['bits'] = bits
    data['padname'] = padname
    data['samplename'] = samplename
    data['attenuation'] = attenuation
    data['recordlength'] = recordlength

    iplots.show()    
    k.source_output(ch = 'A', state = True)
    k.source_level(source_val= sourceVA, source_func='v', ch='A')
 #   k.set_channel_state(channel = 'A', state = True)
 #   k.set_channel_voltage(channel = 'A', voltage = sourceVA)

    plt.pause(1)
    k._it_lua(sourceVA = sourceVA , sourceVB = 0, points = points, interval = interval, rangeI = rangeI , limitI = 1, nplc = nplc)

#    k.it(sourceVA = sourceVA, sourceVB = 0, points = points, interval = interval, rangeI = rangeI, limitI = 1, nplc = nplc, reset_keithley = False)

    ttx.inputstate(1, False)
    ttx.inputstate(2, False)
    ttx.inputstate(3, True)
    if two_channel:
        ttx.inputstate(4, True)
        ttx.scale(4, pulse_scale)
        ttx.position(4, -pulse_position*polarity)
    else:
        ttx.inputstate(3, True)
    ttx.scale(3, answer_scale)
    ttx.position(3, answer_position*polarity)
    ttx.change_samplerate_and_recordlength(100e9, recordlength)




    trigger = trigger*polarity
    if two_channel:
        ttx.arm(source = 4, level = trigger, edge = 'e')
    else:
        ttx.arm(source = 3, level = trigger, edge = 'r')

    if pg5_measurement:
#    if pg5:
#        pg5.set_pulse_width(pulse_width)
        sympuls.set_pulse_width(pulse_width)
        plt.pause(1)
#        pg5.trigger()
        sympuls.trigger()
    
    while not k.done():
        
#        if pg5 and continuous:
        if pg5_measurement and continuous:
#            pg5.trigger()

            sympuls.trigger()

        data.update(k.get_data())

        if ttx.triggerstate():
            plt.pause(0.1)
        else:
            number_of_events +=1
            if two_channel:
                data_scope1 = ttx.get_curve(4)

            data_scope2 = ttx.get_curve(3)
            
            time_array = data['t']
            data['t_scope'].append(data_scope2['t_ttx'])
            if two_channel:
                data['v_pulse'].append(data_scope1['V_ttx'])
            data['v_answer'].append(data_scope2['V_ttx'])
            '''Moritz: last current data point measured after last trigger event so the entry one before
             will be used as time reference (-2 instead of -1, which is the last entry)'''
            data['t_event'].append(time_array[len(time_array)-2])
            print(time_array[len(time_array)-2])
            if two_channel:
                ttx.arm(source = 4, level = trigger, edge = 'e')
            else:
                ttx.arm(source = 3, level = trigger, edge = 'e')

        iplots.updateline(data)

    data.update(k.get_data())
    iplots.updateline(data)
#    k.set_channel_state('A', False)
#    k.set_channel_state('B', False)
    k.source_output(ch = 'A', state = False)
    k.source_output(ch = 'B', state = False)
    ttx.disarm()
    
    datafolder = os.path.join('C:\Messdaten', samplename, padname)
    subfolder = datestr
    file_exits = True
    i=1
    filepath = os.path.join(datafolder, subfolder, 'pcm_measurement_'+str(i))
    while os.path.isfile(filepath + '.s'):
        i +=1
        filepath = os.path.join(datafolder, subfolder, 'pcm_measurement_'+str(i))
    io.write_pandas_pickle(meta.attach(data), filepath)

    return data


def abrupt_measurement(samplename,
padname,
amplitude = np.nan,
bits = np.nan,
sourceVA = -0.2,
points = 250,
Number_of_pulses = 4,
period = 0,
interval = 0.1,
trigger = 0.1,
two_channel = False,
rangeI = 0,
nplc = 1,
pulse_width = 50e-12,
attenuation =0,
pg5_measurement = True,
polarity = 1,
recordlength = 250,
answer_position = -2.5,
pulse_postition = -4,
answer_scale = 0.12,
pulse_scale = 0.12,
continuous = False):
    '''run a measurement during which the Keithley2600 applies a constants voltage and measures the current. 
    Pulses applied during this measurement are also recorded. '''
    setup_pcm_plots()

    number_of_events =0
    data = {}
    data['t_scope'] = []
    data['v_pulse'] = []
    data['v_answer'] = []
    data['t_event'] = []
    data['amplitude'] = amplitude
    data['bits'] = bits
    data['padname'] = padname
    data['samplename'] = samplename
    data['attenuation'] = attenuation
    data['recordlength'] = recordlength
    data['Number_of_pulses'] = Number_of_pulses

    iplots.show()    
    k.source_output(ch = 'A', state = True)
    k.source_level(source_val= sourceVA, source_func='v', ch='A')
 #   k.set_channel_state(channel = 'A', state = True)
 #   k.set_channel_voltage(channel = 'A', voltage = sourceVA)

    plt.pause(1)
    k._it_lua(sourceVA = sourceVA , sourceVB = 0, points = points, interval = interval, rangeI = rangeI , limitI = 1, nplc = nplc)

#    k.it(sourceVA = sourceVA, sourceVB = 0, points = points, interval = interval, rangeI = rangeI, limitI = 1, nplc = nplc, reset_keithley = False)

    ttx.inputstate(1, False)
    ttx.inputstate(2, False)
    ttx.inputstate(3, True)
    if two_channel:
        ttx.inputstate(4, True)
        ttx.scale(4, pulse_scale)
        ttx.position(4, -pulse_position*polarity)
    else:
        ttx.inputstate(3, True)
    ttx.scale(3, answer_scale)
    ttx.position(3, answer_position*polarity)
    ttx.change_samplerate_and_recordlength(100e9, recordlength)




    trigger = trigger*polarity
    if two_channel:
        ttx.arm(source = 4, level = trigger, edge = 'e')
    else:
        ttx.arm(source = 3, level = trigger, edge = 'r')

    if pg5_measurement:
#    if pg5:
#        pg5.set_pulse_width(pulse_width)
##        sympuls.set_pulse_width(pulse_width)
##        plt.pause(1)
#        pg5.trigger()
 #       sympuls.set_pulse_width(pulse_width)
 #      sympuls.set_period(period)
 #       time_executed = (Number_of_pulses ) *period
 #       print('Excecutaion TIme',time_executed)
#        sympuls.write(':TRIG:SOUR IMM')
#        t = 0
#        sympuls.write(':TRIG:SOUR MANUAL')
        sympuls.Apply_Burst(pulse_width, period, Number_of_pulses)

##        sympuls.trigger()

    while not k.done() and t :
        
#        if pg5 and continuous:
        if pg5_measurement and continuous:
#            pg5.trigger()
            sympuls.Apply_Burst_time(pulse_width, period, Number_of_pulses)
            ttx.arm(source = 3, level = trigger, edge = 'r')

#            sympuls.trigger()


        data.update(k.get_data())

        if ttx.triggerstate():
            plt.pause(0.1)
        else:
            number_of_events +=1
            if two_channel:
                data_scope1 = ttx.get_curve(4)

            data_scope2 = ttx.get_curve(3)
            
            time_array = data['t']
            data['t_scope'].append(data_scope2['t_ttx'])
            if two_channel:
                data['v_pulse'].append(data_scope1['V_ttx'])
            data['v_answer'].append(data_scope2['V_ttx'])
            '''Moritz: last current data point measured after last trigger event so the entry one before
             will be used as time reference (-2 instead of -1, which is the last entry)'''
            data['t_event'].append(time_array[len(time_array)-2])
            print(time_array[len(time_array)-2])
            if two_channel:
                ttx.arm(source = 4, level = trigger, edge = 'e')
            else:
                ttx.arm(source = 3, level = trigger, edge = 'e')

        iplots.updateline(data)

    data.update(k.get_data())
    iplots.updateline(data)
#    k.set_channel_state('A', False)
#    k.set_channel_state('B', False)
    k.source_output(ch = 'A', state = False)
    k.source_output(ch = 'B', state = False)
    ttx.disarm()
    
    datafolder = os.path.join('C:\Messdaten', samplename, padname)
    subfolder = datestr
    file_exits = True
    i=1
    filepath = os.path.join(datafolder, subfolder, 'pcm_measurement_'+str(i))
    while os.path.isfile(filepath + '.s'):
        i +=1
        filepath = os.path.join(datafolder, subfolder, 'pcm_measurement_'+str(i))
    io.write_pandas_pickle(meta.attach(data), filepath)

    return data

def ferro_measurement(samplename,
padname,
polarity,
attenuation,
scale1 = 0.12,
scale4 = 1.2,
position1 = -1,
position4 = -4,
trigger_level = 0.7):
    data = {}
    data['padname'] = padname
    data['samplename'] = samplename
    data['attenuation'] = attenuation
    data['scale1'] = scale1
    data['scale4'] = scale4
    data['position1'] = position1
    data['position4'] = position4
    data['trigger_level'] = trigger_level

    
    ttx.inputstate(1, True)
    ttx.inputstate(2, False)
    ttx.inputstate(3, False)
    ttx.inputstate(4, True)


    ttx.scale(1, scale1)
    ttx.scale(4, scale4)
    ttx.position(1, polarity*position1)
    ttx.position(4, polarity*position4)


    ttx.change_samplerate_and_recordlength(samplerate = 100e9, recordlength = 5e3)
    ttx.trigger_position(20)
    if polarity == 1:
        ttx.arm(source = 4, level = trigger_level*polarity, edge = 'r')
    elif polarity == -1:
        ttx.arm(source = 4, level = trigger_level*polarity, edge = 'f')
    else:
        print('wrong polarity')
        return np.nan
    plt.pause(0.2)
    status = ttx.triggerstate()
    while status == True:
        plt.pause(0.1)
        status = ttx.triggerstate()
    plt.pause(0.5)
    data_1 = ttx.get_curve(1)
    data_4 = ttx.get_curve(4)

    ax0.plot(data_1['t_ttx'], data_1['V_ttx'])
    ax1.plot(data_4['t_ttx'], data_4['V_ttx'])

    data['t_ttx'] = data_1['t_ttx']
    data['v_scope'] = data_4['V_ttx']
    data['v_answer'] = data_1['V_ttx']

    datafolder = os.path.join('X:/emrl/Pool/Bulletin/Berg/Messungen/', samplename, padname)
    file_exits = True
    i=1
    filepath = os.path.join(datafolder, 'pulse_'+str(i))
    file_link = Path(filepath + '.s')
    while file_link.is_file():
        i +=1
        filepath = os.path.join(datafolder, 'pulse_'+str(i))
        file_link = Path(filepath + '.s')
    io.write_pandas_pickle(meta.attach(data), filepath)

    return data
    

def vcm_measurement(samplename,
padname,
v1,
v2,
step = 0.01,
step2 = 0.01,
V_read = 0.2,
range_lrs = 1e-3,
range_hrs = 1e-4,
range_sweep = 1e-2,
range_sweep2 = 1e-3,
cycles = 1,
pulse_width = 50e-12,
attenuation = 0,
automatic_measurement = True,
pg5_measurement = True,
recordlength = 250,
trigger_position = 25,
edge = 'r',
sweep = True,
two_sweeps = False,
scale = 0.12,
position = -3,
trigger_level = 0.05,
nplc = 10,
limitI = 3e-4,
limitI2 = 3e-4,
r_window = False,
r_lower = 1e3,
r_upper = 2e3,
cc_step = 25e-6):

    setup_vcm_plots()
    data = {}
    data['padname'] = padname
    data['samplename'] = samplename


    hrs_list = []
    lrs_list = []
    sweep_list = []
    scope_list = []
    vlist = tri(v1 = v1, v2 = v2, step = step)

    abort = False
    for i in range(cycles):
        if not abort:
            ### Reading HRS resistance ############################################################################
            k.source_output(ch = 'A', state = True)
            k.source_level(source_val= V_read, source_func='v', ch='A')

            plt.pause(1)

            k._it_lua(sourceVA = V_read, sourceVB = 0, points =10, interval = 0.1, rangeI = range_hrs , limitI = 1, nplc = nplc)
            while not k.done():
                plt.pause(0.1)
            k.source_output(ch = 'A', state = False)
            k.source_output(ch = 'B', state = False)
            hrs_data = k.get_data()
            hrs_list.append(add_suffix_to_dict(hrs_data,'_hrs'))
            data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
            iplots.updateline(data)
            ### Setting up scope  ################################################################################

            ttx.inputstate(3, True)
            ttx.inputstate(2, False)
            ttx.inputstate(1, False)
            ttx.inputstate(4, False)

            ttx.scale(3, scale)
            ttx.position(3, position)


            ttx.change_samplerate_and_recordlength(samplerate = 100e9, recordlength= recordlength)
            if pulse_width < 100e-12:
                ttx.trigger_position(40)
            elif pulse_width < 150e-12:
                ttx.trigger_position(30)
            else:
                ttx.trigger_position(trigger_position)

            plt.pause(0.1)

            ttx.arm(source = 3, level = trigger_level, edge = edge)


            ### Applying pulse and reading scope data #############################################################
            if pg5_measurement:
                sympuls.set_pulse_width(pulse_width)
            if not automatic_measurement:
                input('Connect the RF probes and press enter')
                plt.pause(0.5)
            else:
                plt.pause(1)
                
            if pg5_measurement:
                sympuls.trigger()
            else:
                print('Apply pulse')
            plt.pause(0.1)
            plt.pause(0.2)
            status = ttx.triggerstate()
            while status == True:
                plt.pause(0.1)
                status = ttx.triggerstate()
            plt.pause(0.5)
            scope_list.append(ttx.get_curve(3))
            data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
            iplots.updateline(data)

            ### Reading LRS resistance #############################################################################

            if not automatic_measurement:
                input('Connect the DC probes and press enter')

            k.source_output(ch = 'A', state = True)
            k.source_level(source_val= V_read, source_func='v', ch='A')

            plt.pause(1)
            k._it_lua(sourceVA = V_read, sourceVB = 0, points = 10, interval = 0.1, rangeI = range_lrs, limitI = 1, nplc = nplc)
            while not k.done():
                plt.pause(0.1)
            k.source_output(ch = 'A', state = False)
            k.source_output(ch = 'B', state = False)
            lrs_data = k.get_data()
            lrs_list.append(add_suffix_to_dict(lrs_data,'_lrs'))
            data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
            iplots.updateline(data)

            ### performing sweep ###################################################################################
            if sweep:
                if two_sweeps:
                    dates_dict = defaultdict(list)
                    vlist1 = tri(v1 = v1, v2 = 0, step = step)
                    vlist2 = tri(v1 = 0, v2 = v2, step = step2)
                    k.iv(vlist1, measure_range = range_sweep, i_limit = limitI) 
                    while not k.done():
                        plt.pause(0.1)
                    sweep_data = k.get_data()
                    k.iv(vlist2, measure_range = range_sweep2, i_limit = limitI2) 
                    while not k.done():
                        plt.pause(0.1)
                    data_2nd_sweep = k.get_data()
                    for key in data_2nd_sweep:
                        data_to_append = data_2nd_sweep[key]
                        if not isinstance(data_to_append,dict) and not isinstance(data_to_append, str):
                            sweep_data[key] = np.append(sweep_data[key], data_to_append)
                else:  
                    k.iv(vlist, measure_range = range_sweep) 
                    while not k.done():
                        plt.pause(0.1)
                    k.source_output(ch = 'A', state = False)
                    k.source_output(ch = 'B', state = False)
                    sweep_data = k.get_data()
                sweep_list.append(add_suffix_to_dict(sweep_data,'_sweep'))
                data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
                iplots.updateline(data)
            if r_window:
                current_compliance = limitI2
                window_hit = False
                u=0
                d=0
                while not window_hit:
                    
                    k.source_output(chan = 'A', state = True)
                    k.source_level(source_val= V_read, source_func='v', ch='A')

                    plt.pause(1)
                    k._it_lua(sourceVA = V_read, sourceVB = 0, points = 5, interval = 0.1, measure_range = range_lrs, limitI = 1, nplc = nplc)
                    while not k.done():
                        plt.pause(0.1)
                    k.source_output(ch = 'A', state = False)
                    k.source_output(ch = 'B', state = False)
                    r_data = k.get_data()
                    resistance = np.mean(r_data['V']/r_data['I']) - 50
                    print('Compliance = ' + str(current_compliance))
                    print('Resistance = ' + str(resistance))

                    if resistance >= r_lower and resistance <= r_upper:
                        window_hit = True
                        break
                    elif resistance < r_lower:
                        current_compliance -= cc_step
                        u = 0
                        d += 1
                    elif resistance > 3.5e4 or u >=50:
                        vlist2 = tri(v1 = 0, v2 = -2, step = step2)
                        current_compliance = 2e-3
                    elif d >= 50:
                        vlist1 = tri(v1 = 2, v2 = 0, step = step)
                    else:
                        current_compliance += cc_step
                        u += 1
                        d = 0

                    if current_compliance < cc_step:
                        current_compliance =cc_step

                    if u > 51 or d > 51:
                        print('Failed hitting resistance window, aborting measurement')
                        window_hit = True
                        abort = True
                        break

                    k.iv(vlist1, measure_range = range_sweep, Ilimit = limitI) 
                    while not k.done():
                        plt.pause(0.1)
                    
                    k.iv(vlist2, measure_range = range_sweep2, Ilimit = current_compliance) 
                    while not k.done():
                        plt.pause(0.1)
                    vlist1 = tri(v1 = v1, v2 = 0, step = step)
                    vlist2 = tri(v1 = 0, v2 = v2, step = step2)

                    if current_compliance > 1e-3:
                        current_compliance = limitI2
            k.source_output(ch = 'A', state = False)
            k.source_output(ch = 'B', state = False)
  
    data['attenuation'] = attenuation
    data['pulse_width'] = pulse_width
    data['scale'] = scale
    data['position'] = position
    data['trigger_level'] = trigger_level

    datafolder = os.path.join('C:\Messdaten', samplename, padname)
    subfolder = datestr
    file_exits = True
    i=1
    filepath = os.path.join(datafolder, subfolder, str(int(pulse_width*1e12)) + 'ps_'+str(int(attenuation)) + 'dB_'+str(i))
    file_link = Path(filepath + '.df')
    while file_link.is_file():
        i +=1
        filepath = os.path.join(datafolder, subfolder, str(int(pulse_width*1e12)) + 'ps_'+str(int(attenuation)) + 'dB_'+str(i))
        file_link = Path(filepath + '.df')
    io.write_pandas_pickle(meta.attach(data), filepath)

    return data, abort


def pcm_resistance_measurement(samplename,
padname,
bits,
amplitude,
V_read = 0.2,
start_range = 1e-3,
cycles = 1,
scale = 0.12,
position = 3,
trigger_level = -0.1,
recordlength=2000,
points = 10
):

    setup_pcm_plots_2()
    data = {}
    data['padname'] = padname
    data['samplename'] = samplename

    pre_list = []
    post_list = []
    scope_list = []


    abort = False
    for i in range(cycles):
        if not abort:
            ### Reading Pre resistance ############################################################################
            _, pre_data = k.read_resistance(start_range = start_range, voltage = V_read, points = points)
            pre_list.append(add_suffix_to_dict(pre_data,'_pre'))
            data = combine_lists_to_data_frame(pre_list, post_list, scope_list)
            iplots.updateline(data)
            ### Setting up scope  ################################################################################

            ttx.inputstate(1, False)
            ttx.inputstate(2, True)
            ttx.inputstate(3, False)
            ttx.inputstate(4, False)

            ttx.scale(2, scale)
            ttx.position(2, position)

            ttx.change_samplerate_and_recordlength(samplerate = 100e9, recordlength=recordlength)
            ttx.trigger_position(20)

            plt.pause(0.1)
            input('Connect RF probes')
            ttx.arm(source = 2, level = trigger_level, edge = 'e')

            ### Applying pulse and reading scope data #############################################################

            print('Apply pulse')
            plt.pause(0.2)
            status = ttx.triggerstate()
            while status == True:
                plt.pause(0.1)
                status = ttx.triggerstate()
            plt.pause(0.5)
            while ttx.busy():
                plt.pause(0.1)
            scope_data = ttx.get_curve(2)
            scope_list.append(scope_data)
            data = combine_lists_to_data_frame(pre_list, post_list, scope_list)
            iplots.updateline(data)

            ### Reading Post resistance ########################
            input('Connect DC probes')
            _, post_data = k.read_resistance(start_range = start_range, voltage = V_read, points = points)
            post_list.append(add_suffix_to_dict(post_data,'_post'))
            data = combine_lists_to_data_frame(pre_list, post_list, scope_list)
            iplots.updateline(data)
  
    data['amplitude'] = amplitude
    data['bits'] = bits
    data['scale'] = scale
    data['position'] = position
    data['trigger_level'] = trigger_level

    datafolder = os.path.join('C:\Messdaten', samplename, padname)
    subfolder = datestr
    file_exits = True
    i=1
    amplitude_decimal = (amplitude % 1)*10
    filepath = os.path.join(datafolder, subfolder, str(int(amplitude)) + 'p' + str(int(amplitude_decimal)) + '_amplitude_'+str(i))
    file_link = Path(filepath + '.df')
    while file_link.is_file():
        i +=1
        filepath = os.path.join(datafolder, subfolder, str(int(amplitude)) + 'p' + str(int(amplitude_decimal)) + '_amplitude_'+str(i))
        file_link = Path(filepath + '.df')
    io.write_pandas_pickle(meta.attach(data), filepath)

    return data
def eval_pcm_measurement(data, manual_evaluation = False):
    '''evaluates saved data (location or variable) from an  measurements. In case of a two channel measurement it determines pulse amplitude and width'''
    print('0')
    setup_pcm_plots()
    ########## declareation of buttons ###########
    def agree(self):
        print('agree')
        waitVar1.set(True)

    def threhsold_visible(self):
        print('threshold')
        pulse_minimum =min(v_answer)
        pulse_index = where(np.array(v_answer) < 0.5* pulse_minimum)
        pulse_start_index = pulse_index[0]
        pulse_start = t_scope[pulse_start_index]
        print(pulse_start)
        ax_dialog.set_title('Please indicate threshold')
        ax_dialog.plot(np.array([pulse_start,pulse_start]),np.array([-1,0.3]))
        ax_agree = plt.axes([0.59, 0.05, 0.1, 0.075])
        b_agree = Button(ax_agree,'Agree')
        b_agree.on_clicked(agree)
        cid = figure_handle.canvas.mpl_connect('pick_event', onpick)
        root.wait_variable(waitVar1)
        if not threshold_written_class.state:
            print(threshold_class.threshold)
            data['t_threshold'].append(threshold_class.threshold-pulse_start)
            threshold_written_class.state = True

        waitVar.set(True)


    def threshold_invisible(self):
        print('threshold_invisible')
        #print(threshold_written_class.state)
        if not threshold_written_class.state:
            data['t_threshold'].append(numpy.nan)
            threshold_written_class.state = True
        #print(threshold_written_class.state)
        waitVar.set(True)

    def onpick(event):
        print('onpick')
        ind = event.ind
        t_threshold = np.take(x_data, ind)
        print('onpick3 scatter:', ind, t_threshold, np.take(y_data, ind))
        threshold_class.set_threshold(t_threshold)
        if len(ind) == 1:
            ax_dialog.plot(np.array([t_threshold,t_threshold]),np.array([-1,0.3]))
            
            plt.pause(0.1)

    ######## beginning of main evalution #############
    if(type(data) == str):
        data = pd.read_pickle(data)
        iplots.show()    
    iplots.updateline(data)
    data['pulse_width'] = []
    data['pulse_amplitude'] = []
    data['t_threshold'] = []

    ########## if two channel experiment: ################
    if data['v_pulse']:      
        for t_scope, v_pulse in zip(data['t_scope'], data['v_pulse']):
            pulse_minimum =min(v_pulse)
            pulse_index = where(np.array(v_pulse) < 0.5* pulse_minimum)
            #pulse_end = t_scope[pulse_index[-1]]
            #pulse_start = t_scope[pulse_index[0]]
            v_max = max(v_pulse)
            v_min = min(v_pulse)
            if v_max > -v_min:
                pulse_width = calc_fwhm(valuelist = v_pulse, time = t_scope)
            else:
                pulse_width = calc_fwhm(valuelist = -v_pulse, time = t_scope)
            data['pulse_width'].append(pulse_width)
            data['pulse_amplitude'].append(np.mean(v_pulse[pulse_index])*2)
        
    ########## if one channel experiment: ################       
    else:
        for t_scope, v_answer in zip(data['t_scope'],data['v_answer']):
            #pulse_minimum =min(v_answer)
            #pulse_index = where(np.array(v_answer) < 0.5* pulse_minimum)
            #pulse_start_index = pulse_index[0]
            #pulse_start = t_scope[pulse_start_index]
            #print(pulse_start)
            #pulse_end_index = pulse_start_index + where(np.array(v_answer[pulse_start_index:-1]) >= 0)[0]
            #pulse_end = t_scope[pulse_end_index]
            
            # '''for short pulses the width is determined as FWHM, otherwise from pulse start until 
            #  the zero line is crossed for the first time '''
            # if pulse_end - pulse_start < 1e-9:
            #     pulse_end = t_scope[pulse_index[-1]]
            #     pulse_width = pulse_end - pulse_start
            # else:

            v_max = max(v_answer)
            v_min = min(v_answer)
            if v_max > -v_min:
                pulse_width = calc_fwhm(valuelist = v_answer, time = t_scope)
            else:
                pulse_width = calc_fwhm(valuelist = -v_answer, time = t_scope)

            data['pulse_width'].append(pulse_width)
            data['pulse_amplitude'].append(get_pulse_amplitude_of_PSPL125000(amplitude = data['amplitude'], bits = data['bits']))
            #import pdb; pdb.set_trace()
    ######## detection of threshold event by hand ###########
    if manual_evaluation:
        threshold_class = tmp_threshold()
        threshold_written_class = threshold_written()
        
        root = tk.Tk()
        root.withdraw()
        waitVar = tk.BooleanVar()
        waitVar1 = tk.BooleanVar()
        for t_scope, v_answer in zip(data['t_scope'], data['v_answer']):
            threshold_written_class.state = False
            x_data = t_scope
            y_data = v_answer/max(abs(v_answer))
            figure_handle, ax_dialog = plt.subplots()
            plt.title('Is a threshold visible?')
            plt.subplots_adjust(bottom=0.25)
            ax_dialog.plot(x_data,y_data, picker = True)
            ax_yes = plt.axes([0.7, 0.05, 0.1, 0.075])
            ax_no = plt.axes([0.81, 0.05, 0.1, 0.075])
            b_yes = Button(ax_yes, 'Yes')
            b_yes.on_clicked(threhsold_visible)
            b_no = Button(ax_no, 'No')
            b_no.on_clicked(threshold_invisible)
            root.wait_variable(waitVar)
            plt.close(figure_handle)
            #print(len(data['pulse_amplitude'])-len(data['t_threshold']))         
        plot_pcm_threshold(data)   
        root.destroy()
    return data

def eval_pcm_r_measurement(data, manual_evaluation = False, t_cap = np.nan, v_cap = np.nan, filename = ''):
    '''evaluates saved data (location or variable) from an  measurements. In case of a two channel measurement it determines pulse amplitude and width'''
    setup_pcm_plots_2()
    ########## declareation of buttons ###########
    # def agree(self):
    #     waitVar1.set(True)

    def threhsold_visible(self):
        user_aproove = False
        pulse_minimum =min(v_answer)
        pulse_index = where(np.array(v_answer)[1:-1] < 0.2* pulse_minimum) #ignoring first value which is ofter just wrong
        pulse_start_index = pulse_index[0]
        pulse_start = t_scope[pulse_start_index]
        #print(pulse_start)
        ax_dialog.set_title('Please indicate threshold')
        #plt.autoscale('False')
        ax_dialog.autoscale(False)
        ax_dialog.plot(np.array([pulse_start,pulse_start]),np.array([-1,0.3]))
        # ax_agree = plt.axes([0.59, 0.05, 0.1, 0.075])
        # b_agree = Button(ax_agree,'Agree')
        # b_agree.on_clicked(agree)
        above_threshold_level = where(np.array(v_diff/50 < -100e-6))
        try:
            threshold_event = where(above_threshold_level>pulse_start_index+6)[0]
        except:
            print('No threshold')
        while not user_aproove:
            threshold_index = above_threshold_level[threshold_event]-1
            t_threshold = t_scope[threshold_index]-pulse_start
            print(t_threshold)
            ax_dialog.plot(np.array([t_scope[threshold_index],t_scope[threshold_index]]),np.array([-1,0.3]))
            plt.pause(0.1)
            user_answer = input('Do you approve? ')            
            if user_answer == 'n':
                user_aproove = False
                threshold_event = where(above_threshold_level>above_threshold_level[threshold_event])[0]
            elif user_answer == 'd':
                t_threshold = np.nan
                user_aproove = True
            elif user_answer == 'y':
                user_aproove = True

        data['t_threshold'][x].append(t_threshold)
        #cid = figure_handle.canvas.mpl_connect('pick_event', onpick)
        # if not threshold_written_class.state:
        #     print(threshold_class.threshold)
        #     data['t_threshold'][x].append(threshold_class.threshold-pulse_start)
        #     threshold_written_class.state = True

        waitVar.set(True)


    def threshold_invisible(self):
        #print(threshold_written_class.state)
        if not threshold_written_class.state:
            data['t_threshold'][x].append(numpy.nan)
            threshold_written_class.state = True
        #print(threshold_written_class.state)
        waitVar.set(True)

    def onpick(event):
        ind = event.ind
        t_threshold = np.take(x_data, ind)
        #print('onpick3 scatter:', ind, t_threshold, np.take(y_data, ind))
        threshold_class.set_threshold(t_threshold)
        if len(ind) == 1:
            ax_dialog.plot(np.array([t_threshold,t_threshold]),np.array([-1,0.3]))
            
            plt.pause(0.1)

    ######## beginning of main evalution #############
    if(type(data) == str):
        data = pd.read_pickle(data)
        iplots.show()    
    iplots.updateline(data)
    data['pulse_width'] = [list() for x in range(len(data.index))]
    data['pulse_amplitude'] = [list() for x in range(len(data.index))]
    data['t_threshold'] = [list() for x in range(len(data.index))]
    ########## if two channel experiment: ################
    for x in range(len(data.index)):
        if 'v_pulse' in data.keys():   
            for t_scope, v_pulse in zip(data['t_scope'], data['v_pulse']):
               # pulse_minimum =min(v_pulse)
                #pulse_index = where(np.array(v_pulse) < 0.15* pulse_minimum)
                #pulse_end = t_scope[pulse_index[-1]]
                #pulse_start = t_scope[pulse_index[0]]
                v_max = max(v_pulse)
                v_min = min(v_pulse)
                if v_max > -v_min:
                    pulse_width = calc_fwhm(valuelist = v_pulse, time = t_scope)
                else:
                    pulse_width = calc_fwhm(valuelist = -v_pulse, time = t_scope)
                data['pulse_width'][x].append(pulse_width)
                data['pulse_amplitude'][x].append(np.mean(v_pulse[pulse_index])*2)
            
        ########## if one channel experiment: ################       
        else:
            for t_scope, v_answer in zip(data['t_ttx'],data['V_ttx']):

                v_max = max(v_answer)
                v_min = min(v_answer)
                if v_max > -v_min:
                    pulse_width = calc_fwhm(valuelist = v_answer, time = t_scope)
                else:
                    pulse_width = calc_fwhm(valuelist = -v_answer, time = t_scope)

                data['pulse_width'][x].append(pulse_width)
                data['pulse_amplitude'][x].append(get_pulse_amplitude_of_PSPL125000(amplitude = data['amplitude'][x], bits = data['bits'][x]))
                #import pdb; pdb.set_trace()
        ######## detection of threshold event by hand ###########

        if manual_evaluation:
            above_threshold_level = np.array(np.nan)
            threshold_event =np.nan
            threshold_class = tmp_threshold()
            threshold_written_class = threshold_written()
            
            root = tk.Tk()
            root.withdraw()
            waitVar = tk.BooleanVar()
            waitVar1 = tk.BooleanVar()
            for t_scope, v_answer in zip(data['t_ttx'], data['V_ttx']):
                threshold_written_class.state = False
                x_data = t_scope
                y_data = savgol_filter(v_answer,11,3)
                figure_handle, ax_dialog = plt.subplots()
                figure_handle.show()
                plt.title('Is a threshold visible?')
                plt.subplots_adjust(bottom=0.25)
                if type(t_cap) == float:
                    ax_dialog.plot(x_data,y_data, picker = True)
                else:
                    ax_dialog.plot(x_data,y_data/50, label = '$I_{\mathrm{Meas.}}$')
                    ax_dialog.plot(t_cap,v_cap/50, label = '$I_{\mathrm{Cap.}}$')
                    v_diff = subtract_capacitive_current(y_data, v_cap)
                    ax_dialog.plot(x_data, v_diff/50, label = '$I_{\mathrm{Diff.}}$', picker = True)
                    ax_dialog.set_xlabel('Time [s]')
                    ax_dialog.set_ylabel('Curreent [A]')
                    ax_dialog.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
                    ax_dialog.yaxis.set_major_formatter(mpl.ticker.EngFormatter())

                ax_yes = plt.axes([0.7, 0.05, 0.1, 0.075])
                ax_no = plt.axes([0.81, 0.05, 0.1, 0.075])
                b_yes = Button(ax_yes, 'Yes')
                b_yes.on_clicked(threhsold_visible)
                b_no = Button(ax_no, 'No')
                b_no.on_clicked(threshold_invisible)
                root.wait_variable(waitVar)
                if filename != '':
                    figure_handle.tight_layout()
                    figure_handle.savefig(filename + '.png', dpi =600)
                    plt.rcParams['pdf.fonttype'] = 'truetype'
                    figure_handle.savefig(filename + '.pdf')
                plt.close(figure_handle)
                #print(len(data['pulse_amplitude'])-len(data['t_threshold']))         
            root.destroy()
    return data

def eval_vcm_measurement(data, 
hrs_upper = np.nan, 
hrs_lower = np.nan, 
lrs_upper = np.nan,
lrs_lower = np.nan,
do_plots = True):
    impedance = 50
    if(type(data) == str):
        data = pd.read_pickle(data)
    
    if do_plots == True:
        setup_vcm_plots()
        iplots.updateline(data)
        iplots.show()
    #print(type(data))
    fwhm_list = []
    pulse_amplitude = []
    R_hrs = []
    R_lrs = []

    ###### Eval Reads ##########################

    for I_hrs, V_hrs, I_lrs, V_lrs in zip(data['I_hrs'], data['V_hrs'], data['I_lrs'], data['V_lrs']):
        r_hrs = determine_resistance(v = V_hrs, i = I_hrs)-impedance
        r_lrs = determine_resistance(v = V_lrs, i = I_lrs)-impedance
        if r_hrs > hrs_upper:
            r_hrs = np.nan
        if r_lrs > lrs_upper:
            r_lrs = np.nan
        if r_hrs < hrs_lower:
            r_hrs = np.nan
        if r_lrs < lrs_lower:
            r_lrs = np.nan
        R_hrs.append(r_hrs)
        R_lrs.append(r_lrs)

    ##### Eval Pulses ##########################

    for t_ttx, V_ttx, pulse_width in zip(data['t_ttx'], data['V_ttx'], data['pulse_width']):
        fwhm_value = calc_fwhm(valuelist = V_ttx, time = t_ttx)
        if fwhm_value < pulse_width:
            fwhm_list.append(pulse_width)
        else:
            fwhm_list.append(calc_fwhm(valuelist = V_ttx, time = t_ttx))

   
    data['R_hrs'] = R_hrs
    data['R_lrs'] = R_lrs
    data['fwhm'] = fwhm_list
    return data

def eval_all_pcm_measurements(filepath):
    ''' executes all eval_pcm_measurements in one directory and bundles the results'''
    print('b0')
    if filepath[-1] != '/':
        filepath = filepath + '/'
    files = os.listdir(filepath)
    all_data = []
    for f in files:
        filename = filepath+f
        print(filename)
        print('b01')
        all_data.append(eval_pcm_measurement(filename, manual_evaluation = True))
        print('b02')
    print('b1')
    t_threshold = np.array(all_data[0]['t_threshold'])
    pulse_amplitude = np.array(all_data[0]['pulse_amplitude'])
    t_threshold = []
    for data in all_data:
        if len(t_threshold)>0:
            print('b2')
            t_threshold = np.append(t_threshold,np.array(data['t_threshold']))
            pulse_amplitude = np.append(pulse_amplitude,np.array(data['pulse_amplitude']))
        else:
            print('b3')
            t_threshold = np.array(data['t_threshold'])
            pulse_amplitude = np.array(data['pulse_amplitude'])
    print('b4')
    plot_pcm_vt(pulse_amplitude, t_threshold)
    return all_data, t_threshold, pulse_amplitude

def eval_all_vcm_measurements(filepath, **kwargs):
    ''' executes all eval_vcm_measurements in one directory and bundles the results. Also error propagation is included.'''
    if filepath[-1] != '/':
        filepath = filepath + '/'
    files = os.listdir(filepath)
    all_data = []
    R_hrs_mean = []
    R_hrs_std = []
    R_lrs_mean = []
    R_lrs_std = []
    fwhm_mean = []
    fwhm_std = []
    R_ratio_mean =[]
    R_ratio_std = []

    for f in files:
        filename = filepath+f
        print(filename)
        data = eval_vcm_measurement(filename, **kwargs)
        all_data.append(data)
        R_hrs_mean.append(np.mean(data['R_hrs']))
        R_hrs_std.append(np.std(data['R_hrs']))
        R_lrs_mean.append(np.mean(data['R_lrs']))
        R_lrs_std.append(np.std(data['R_lrs']))
        fwhm_mean.append(np.mean(data['fwhm']))
        fwhm_std.append(np.std(data['fwhm']))
        R_ratio_mean.append(np.mean(data['R_lrs']/data['R_hrs']))
        R_ratio_std.append(R_ratio_mean[-1]*np.sqrt(np.power(R_hrs_std[-1]/R_hrs_mean[-1], 2)+np.power(R_lrs_std[-1]/R_lrs_mean[-1], 2)))

    return all_data, R_hrs_mean, R_hrs_std, R_lrs_mean, R_lrs_std, fwhm_mean, fwhm_std, R_ratio_mean, R_ratio_std

def eval_all_pcm_r_measurements(filepath, t_cap = np.nan, v_cap = np.nan):
    print('d1')
    if filepath[-1] != '/':
        filepath = filepath + '/'
    files = os.listdir(filepath)
    all_data = []
    for f in files:
        if 'data' in f or 'Thumbs' in f or '.png' in f:
            pass
        else:
            filename = filepath+f
            print(filename)
            all_data.append(eval_pcm_r_measurement(filename, manual_evaluation = True, t_cap = t_cap, v_cap = v_cap, filename = f))
    
    t_threshold = []
    pulse_amplitude = []
    R_pre = []
    R_post = []
    for data in all_data:
        t_threshold.append(np.array(data['t_threshold'][0])[0])
        pulse_amplitude.append(data['pulse_amplitude'][0])
        R_pre.append(np.mean(data['V_pre'][0]/data['I_pre'][0]))
        R_post.append(np.mean(data['V_post'][0]/data['I_post'][0]))
    plot_R_threshold(R_pre, t_threshold)
    print('Amplitude = ' + str(pulse_amplitude[0]) + 'V')
    
    export_data = {}
    export_data['all_data'] = all_data
    export_data['t_threshold'] = t_threshold
    export_data['pulse_amplitude'] = pulse_amplitude
    export_data['R_pre'] = R_pre
    export_data['R_post'] = R_post
    export_data = pd.DataFrame(export_data)
    

    file_name = os.path.join(filepath + 'data')
    file_link = Path(file_name + '.df')
    i=0
    while file_link.is_file():
        i +=1
        file_name = os.path.join(filepath + 'data_' + str(i))
        file_link = Path(file_name+ '.df')

    write_pandas_pickle(data, file_name)

    return all_data, t_threshold, pulse_amplitude, R_pre, R_post

def get_pulse_amplitude_of_PSPL125000(amplitude, bits):
    '''returns pulse amplitude in Volts depending on the measured output of the PSPL12500'''
    pulse_amplitude = np.nan
    if np.isnan(amplitude):
        return np.nan

    if bits == 1:
        amplitude_array = [0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        return_values = [2*0.728, 2*0.776, 2*0.8356, 2*1.1088, 2*1.5314, 2*2.0028, 
        2*2.306727, 2*2.622, 2*2.8624, 2*3.144727, 2*3.378, 2*3.652, 2*4.184]

    else:
        amplitude_array = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        return_values = [2*0.5956, 2*0.6481, 2*0.7188, 2*0.7757, 2*0.8182, 2*0.8952, 2*1.1693, 2*1.7592, 2*2.2008, 
        2*2.605455, 2*2.9248, 2*3.2552, 2*3.541818, 2*3.872, 2*4.2144, 2*4.7756]

    index = where(np.array(amplitude_array) == amplitude)

    if index.size > 0:
        pulse_amplitude = return_values[int(index)]
    else:
        print('Unknown amplitude')
        index_pre = where(np.array(amplitude_array) > amplitude)[0]

        x= amplitude%1
        pulse_amplitude = (x*return_values[int(index_pre+1)]+(1-x)*return_values[int(index_pre)])

    return pulse_amplitude

def plot_pcm_amp_comp(data, i = 0):
    fig1 = plt.figure()
    ax_cmp = plt.gca()

    for j in range(0,len(data)):
        ax_cmp.plot(data[j]['t_scope'][i],data[j]['v_answer'][i]/max(abs(data[j]['v_answer'][i])), 
        label=str(round(data[j]['pulse_amplitude'][i],2)) + ' V')

    fig1.show()
    ax_cmp.set_ylabel('Norm. voltage [a.u.]')
    ax_cmp.set_xlabel('Time [s]')
    ax_cmp.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    handles, labels = ax_cmp.get_legend_handles_labels()
    ax_cmp.legend(handles, labels, loc = 'lower right')

def plot_pcm_vt(pulse_amplitude, t_threshold):
    fig = plt.figure()
    ax_vt = plt.gca()
    ax_vt.semilogy(pulse_amplitude, t_threshold,'.k')
    ax_vt.set_ylabel('t_Threshold [s]')
    ax_vt.set_xlabel('Votlage [V]')
    ax_vt.yaxis.set_major_formatter(mpl.ticker.EngFormatter())
    fig.show()

class tmp_threshold():
    '''Allows to save the threshold obtained by clicking eval_pcm_measurement => there maybe a better solultion'''
    threshold = np.nan
    def set_threshold(self, threshold_value):
        if len(threshold_value) > 1:
            print('More than one point selected. Zoom closer to treshold event')
            self.threshold = numpy.nan
        else:
            self.threshold = threshold_value[0]

def add_suffix_to_dict(data, suffix):
    return {k+suffix:v for k,v in data.items()}

class threshold_written():
    state = False

def combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list = np.nan):
    hrs_df = pd.DataFrame(hrs_list)
    lrs_df = pd.DataFrame(lrs_list)
    scope_df = pd.DataFrame(scope_list)
    if sweep_list is not np.nan:
        sweep_df = pd.DataFrame(sweep_list)
        return_frame = pd.concat([hrs_df, lrs_df, scope_df, sweep_df] , axis = 1)
    else: 
        return_frame = pd.concat([hrs_df, lrs_df, scope_df] , axis = 1)
    expected = ['t_hrs','V_hrs','I_hrs',
            't_ttx','V_ttx',
            't_lrs','V_lrs','I_lrs',
            'V_sweep','I_sweep']
    for col in expected:
        if col not in return_frame.columns:
            return_frame[col] = np.nan
    return return_frame


###### save additional data for pick_off function (6 plotters) #############
def combine_2_scope_lists_to_data_frame(hrs_list, lrs_list, scope_list, scope_app_list, sweep_list = np.nan):
    hrs_df = pd.DataFrame(hrs_list)
    lrs_df = pd.DataFrame(lrs_list)
    scope_df = pd.DataFrame(scope_list)
    scope_app_df = pd.DataFrame(scope_app_list)

    if sweep_list is not np.nan:
        sweep_df = pd.DataFrame(sweep_list)
        return_frame = pd.concat([hrs_df, lrs_df, scope_df, scope_app_df, sweep_df] , axis = 1)
    else: 
        return_frame = pd.concat([hrs_df, lrs_df, scope_df,scope_app_df] , axis = 1)
    return return_frame

def calc_fwhm(valuelist, time, peakpos=-1):
    """calculates the full width at half maximum (fwhm) of some curve.
    the function will return the fwhm with sub-pixel interpolation. It will start at the maximum position and 'walk' left and right until it approaches the half values.
    INPUT: 
    - valuelist: e.g. the list containing the temporal shape of a pulse 
    OPTIONAL INPUT: 
    -peakpos: position of the peak to examine (list index)
    the global maximum will be used if omitted.
    OUTPUT:
    -fwhm (value)
    """
    if peakpos== -1: #no peakpos given -> take maximum
        peak = np.max(valuelist)
        peakpos = np.min( np.nonzero( valuelist==peak  )  )

    peakvalue = valuelist[peakpos]
    phalf = peakvalue / 2.0

    # go left and right, starting from peakpos
    ind1 = peakpos
    ind2 = peakpos   

    while ind1>2 and valuelist[ind1]>phalf:
        ind1=ind1-1
    while ind2<len(valuelist)-1 and valuelist[ind2]>phalf:
        ind2=ind2+1  
    #ind1 and 2 are now just below phalf
    grad1 = valuelist[ind1+1]-valuelist[ind1]
    grad2 = valuelist[ind2]-valuelist[ind2-1]
    #calculate the linear interpolations
    p1interp= ind1 + (phalf -valuelist[ind1])/grad1
    p2interp= ind2 + (phalf -valuelist[ind2])/grad2
    #calculate the width
    width = p2interp-p1interp

    ### calculate pulse widht
    time_step = time[1]-time[0]
    fwhm = width*time_step
    if np.isinf(fwhm):
        return np.nan 
    return fwhm

def determine_resistance(i, v):
    '''returns average resistance of all entries'''
    i = np.array(i)
    v = np.array(v)
    r = np.mean(v/i)
    if r < 0:
        return np.nan
    else:
        return r

def deb_to_atten(deb):
    return np.power(10, -deb/20)

def savefig2(fig_handle, location):
    location = location + '.fig.pickle'
    pickle.dump(fig_handle, open(location, 'wb'))

def openfig(location):
    fig = pickle.load(open(location, 'rb'))
    fig.show()
    return fig

def plot_pcm_transients(data):
    i=1
    fig, ax = plt.subplots()
    for time, voltage in zip(data['t_scope'], data['v_answer']):
        ax.plot(time, np.array(voltage)/50, label = str(i))
        i+=1
    ax.legend(loc = 'lower right')
    ax.set_ylabel('Current [A]')
    ax.set_xlabel('Time [s]')
    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    
    fig.tight_layout()
    fig.show()

def plot_pcm_threshold(data):
    fig, ax = plt.subplots()
    ax.semilogy(data['t_threshold'],'.')
    ax.set_xlabel('Pulse No')
    ax.set_ylabel('t_threshold [s]')
    fig.tight_layout()
    fig.show()


def transition_time(fwhm, R_ratio, upper_limit = 0.9, lower_limit = 0.1, reset = False):
    R_ratio = np.array(R_ratio)
    fwhm = np.array(fwhm)

    #sorting
    sorted_index = np.argsort(fwhm)
    fwhm=fwhm[sorted_index]
    R_ratio = R_ratio[sorted_index]

    if not reset:
        index = where(R_ratio < upper_limit)
        if index.size < 1: 
            return np.nan, np.nan, np.nan
        start_index = index[0]-1    #last entry at which all values are above the upper limit
        index = where(R_ratio > lower_limit)
        if index.size < 1: 
            return np.nan, np.nan, np.nan
        end_index = index[-1] + 1   #first entry at which all values are below the lower limit
    else:
        index = where(R_ratio > lower_limit)
        if index.size < 1: 
            return np.nan, np.nan, np.nan
        start_index = index[0]-1    #last entry at which all values are below the lower limit
        index = where(R_ratio < upper_limit)
        if index.size < 1: 
            return np.nan, np.nan, np.nan
        end_index = index[-1] + 1   #first entry at which all values are below the upper limit
    try:    
        t_start = fwhm[start_index]    
        t_end = fwhm[end_index]
    except:
        print('Out of array error => returning nan')
        return np.nan, np.nan, np.nan

    transition_time = t_end-t_start
    return transition_time, t_start, t_end

def roundup10(value):
    '''Rounds to the next higher order of magnitude. E.g. for a value of 3.17e-3 this function 
    would return 1e-2'''
    #Usefull for detecting the right range of a smu.
    log_value = np.log10(value)
    exponent = np.ceil(log_value)
    return np.power(10,exponent) 

def set_time(fwhm, R_ratio, limit = 0.5, reset = False):
    fwhm = np.array(fwhm)
    R_ratio = np.array(R_ratio)
    
    #sorting
    sorted_index = np.argsort(fwhm)
    fwhm=fwhm[sorted_index]
    R_ratio = R_ratio[sorted_index]

    if not reset:
        index = where(R_ratio > limit)
        if index.size < 1: 
            return fwhm[0]
        t_set_index = index[-1] + 1
    else:
        index = where(R_ratio < limit)
        if index.size < 1: 
            return fwhm[0]
        t_set_index = index[-1] + 1

    try:
        t_set = fwhm[t_set_index]
    except:
        print('Out of array error => returning nan')
        return np.nan
    return t_set

def get_R_median(all_data):
    R = []
    fwhm_array =  []
    for data in all_data:
        ratio = np.array(data['R_lrs']/data['R_hrs'])
        index = where(~np.isnan(ratio))
        ratio = ratio[index]
        R.append(np.median(ratio))
        fwhm_array.append(np.mean(data['fwhm']))
    fwhm_array = np.array(fwhm_array)
    R = np.array(R)   
    sorted_index = np.argsort(np.array(fwhm_array))
    fwhm_array = fwhm_array[sorted_index]
    R = R[sorted_index]     
    return fwhm_array, R

def Boxplot_array(all_data, return_resistance = False):
    R = []
    fwhm_array = []
    R_lrs = []
    R_hrs = []
    for data in all_data:
        ratio = np.array(data['R_lrs']/data['R_hrs'])
        index = where(~np.isnan(ratio))
        #index = where((data['R_lrs']<35e3) & (data['R_hrs']>10e3)) # ZrOx
        #index = where((data['R_lrs']<35e3) & (data['R_hrs']>10e3)) # TaOx
        ratio = ratio[index]
        rl = np.array(data['R_lrs'])[index]
        rh = np.array(data['R_hrs'])[index]
        if np.size(ratio)>0:
            R.append(ratio)
            R_lrs.append(rl)
            R_hrs.append(rh)
            fwhm_array.append(np.mean(data['fwhm']))

    fwhm_array = np.array(fwhm_array)
    R = np.array(R)
    R_lrs = np.array(R_lrs)
    R_hrs = np.array(R_hrs)

    sorted_index = np.argsort(np.array(fwhm_array))
    fwhm_array = fwhm_array[sorted_index]
    R = R[sorted_index]
    R_lrs = R_lrs[sorted_index]
    R_hrs = R_hrs[sorted_index]

    R = np.ndarray.tolist(R)
    fwhm_array = np.ndarray.tolist(fwhm_array)
    R_lrs = np.ndarray.tolist(R_lrs)
    R_hrs = np.ndarray.tolist(R_hrs)

    if return_resistance:
        return fwhm_array, R_lrs, R_hrs
    else:
        return fwhm_array, R

def plot_R_threshold(r, t):
    fig, ax = plt.subplots()
    ax.loglog(r,t,'.')
    ax.set_xlabel('Resistance [$\Omega$]')
    ax.set_ylabel('$t_{\mathrm{Threshold}}$ [s]')
    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter())
    fig.tight_layout()
    fig.show()
    return fig, ax

def plot_R_threshold_color(r, t):
    fig, ax = plt.subplots()
    sc = ax.scatter(r, t, cmap = 'rainbow', c = np.arange(len(t)))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(left=1e4, right =1.2e7)
    ax.set_ylim(bottom = 1e-10, top = 12e-9)
    ax.set_xlabel('Resistance [$\Omega$]')
    ax.set_ylabel('$t_{\mathrm{Threshold}}$ [s]')
    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    #ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter())
    plt.colorbar(sc)
    fig.tight_layout()
    fig.show()
    return fig, ax

def subtract_capacitive_current(v_meas, v_cap, trigger_position = 20):
    start_index_meas = int(trigger_position/100*len(v_meas))
    start_index_cap = int(trigger_position/100*len(v_cap))
    begin_cap = start_index_cap-start_index_meas
    end_cap = len(v_meas) + start_index_cap -start_index_meas
    try:
        v_diff = v_meas - v_cap[begin_cap:end_cap]
    except:
        v_diff = v_meas
        print('v_cap to short')
    return v_diff

def hs(x=0):
    return np.heaviside(x,1)

def threshold(time, t_start= 7.7e-9, t_diff = 0.7e-9, r_start = 1e6, r_end = 1200):
    r_diff = r_start-r_end
    t_end = t_start + t_diff
    r_slope = r_diff/t_diff
    return r_start - hs(t_end-time)*hs(time-t_start)*r_slope*(time-t_start)

def complex_interpolation(x, xp, yp, **kwargs):
    f_real = interp1d(xp, np.real(yp), **kwargs)
    f_imag = interp1d(xp, np.imag(yp), **kwargs)
    return f_real(x) + 1j*f_imag(x)

def calculate_transmission(file, t_signal, v_signal, 
rf_file = None,  
t_meas = [],
v_meas = [],  
do_plots = False,
show_results = True,
time_shift = 0,
reflection_offset = 0,
transmission_offset = 0,
return_figs = False, 
conjugate = False,
cut_off_frequency = None):
    '''uses scattering parameters of a device and a applied signal to calculate the transmission through and the reflection from the device'''
    ntwk_kHz = rf.Network(file)
    frequencies_kHz = ntwk_kHz.f

    s11_kHz = ntwk_kHz.s11.s[:,0,0]
    s11angle_kHz = ntwk_kHz.s11.s_rad_unwrap[:,0,0]/np.max(np.abs(ntwk_kHz.s11.s_rad_unwrap[:,0,0]))
    s11mag_kHz = ntwk_kHz.s11.s_db[:,0,0]

    s21_kHz = ntwk_kHz.s21.s[:,0,0]
    s21angle_kHz = ntwk_kHz.s21.s_rad_unwrap[:,0,0]/np.max(np.abs(ntwk_kHz.s21.s_rad_unwrap[:,0,0]))
    s21mag_kHz = ntwk_kHz.s21.s_db[:,0,0]

    if rf_file != None:
        ntwk_MHz = rf.Network(rf_file)
        frequencies_MHz = ntwk_MHz.f

        s11_MHz = ntwk_MHz.s11.s[:,0,0]
        s11angle_MHz = ntwk_MHz.s11.s_rad_unwrap[:,0,0]/np.max(np.abs(ntwk_MHz.s11.s_rad_unwrap[:,0,0]))
        s11mag_MHz = ntwk_MHz.s11.s_db[:,0,0]

        s21_MHz = ntwk_MHz.s21.s[:,0,0]
        s21angle_MHz = ntwk_MHz.s21.s_rad_unwrap[:,0,0]/np.max(np.abs(ntwk_MHz.s21.s_rad_unwrap[:,0,0]))
        s21mag_MHz = ntwk_MHz.s21.s_db[:,0,0]

    if do_plots or return_figs:
        fig_s, ax_s = plt.subplots()
        if rf_file != None:
            ax_s.semilogx(frequencies_kHz, s21mag_kHz, color = 'blue', label = 'S$_{21}$ (k)')
            ax_s.semilogx(frequencies_MHz, s21mag_MHz,'--', color = 'blue', label = 'S$_{21}$ (M)')
            ax_s.semilogx(frequencies_kHz, s11mag_kHz, color = 'green', label = 'S$_{11}$ (k)')
            ax_s.semilogx(frequencies_MHz, s11mag_MHz,'--', color = 'green', label = 'S$_{11}$ (M)')
            ax_s.set_xbound(np.min(frequencies_kHz), np.max(frequencies_MHz))
        else:
            ax_s.semilogx(frequencies_kHz, s21mag_kHz, color = 'blue', label = 'S$_{21}$')
            ax_s.semilogx(frequencies_kHz, s11mag_kHz, color = 'green', label = 'S$_{11}$')
            ax_s.set_xbound(np.min(frequencies_kHz), np.max(frequencies_kHz))
        ax_s.set_xlabel('Frequency [Hz]')
        ax_s.set_ylabel('Magnitude [dB]')
        ax_s.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        ax_s.legend()
        fig_s.tight_layout()
        if do_plots:
            fig_s.show()

        fig_ph, ax_ph = plt.subplots()
        if rf_file != None:
            ax_ph.semilogx(frequencies_kHz, s21angle_kHz/np.pi, color = 'blue', label = 'S$_{21}$ (k)')
            ax_ph.semilogx(frequencies_MHz, s21angle_MHz/np.pi,'--', color = 'blue', label = 'S$_{21}$ (M)')
            ax_ph.semilogx(frequencies_kHz, s11angle_kHz/np.pi, color = 'green', label = 'S$_{11}$ (k)')
            ax_ph.semilogx(frequencies_MHz, s11angle_MHz/np.pi,'--', color = 'green', label = 'S$_{11}$ (M)')
            ax_ph.set_xbound(np.min(frequencies_kHz), np.max(frequencies_MHz))
        else:
            ax_ph.semilogx(frequencies_kHz, s21angle_kHz/np.pi, color = 'blue', label = 'S$_{21}$')
            ax_ph.semilogx(frequencies_kHz, s11angle_kHz/np.pi, color = 'green', label = 'S$_{11}$')
            ax_ph.set_xbound(np.min(frequencies_kHz), np.max(frequencies_kHz))
        ax_ph.set_xlabel('Frequency [Hz]')
        ax_ph.set_ylabel('Angle [rad]')
        ax_ph.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        ax_ph.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%g $\pi$'))
        ax_ph.legend()
        fig_ph.tight_layout()
        if do_plots:
            fig_ph.show()

    ################  interpolation and concatenation of the kMz and MHz regime ##################

    if rf_file != None:
        idx_overlap = (frequencies_kHz >= np.min(frequencies_MHz)) & (frequencies_kHz <= np.max(frequencies_kHz))
        overlapFrequencies = frequencies_kHz[idx_overlap]
        overlap_s11_MHz_interp= complex_interpolation(overlapFrequencies, frequencies_MHz, s11_MHz, kind='cubic')
        overlap_s21_MHz_interp= complex_interpolation(overlapFrequencies, frequencies_MHz, s21_MHz, kind='cubic')

        idx_kHz = (frequencies_kHz < np.min(frequencies_MHz)) 
        idx_MHz = (frequencies_MHz > np.max(frequencies_kHz)) 
        frequencies_combined = np.concatenate([frequencies_kHz[idx_kHz], overlapFrequencies, frequencies_MHz[idx_MHz]])
        s11_combined = np.concatenate([s11_kHz[idx_kHz], overlap_s11_MHz_interp, s11_MHz[idx_MHz]]) 
        s21_combined = np.concatenate([s21_kHz[idx_kHz], overlap_s21_MHz_interp, s21_MHz[idx_MHz]]) 

        if do_plots or return_figs:
            fig_tf0, ax_tf0 = plt.subplots()
            fig_tf1, ax_tf1 = plt.subplots()
            ax_tf0.loglog(frequencies_combined, np.abs(s21_combined))
            ax_tf1.semilogx(frequencies_combined, np.angle(s21_combined))

            fig_rf0, ax_rf0 = plt.subplots()
            fig_rf1, ax_rf1 = plt.subplots()
            ax_rf0.loglog(frequencies_combined, np.abs(s11_combined))
            ax_rf1.semilogx(frequencies_combined, np.angle(s11_combined))

        transferFunction = s21_combined
        reflectionFunction = s11_combined
    else:
        frequencies_combined = frequencies_kHz
        if do_plots or return_figs:
            fig_tf0, ax_tf0 = plt.subplots()
            fig_tf1, ax_tf1 = plt.subplots()
            ax_tf0.loglog(frequencies_combined, np.abs(s21_kHz))
            ax_tf1.semilogx(frequencies_combined, np.angle(s21_kHz))

            fig_rf0, ax_rf0 = plt.subplots()
            fig_rf1, ax_rf1 = plt.subplots()
            ax_rf0.loglog(frequencies_combined, np.abs(s11_kHz))
            ax_rf1.semilogx(frequencies_combined, np.angle(s11_kHz))

        transferFunction = s21_kHz
        reflectionFunction = s11_kHz
        if conjugate:
            transferFunction = np.conj(transferFunction)
            reflectionFunction = np.conj(reflectionFunction)
    if do_plots or return_figs:
        ax_tf = [ax_tf0, ax_tf1]
        ax_rf = [ax_rf0, ax_rf1]
    ################################### get signal and perform fft #####################################

    #L = len(t_signal) # length of the signal
    #Fs = L/abs(max(t_signal)-min(t_signal)) #sampling Frequency
    #f = Fs*np.arange(0, L/2+1)/L # frequency content of the signal

    Signal_f = np.fft.rfft(v_signal)
    f = np.fft.rfftfreq(np.size(t_signal), t_signal[1]-t_signal[0])

    if do_plots or return_figs:
        fig_fft, ax_fft = plt.subplots()
        ax_fft.grid(True)
        ax_fft.loglog(f, np.abs(Signal_f), linewidth = 1)
        ax_fft.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        #ax_fft.set_title('Frequency content of Signal. Fs = ' + str(round(Fs/1e9, 0)) + ' GHz')
        ax_fft.set_xlabel('Frequency [Hz]')
        ax_fft.set_ylabel('|P1(f)|')
        if do_plots:
            fig_fft.show()

    ###################### interpolate transfer function to frequency content of signal #################################
    idx_extrapolation = np.where(f>np.max(frequencies_combined))[0]

    abs_transferFunction_interp_f = interp1d(frequencies_combined, np.abs(transferFunction), kind = 'cubic', fill_value = "extrapolate")
    angle_transferFunction_interp_f = interp1d(frequencies_combined, np.unwrap(np.angle(transferFunction)), kind = 'cubic', fill_value = "extrapolate")
    abs_transferFunction_interp = abs_transferFunction_interp_f(f)
    if len(idx_extrapolation) > 0:
        abs_transferFunction_interp[idx_extrapolation] = abs_transferFunction_interp[idx_extrapolation[0]-1]
    
    abs_reflectionFunction_interp_f = interp1d(frequencies_combined, np.abs(reflectionFunction), kind = 'cubic', fill_value="extrapolate")
    angle_reflectionFunction_interp_f = interp1d(frequencies_combined, np.unwrap(np.angle(reflectionFunction)), kind = 'cubic', fill_value="extrapolate")
    abs_reflectionFunction_interp = abs_reflectionFunction_interp_f(f)
    if len(idx_extrapolation) > 0:
        abs_reflectionFunction_interp[idx_extrapolation] = abs_reflectionFunction_interp[idx_extrapolation[0]-1]

    idx = (abs_transferFunction_interp > np.max(np.abs(transferFunction)))   
    abs_transferFunction_interp[idx] = np.max(np.abs(transferFunction))    
    if abs_transferFunction_interp[0] < 0:
        abs_transferFunction_interp[0] =  np.max(np.abs(transferFunction))
    angle_transferFunction_interp = angle_transferFunction_interp_f(f)
    if len(idx_extrapolation) > 0:
        angle_transferFunction_interp[idx_extrapolation] = angle_transferFunction_interp[idx_extrapolation[0]-1]
    transferFunction_interp = abs_transferFunction_interp*np.exp(1j*np.unwrap(angle_transferFunction_interp))
    
    idx_r = (abs_reflectionFunction_interp > np.max(np.abs(reflectionFunction)))
    abs_reflectionFunction_interp[idx_r] = np.max(np.abs(reflectionFunction))
    if abs_reflectionFunction_interp[0] < 0:
        abs_reflectionFunction_interp[0] =  np.max(np.abs(reflectionFunction))
    angle_reflectionFunction_interp = angle_reflectionFunction_interp_f(f)
    if len(idx_extrapolation) > 0:
        angle_reflectionFunction_interp[idx_extrapolation] = angle_reflectionFunction_interp[idx_extrapolation[0]-1]
    reflectionFunction_interp = abs_reflectionFunction_interp*np.exp(1j*np.unwrap(angle_reflectionFunction_interp))

    if do_plots or return_figs:
        ax_tf0.loglog(f, np.abs(transferFunction_interp), 'r-')
        ax_tf1.semilogx(f, np.angle(transferFunction_interp), 'r-') 
        ax_tf0.set_ylabel('abs(Tf)')
        ax_tf1.set_ylabel('angle(Tf)')
        for a in ax_tf:
            a.set_xlabel('Frequency [Hz]')
            a.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        if do_plots:
            fig_tf0.show()
            fig_tf1.show()

        ax_rf0.loglog(f, np.abs(reflectionFunction_interp), 'r-')
        ax_rf1.semilogx(f, np.angle(reflectionFunction_interp), 'r-') 
        ax_rf0.set_ylabel('abs(Rf)')
        ax_rf1.set_ylabel('angle(Rf)')
        for a in ax_rf:
            a.set_xlabel('Frequency [Hz]')
            a.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        if do_plots:
            fig_rf0.show()
            fig_rf1.show()


    ############################# convolute and inverse fourier transformat #######################################

    Signal_f_conv = Signal_f*transferFunction_interp
    Signal_f_conv_r = Signal_f*reflectionFunction_interp
    if cut_off_frequency != None:
        idx_f = np.where(f > cut_off_frequency)[0]
        Signal_f_conv[idx_f] = 0
    Signal_t_conv  = np.fft.irfft(Signal_f_conv) - transmission_offset
    Signal_t_conv_r  = np.fft.irfft(Signal_f_conv_r) - reflection_offset
    if len(v_signal) > len(Signal_t_conv_r):
        t_signal = t_signal[1:]
        v_signal = v_signal[1:]
    v_stimulus = v_signal + Signal_t_conv_r - Signal_t_conv 

    if show_results or return_figs:
        fig_sig, ax_sig = plt.subplots()
        ax_sig.set_xlabel('Time [s]')
        ax_sig.set_ylabel('Voltage [V]')
        ax_sig.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        
        if len(t_meas) >= 1 and len(v_meas) >= 1:
            ax_sig.plot(t_meas, v_meas, label = 'Measurement')
        ax_sig.plot(t_signal, Signal_t_conv, label = 'Calculation')
        ax_sig.legend()
        if show_results:
            fig_sig.show()

        fig_refl, ax_refl = plt.subplots()
        ax_refl.set_xlabel('Time [s]')
        ax_refl.set_ylabel('Voltage [V]')
        ax_refl.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        ax_refl.plot(t_signal, v_signal, label = 'Signal')
        ax_refl.plot(t_signal, Signal_t_conv_r, label = 'Reflection')
        ax_refl.plot(t_signal, v_stimulus, label = 'Stimulus')
        ax_refl.legend()
        if show_results:
            fig_refl.show()

    if return_figs == False:
        return t_signal, Signal_t_conv_r, Signal_t_conv
    else:
        fig = [fig_s, fig_ph, fig_tf0, fig_tf1, fig_rf0, fig_rf1, fig_fft, fig_sig, fig_refl]
        ax = [ax_s, ax_ph, ax_tf0, ax_tf1, ax_rf0, ax_rf1, ax_fft, ax_sig, ax_refl]
        return t_signal, Signal_t_conv_r, Signal_t_conv, fig, ax

def calc_t_SET(t_meas_raw, v_meas_raw, v_capa_raw, factor = 0.2, do_plots = False):


    v_meas_f = interp1d(t_meas_raw, v_meas_raw)#, kind = 'cubic')
    v_capa_f = interp1d(t_meas_raw, v_capa_raw)#, kind = 'cubic')
    t_meas = np.arange(t_meas_raw[0], t_meas_raw[-1], 1e-13)
    v_meas = v_meas_f(t_meas)
    v_capa = v_capa_f(t_meas)
    v_meas = savgol_filter(v_meas, 10001, 3)
    v_capa = savgol_filter(v_capa, 10001, 3)
    v_capa_max = np.max(np.abs(v_capa))
    idx_meas_10 = np.where(np.abs(v_meas) > factor*v_capa_max)[0][0]
    idx_capa_10 = np.where(np.abs(v_capa) > factor*v_capa_max)[0][0]
    difference = idx_capa_10 - idx_meas_10
    if difference != 0: # idx_meas_10 > idx_capa_10
       v_capa_new = shift(v_capa, -difference, cval=np.NaN)
       v_capa = v_capa_new[~np.isnan(v_capa_new)]
       v_meas = v_meas[~np.isnan(v_capa_new)]
       t_meas = t_meas[~np.isnan(v_capa_new)]
    v_device = v_meas-v_capa
    #v_device = savgol_filter(v_device_, 10001, 3)
    v_device_max = np.max(np.abs(v_device))
    idx_meas_10 = np.where(np.abs(v_meas) > factor*v_capa_max)[0][0]
    idx_device_10 = np.where(np.abs(v_device) > factor*v_device_max)[0][0]
    t_meas = t_meas - t_meas[idx_meas_10]
    t_set = t_meas[idx_device_10] - t_meas[idx_meas_10]
    if do_plots:
        fig, ax = plt.subplots()
        ax.plot(t_meas, v_meas)
        ax.plot(t_meas, v_capa)
        #ax.plot(t_meas, v_device_raw)
        ax.plot(t_meas, v_device)
        ax.vlines([t_meas[idx_device_10], t_meas[idx_meas_10]], ymin = -0.85, ymax= 0.6)
        fig.tight_layout()
        fig.show()
    return t_set

def calc_t_reset(filename, min_current = -215e-6):
    data = eval_vcm_measurement(filename, do_plots = False)
    print('$R_pre = ' + str(data['R_hrs'][0]))
    print('$R_post = ' + str(data['R_lrs'][0]))
    v = np.array(data['V_ttx'][0][1:])
    i = -v/50
    t = data['t_ttx'][0][1:]
    i_sg = i
    i_sg = savgol_filter(i, 15, 3)

    i_max = np.max(np.abs(i_sg))
    i_min = np.abs(min_current)
    i_diff = i_max-i_min
    i_half = i_min + 0.5*i_diff

    idx_max = np.where(np.abs(i_sg) == i_max)[0][0]
    idx_start = np.where(np.abs(i_sg) >= 0.2*i_max)[0][0]
    idx_half = idx_max + np.where(np.abs(i_sg[idx_max:]) >= i_half)[0][-1]+1

    t_start = t[idx_start]
    t_max = t[idx_max]
    t_half = t[idx_half]

    t_old = t_half - t_max
    t_new = t_half - t_start

    print(t_new)

    fig, ax = plt.subplots()

    ax.plot(t*1e9, i_sg*1e3)
    ymin, ymax = ax.get_ybound()
    xmin, xmax = ax.get_xbound()
    #ymin -= 0.2
    

    ax.vlines([t_start*1e9, t_half*1e9], ymin, ymax, linestyle = 'dotted')
    ax.hlines([i_half*1e3, i_max*1e3, i_min*1e3], xmin, xmax, linestyle = 'dotted')

    # ax.annotate('', xy=(1.93, i_min*1e3), xytext=(1.93, i_max*1e3), arrowprops=dict(facecolor='black', arrowstyle='<->'),)
    # ax.annotate('$\Delta I$' , xy=(1.8, 1.2), xytext=(1.8, 1.2), fontsize = 8)

    # ax.annotate('', xy=(1.7, i_min*1e3), xytext=(1.7, i_half*1e3), arrowprops=dict(facecolor='black', arrowstyle='<->'),)
    # ax.annotate('$\Delta I$/2' , xy=(1.5, 0.5), xytext=(1.5, 0.5), fontsize = 8)

    # ax.annotate('', xy=(t_max*1e9, -0.03), xytext=(t_half*1e9, -0.03), arrowprops=dict(facecolor='black', arrowstyle='<->'),)
    # t_old_ps = t_old*1e12
    # arrow_label = str("%.0f" % t_old_ps) + ' ps'
    # ax.annotate(arrow_label , xy=(t_max*1e9, 0.055), xytext=(t_max*1e9+0.04, 0.055), fontsize = 8)

    ax.annotate('', xy=(t_start*1e9, 0), xytext=(t_half*1e9, 0), arrowprops=dict(facecolor='black', arrowstyle='<->'),)
    t_new_ps = t_new*1e12
    arrow_label = str("%.0f" % t_new_ps) + ' ps'
    ax.annotate(arrow_label , xy=(t_start*1e9, 0.01), xytext=(t_start*1e9+0.02, 0.01), fontsize = 8)
    ax.set_ybound(ymin, ymax)
    ax.set_xbound(xmin, xmax)

    ax.set_xlabel('Time [ns]', fontsize = 9)
    ax.set_ylabel('Current [mA]', fontsize = 9)
    #ax.legend(fontsize = 8, loc = 'lower right')
    ax.tick_params(direction = 'in', top = True, right = True, labelsize = 8)

    fig.tight_layout()
    fig.show()



    #### making list for set_pattern
    # Widthfactor: number of ones, num_delayFactor: number of zeros between ones, num_of_pulses: total pulses repeated

def pattern_list(widthFactor, num_delayFactor, num_of_pulses):
    patternOn = [1]*widthFactor
    patternOff = [0]*num_delayFactor

    out = patternOn[:]
 
    for i in range(num_of_pulses-1):      
        out.extend(patternOff)
        out.extend(patternOn)    

    return out



def find_delays(set_pattern):
    # Count the total number of ones
    number_of_positive_pulses =  set_pattern.count(1)
    number_of_negative_pulses= set_pattern.count(-1)
    # Find the index of the first occurrence of 1
    first_one_index = set_pattern.index(1)

    try:
        # Find the index of the second occurrence of 1, starting the search after the first 1
        second_one_index = set_pattern.index(1, first_one_index + 1)

        # Count the number of zeros between the first and second occurrences of 1
        delay_factor = set_pattern[first_one_index + 1:second_one_index].count(0)
        
    except ValueError:
        # If there's no second occurrence of 1, count zeros after the first 1
        delay_factor = set_pattern[first_one_index + 1:].count(0)

    return delay_factor, number_of_positive_pulses, number_of_negative_pulses




def PPG30_measurement_old(samplename,
padname,
v1,
v2,
positive_amplitude = 1000e-3,
negative_amplitude = 600e-3,
set_pattern_format = 'LU',  # Since only LU pattern is user giver pattern
set_pattern_length = 1,     # Set word lenght. But since it is autmatically (line 329 in sympuls.PG30) selected so we dont have to use this. 
widthFactor = 1,            # (Daniel way of word function, line 2532 in this file). use in a function to give 1s, number of pulses and delays.
num_delayFactor = 0,        # Part of function to give 1s, number of pulses and delays
num_of_pulses = 0,          # Part of function to give 1s, number of pulses and delays
set_pattern = pattern_list, # give pattern in the form of list. [1,0,-1]
set_trigger_soruce = 'IMM',
set_ppg_channel = 'BIP',
step = 0.01,
step2 = 0.01,
V_read = 0.2,
range_lrs = 1e-3,
range_hrs = 1e-4,
range_sweep = 1e-2,
range_sweep2 = 1e-3,
cycles = 1,
pulse_width = 50e-12 ,
attenuation = 0,
automatic_measurement = True,
pg5_measurement = True,
recordlength = 250,
trigger_position = 25,
edge = 'r',
sweep = True,
two_sweeps = False,
scale = 0.12,
position = -3,
trigger_level = 0.05,
nplc = 10,
limitI = 3e-4,
limitI2 = 3e-4,
r_window = False,
r_lower = 1e3,
r_upper = 2e3,
cc_step = 25e-6):

    setup_vcm_plots()
    data = {}
    data['padname'] = padname
    data['samplename'] = samplename

    data['positive_amplitude'] = positive_amplitude
    data['negative_amplitude'] = negative_amplitude

    ##### Daniel way of setting pattern ####
    set_pattern  = pattern_list(widthFactor= widthFactor, num_delayFactor = num_delayFactor, num_of_pulses = num_of_pulses )
    #print(len(set_pattern))

    #data['set_pattern'] = set_pattern
    #data['set_pattern_format'] = set_pattern_format
    #data['set_pattern_length'] = set_pattern_length
    #data['set_ppg_channel'] = set_ppg_channel

    data['pulse_width'] = pulse_width
    #data['attenuation'] = attenuation
    #data['recordlength'] = recordlength
    #data['V_read'] = V_read
    

    #data['nplc'] = nplc
    #data['cycles'] = cycles

    ###### Setting PPG30 ###############################################################

    sympulsPG30.amplitude1(Amp1 = positive_amplitude)
    sympulsPG30.amplitude2(Amp2 = negative_amplitude)
    sympulsPG30.pattern(pattern = set_pattern_format)
    sympulsPG30.format(form = set_ppg_channel)

    sympulsPG30.set_lupattern(pattern = set_pattern)
    #set_length = np.ceil(len(set_pattern)/128)                            # setting number of words on PPG30 dividing by digits
    #sympulsPG30.set_lupattern_length(num_words = set_pattern_length)      # uncomment if we have to use user given word.
    sympulsPG30.trigger_source(trig = set_trigger_soruce)

    ##delay_factor, number_of_positive_pulses, number_of_negative_pulses =find_delays(set_pattern)
    ##delay = delay_factor * pulse_width *1e12                               # find delays by multiplying pulsewidth with delays factor

    delay = num_delayFactor * pulse_width *1e12                              # find delays by multiplying pulsewidth with delays factor

    ##Number_of_pulses = number_of_positive_pulses + number_of_negative_pulses
    total_pulse_duration = ((pulse_width *widthFactor)+(num_delayFactor*pulse_width))*num_of_pulses       # count  pulses with delays

    pos_amp_deb = round(deb_to_atten(attenuation)*2*positive_amplitude,2)  # as PPG30 have variable amplitude so we have to calculate actual amplitude
    pos_amp = str(pos_amp_deb).replace('.', 'p')



    hrs_list = []
    lrs_list = []
    sweep_list = []
    scope_list = []
    
    vlist = tri(v1 = v1, v2 = v2, step = step)

    abort = False
    for i in range(cycles):
        if not abort:
            ### Reading HRS resistance ############################################################################
            k.source_output(ch = 'A', state = True)
            k.source_level(source_val= V_read, source_func='v', ch='A')

            plt.pause(1)

            k._it_lua(sourceVA = V_read, sourceVB = 0, points =10, interval = 0.1, rangeI = range_hrs , limitI = 1, nplc = nplc)
            while not k.done():
                plt.pause(0.1)
            k.source_output(ch = 'A', state = False)
            k.source_output(ch = 'B', state = False)
            hrs_data = k.get_data()
            hrs_list.append(add_suffix_to_dict(hrs_data,'_hrs'))
            data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
            iplots.updateline(data)
            ### Setting up scope  ################################################################################

            ttx.inputstate(3, True)
            ttx.inputstate(2, False)
            ttx.inputstate(1, False)
            ttx.inputstate(4, False)

            ttx.scale(3, scale)
            ttx.position(3, position)


            ttx.change_samplerate_and_recordlength(samplerate = 100e9, recordlength= recordlength)
            ttx.trigger_position(trigger_position)

            plt.pause(0.1)

            ttx.arm(source = 3, level = trigger_level, edge = edge)


            ### Applying pulse and reading scope data #############################################################
            if pg5_measurement:
                sympulsPG30.set_pulse_width(pulse_width)
            if not automatic_measurement:
                input('Connect the RF probes and press enter')
                plt.pause(0.5)
            else:
                plt.pause(1)
                
            if pg5_measurement:
                sympulsPG30.trigger()
            else:
                print('Apply pulse')
            plt.pause(0.1)
      
            plt.pause(0.2)
            status = ttx.triggerstate()
            while status == True:
                plt.pause(0.1)
                status = ttx.triggerstate()
            plt.pause(0.5)
            scope_list.append(ttx.get_curve(3))
            data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
            iplots.updateline(data)

            ### Reading LRS resistance #############################################################################

            if not automatic_measurement:
                input('Connect the DC probes and press enter')

            k.source_output(ch = 'A', state = True)
            k.source_level(source_val= V_read, source_func='v', ch='A')

            plt.pause(1)
            k._it_lua(sourceVA = V_read, sourceVB = 0, points = 10, interval = 0.1, rangeI = range_lrs, limitI = 1, nplc = nplc)
            while not k.done():
                plt.pause(0.1)
            k.source_output(ch = 'A', state = False)
            k.source_output(ch = 'B', state = False)
            lrs_data = k.get_data()
            lrs_list.append(add_suffix_to_dict(lrs_data,'_lrs'))
            data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
            iplots.updateline(data)

            ### performing sweep ###################################################################################
            if sweep:
                if two_sweeps:
                    dates_dict = defaultdict(list)
                    vlist1 = tri(v1 = v1, v2 = 0, step = step)
                    vlist2 = tri(v1 = 0, v2 = v2, step = step2)
                    k.iv(vlist1, measure_range = range_sweep, i_limit = limitI) 
                    while not k.done():
                        plt.pause(0.1)
                    sweep_data = k.get_data()
                    k.iv(vlist2, measure_range = range_sweep2, i_limit = limitI2) 
                    while not k.done():
                        plt.pause(0.1)
                    data_2nd_sweep = k.get_data()
                    for key in data_2nd_sweep:
                        data_to_append = data_2nd_sweep[key]
                        if not isinstance(data_to_append,dict) and not isinstance(data_to_append, str):
                            sweep_data[key] = np.append(sweep_data[key], data_to_append)
                else:  
                    k.iv(vlist, measure_range = range_sweep) 
                    while not k.done():
                        plt.pause(0.1)
                    k.source_output(ch = 'A', state = False)
                    k.source_output(ch = 'B', state = False)
                    sweep_data = k.get_data()
                sweep_list.append(add_suffix_to_dict(sweep_data,'_sweep'))
                data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
                iplots.updateline(data)
            if r_window:
                current_compliance = limitI2
                window_hit = False
                u=0
                d=0
                while not window_hit:
                    
                    k.source_output(chan = 'A', state = True)
                    k.source_level(source_val= V_read, source_func='v', ch='A')

                    plt.pause(1)
                    k._it_lua(sourceVA = V_read, sourceVB = 0, points = 5, interval = 0.1, measure_range = range_lrs, limitI = 1, nplc = nplc)
                    while not k.done():
                        plt.pause(0.1)
                    k.source_output(ch = 'A', state = False)
                    k.source_output(ch = 'B', state = False)
                    r_data = k.get_data()
                    resistance = np.mean(r_data['V']/r_data['I']) - 50
                    print('Compliance = ' + str(current_compliance))
                    print('Resistance = ' + str(resistance))

                    if resistance >= r_lower and resistance <= r_upper:
                        window_hit = True
                        break
                    elif resistance < r_lower:
                        current_compliance -= cc_step
                        u = 0
                        d += 1
                    elif resistance > 3.5e4 or u >=50:
                        vlist2 = tri(v1 = 0, v2 = -2, step = step2)
                        current_compliance = 2e-3
                    elif d >= 50:
                        vlist1 = tri(v1 = 2, v2 = 0, step = step)
                    else:
                        current_compliance += cc_step
                        u += 1
                        d = 0

                    if current_compliance < cc_step:
                        current_compliance =cc_step

                    if u > 51 or d > 51:
                        print('Failed hitting resistance window, aborting measurement')
                        window_hit = True
                        abort = True
                        break

                    k.iv(vlist1, measure_range = range_sweep, Ilimit = limitI) 
                    while not k.done():
                        plt.pause(0.1)
                    
                    k.iv(vlist2, measure_range = range_sweep2, Ilimit = current_compliance) 
                    while not k.done():
                        plt.pause(0.1)
                    vlist1 = tri(v1 = v1, v2 = 0, step = step)
                    vlist2 = tri(v1 = 0, v2 = v2, step = step2)

                    if current_compliance > 1e-3:
                        current_compliance = limitI2
            k.source_output(ch = 'A', state = False)
            k.source_output(ch = 'B', state = False)
  
    data['attenuation'] = attenuation
    data['pulse_width'] = pulse_width*widthFactor
    data['scale'] = scale
    data['position'] = position
    data['trigger_level'] = trigger_level
    data['positive_amplitude'] = positive_amplitude
    data['negative_amplitude'] = negative_amplitude
    #data['set_pattern'] = set_pattern
    data['set_pattern_format'] = set_pattern_format
    data['set_pattern_length'] = set_pattern_length
    data['set_ppg_channel'] = set_ppg_channel
    data['attenuation'] = attenuation
    data['recordlength'] = recordlength
    data['V_read'] = V_read
    data['delay'] = delay
    data['number_of_pulses'] = num_of_pulses 


    

    data['nplc'] = nplc
    data['cycles'] = cycles


    datafolder = os.path.join('C:\Messdaten', samplename, padname)
    subfolder = datestr
    file_exits = True
    i=1
    filepath = os.path.join(datafolder, subfolder, str(int(pulse_width*1e12*widthFactor)) + 'ps_'+str(int(num_of_pulses )) + 'pulses_'+str(int(delay)) + 'ps_delay_' 
        +str(pos_amp) + 'v_' +str(int(attenuation)) + 'dB_'+str(i))
    file_link = Path(filepath + '.df')
    while file_link.is_file():
        i +=1
        filepath = os.path.join(datafolder, subfolder, str(int(pulse_width*1e12*widthFactor)) + 'ps_'+str(int(num_of_pulses )) + 'pulses_'+str(int(delay)) + 'ps_delay_' 
            +str(pos_amp) + 'v_' +str(int(attenuation)) + 'dB_'+str(i))
        file_link = Path(filepath + '.df')
    io.write_pandas_pickle(meta.attach(data), filepath)

    print("number of pulses:", num_of_pulses)
    print("total_duration in ns:", total_pulse_duration *1e9)
    print("delay width in ps:", delay)


    return data, abort



def PPG30_measurement(samplename,
padname,
v1,
v2,
positive_amplitude = 1000e-3,
negative_amplitude = 600e-3,
set_pattern_format = 'LU',
#set_pattern_length = 1,
set_pattern = [0,0,0],
set_trigger_soruce = 'IMM',
set_ppg_channel = 'BIP',
step = 0.01,
step2 = 0.01,
V_read = 0.2,
range_lrs = 1e-3,
range_hrs = 1e-4,
range_sweep = 1e-2,
range_sweep2 = 1e-3,
cycles = 1,
pulse_width = 50e-12,
attenuation = 0,
automatic_measurement = True,
pg5_measurement = True,
recordlength = 250,
trigger_position = 25,
edge = 'r',
sweep = True,
two_sweeps = False,
scale = 0.12,
position = -3,
trigger_level = 0.05,
nplc = 10,
limitI = 3e-4,
limitI2 = 3e-4,
r_window = False,
r_lower = 1e3,
r_upper = 2e3,
cc_step = 25e-6):

    setup_vcm_plots()
    data = {}
    data['padname'] = padname
    data['samplename'] = samplename

    data['positive_amplitude'] = positive_amplitude
    data['negative_amplitude'] = negative_amplitude

    data['set_pattern'] = set_pattern
    data['set_pattern_format'] = set_pattern_format
    #data['set_pattern_length'] = set_pattern_length
    data['set_ppg_channel'] = set_ppg_channel

    data['pulse_width'] = pulse_width
    data['attenuation'] = attenuation
    data['recordlength'] = recordlength
    data['V_read'] = V_read
    

    data['nplc'] = nplc
    data['cycles'] = cycles
    ###### Setting PPG30 ###############################################################

    sympulsPG30.amplitude1(Amp1 = positive_amplitude)
    sympulsPG30.amplitude2(Amp2 = negative_amplitude)
    sympulsPG30.pattern(pattern = set_pattern_format)
    sympulsPG30.format(form = set_ppg_channel)
    sympulsPG30.set_lupattern(pattern = set_pattern)
    #sympulsPG30.set_lupattern_length(num_words = set_pattern_length)
    sympulsPG30.trigger_source(trig = set_trigger_soruce)




    hrs_list = []
    lrs_list = []
    sweep_list = []
    scope_list = []
    
    vlist = tri(v1 = v1, v2 = v2, step = step)

    abort = False
    for i in range(cycles):
        if not abort:
            ### Reading HRS resistance ############################################################################
            k.source_output(ch = 'A', state = True)
            k.source_level(source_val= V_read, source_func='v', ch='A')

            plt.pause(1)

            k._it_lua(sourceVA = V_read, sourceVB = 0, points =10, interval = 0.1, rangeI = range_hrs , limitI = 1, nplc = nplc)
            while not k.done():
                plt.pause(0.1)
            k.source_output(ch = 'A', state = False)
            k.source_output(ch = 'B', state = False)
            hrs_data = k.get_data()
            hrs_list.append(add_suffix_to_dict(hrs_data,'_hrs'))
            data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
            iplots.updateline(data)
            ### Setting up scope  ################################################################################

            ttx.inputstate(3, True)
            ttx.inputstate(2, False)
            ttx.inputstate(1, False)
            ttx.inputstate(4, False)

            ttx.scale(3, scale)
            ttx.position(3, position)


            ttx.change_samplerate_and_recordlength(samplerate = 100e9, recordlength= recordlength)
            if pulse_width < 100e-12:
                ttx.trigger_position(40)
            elif pulse_width < 150e-12:
                ttx.trigger_position(30)
            else:
                ttx.trigger_position(trigger_position)

            plt.pause(0.1)

            ttx.arm(source = 3, level = trigger_level, edge = edge)


            ### Applying pulse and reading scope data #############################################################
            if pg5_measurement:
                sympulsPG30.set_pulse_width(pulse_width)
            if not automatic_measurement:
                input('Connect the RF probes and press enter')
                plt.pause(0.5)
            else:
                plt.pause(1)
                
            if pg5_measurement:
                sympulsPG30.trigger()
            else:
                print('Apply pulse')
            plt.pause(0.1)
            plt.pause(0.2)
            status = ttx.triggerstate()
            while status == True:
                plt.pause(0.1)
                status = ttx.triggerstate()
            plt.pause(0.5)
            scope_list.append(ttx.get_curve(3))
            data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
            iplots.updateline(data)

            ### Reading LRS resistance #############################################################################

            if not automatic_measurement:
                input('Connect the DC probes and press enter')

            k.source_output(ch = 'A', state = True)
            k.source_level(source_val= V_read, source_func='v', ch='A')

            plt.pause(1)
            k._it_lua(sourceVA = V_read, sourceVB = 0, points = 10, interval = 0.1, rangeI = range_lrs, limitI = 1, nplc = nplc)
            while not k.done():
                plt.pause(0.1)
            k.source_output(ch = 'A', state = False)
            k.source_output(ch = 'B', state = False)
            lrs_data = k.get_data()
            lrs_list.append(add_suffix_to_dict(lrs_data,'_lrs'))
            data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
            iplots.updateline(data)

            ### performing sweep ###################################################################################
            if sweep:
                if two_sweeps:
                    dates_dict = defaultdict(list)
                    vlist1 = tri(v1 = v1, v2 = 0, step = step)
                    vlist2 = tri(v1 = 0, v2 = v2, step = step2)
                    k.iv(vlist1, measure_range = range_sweep, i_limit = limitI) 
                    while not k.done():
                        plt.pause(0.1)
                    sweep_data = k.get_data()
                    k.iv(vlist2, measure_range = range_sweep2, i_limit = limitI2) 
                    while not k.done():
                        plt.pause(0.1)
                    data_2nd_sweep = k.get_data()
                    for key in data_2nd_sweep:
                        data_to_append = data_2nd_sweep[key]
                        if not isinstance(data_to_append,dict) and not isinstance(data_to_append, str):
                            sweep_data[key] = np.append(sweep_data[key], data_to_append)
                else:  
                    k.iv(vlist, measure_range = range_sweep) 
                    while not k.done():
                        plt.pause(0.1)
                    k.source_output(ch = 'A', state = False)
                    k.source_output(ch = 'B', state = False)
                    sweep_data = k.get_data()
                sweep_list.append(add_suffix_to_dict(sweep_data,'_sweep'))
                data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
                iplots.updateline(data)
            if r_window:
                current_compliance = limitI2
                window_hit = False
                u=0
                d=0
                while not window_hit:
                    
                    k.source_output(chan = 'A', state = True)
                    k.source_level(source_val= V_read, source_func='v', ch='A')

                    plt.pause(1)
                    k._it_lua(sourceVA = V_read, sourceVB = 0, points = 5, interval = 0.1, measure_range = range_lrs, limitI = 1, nplc = nplc)
                    while not k.done():
                        plt.pause(0.1)
                    k.source_output(ch = 'A', state = False)
                    k.source_output(ch = 'B', state = False)
                    r_data = k.get_data()
                    resistance = np.mean(r_data['V']/r_data['I']) - 50
                    print('Compliance = ' + str(current_compliance))
                    print('Resistance = ' + str(resistance))

                    if resistance >= r_lower and resistance <= r_upper:
                        window_hit = True
                        break
                    elif resistance < r_lower:
                        current_compliance -= cc_step
                        u = 0
                        d += 1
                    elif resistance > 3.5e4 or u >=50:
                        vlist2 = tri(v1 = 0, v2 = -2, step = step2)
                        current_compliance = 2e-3
                    elif d >= 50:
                        vlist1 = tri(v1 = 2, v2 = 0, step = step)
                    else:
                        current_compliance += cc_step
                        u += 1
                        d = 0

                    if current_compliance < cc_step:
                        current_compliance =cc_step

                    if u > 51 or d > 51:
                        print('Failed hitting resistance window, aborting measurement')
                        window_hit = True
                        abort = True
                        break

                    k.iv(vlist1, measure_range = range_sweep, Ilimit = limitI) 
                    while not k.done():
                        plt.pause(0.1)
                    
                    k.iv(vlist2, measure_range = range_sweep2, Ilimit = current_compliance) 
                    while not k.done():
                        plt.pause(0.1)
                    vlist1 = tri(v1 = v1, v2 = 0, step = step)
                    vlist2 = tri(v1 = 0, v2 = v2, step = step2)

                    if current_compliance > 1e-3:
                        current_compliance = limitI2
            k.source_output(ch = 'A', state = False)
            k.source_output(ch = 'B', state = False)
  
    data['attenuation'] = attenuation
    data['pulse_width'] = pulse_width
    data['scale'] = scale
    data['position'] = position
    data['trigger_level'] = trigger_level

    datafolder = os.path.join('C:\Messdaten', samplename, padname)
    subfolder = datestr
    file_exits = True
    i=1
    filepath = os.path.join(datafolder, subfolder, str(int(pulse_width*1e12)) + 'ps_'+str(int(attenuation)) + 'dB_'+str(i))
    file_link = Path(filepath + '.df')
    while file_link.is_file():
        i +=1
        filepath = os.path.join(datafolder, subfolder, str(int(pulse_width*1e12)) + 'ps_'+str(int(attenuation)) + 'dB_'+str(i))
        file_link = Path(filepath + '.df')
    io.write_pandas_pickle(meta.attach(data), filepath)

    return data, abort


def PPG30_pick_off(samplename,
padname,
v1,
v2,
positive_amplitude = 1000e-3,
negative_amplitude = 600e-3,
set_pattern_format = 'LU',  # Since only LU pattern is user giver pattern
set_pattern_length = 1,     # Set word lenght. But since it is autmatically (line 329 in sympuls.PG30) selected so we dont have to use this. 
widthFactor = 1,            # (Daniel way of word function, line 2532 in this file). use in a function to give 1s, number of pulses and delays.
num_delayFactor = 0,        # Part of function to give 1s, number of pulses and delays
num_of_pulses = 0,          # Part of function to give 1s, number of pulses and delays
set_pattern = pattern_list, # give pattern in the form of list. [1,0,-1]
set_trigger_soruce = 'IMM',
set_ppg_channel = 'BIP',
step = 0.01,
step2 = 0.01,
V_read = 0.2,
range_lrs = 1e-3,
range_hrs = 1e-4,
range_sweep = 1e-2,
range_sweep2 = 1e-3,
cycles = 1,
pulse_width = 50e-12 ,
attenuation = 0,
automatic_measurement = True,
pg5_measurement = True,
recordlength = 250,
trigger_position = 25,
edge = 'r',
sweep = True,
two_sweeps = False,
scale = 0.12,
position = -3,
trigger_level = 0.05,
nplc = 10,
limitI = 3e-4,
limitI2 = 3e-4,
r_window = False,
r_lower = 1e3,
r_upper = 2e3,
cc_step = 25e-6):

    setup_vcm_pick_off_plots()
    data = {}
    data['padname'] = padname
    data['samplename'] = samplename

    data['positive_amplitude'] = positive_amplitude
    data['negative_amplitude'] = negative_amplitude

    ##### Daniel way of setting pattern ####
    set_pattern  = pattern_list(widthFactor= widthFactor, num_delayFactor = num_delayFactor, num_of_pulses = num_of_pulses )
    #print(len(set_pattern))

    #data['set_pattern'] = set_pattern
    #data['set_pattern_format'] = set_pattern_format
    #data['set_pattern_length'] = set_pattern_length
    #data['set_ppg_channel'] = set_ppg_channel

    data['pulse_width'] = pulse_width
    #data['attenuation'] = attenuation
    #data['recordlength'] = recordlength
    #data['V_read'] = V_read
    

    #data['nplc'] = nplc
    #data['cycles'] = cycles

    ###### Setting PPG30 ###############################################################

    sympulsPG30.amplitude1(Amp1 = positive_amplitude)
    sympulsPG30.amplitude2(Amp2 = negative_amplitude)
    sympulsPG30.pattern(pattern = set_pattern_format)
    sympulsPG30.format(form = set_ppg_channel)

    sympulsPG30.set_lupattern(pattern = set_pattern)
    #set_length = np.ceil(len(set_pattern)/128)                            # setting number of words on PPG30 dividing by digits
    #sympulsPG30.set_lupattern_length(num_words = set_pattern_length)      # uncomment if we have to use user given word.
    sympulsPG30.trigger_source(trig = set_trigger_soruce)

    ##delay_factor, number_of_positive_pulses, number_of_negative_pulses =find_delays(set_pattern)
    ##delay = delay_factor * pulse_width *1e12                               # find delays by multiplying pulsewidth with delays factor

    delay = num_delayFactor * pulse_width *1e12                              # find delays by multiplying pulsewidth with delays factor

    ##Number_of_pulses = number_of_positive_pulses + number_of_negative_pulses
    total_pulse_duration = ((pulse_width *widthFactor)+(num_delayFactor*pulse_width))*num_of_pulses       # count  pulses with delays

    pos_amp_deb = round(deb_to_atten(attenuation)*2*positive_amplitude,2)  # as PPG30 have variable amplitude so we have to calculate actual amplitude
    pos_amp = str(pos_amp_deb).replace('.', 'p')



    hrs_list = []
    lrs_list = []
    sweep_list = []
    scope_list = []
    scope_app_list = []
    
    vlist = tri(v1 = v1, v2 = v2, step = step)

    abort = False
    for i in range(cycles):
        if not abort:
            ### Reading HRS resistance ############################################################################
            k.source_output(ch = 'A', state = True)
            k.source_level(source_val= V_read, source_func='v', ch='A')

            plt.pause(1)

            k._it_lua(sourceVA = V_read, sourceVB = 0, points =10, interval = 0.1, rangeI = range_hrs , limitI = 1, nplc = nplc)
            while not k.done():
                plt.pause(0.1)
            k.source_output(ch = 'A', state = False)
            k.source_output(ch = 'B', state = False)
            hrs_data = k.get_data()
            hrs_list.append(add_suffix_to_dict(hrs_data,'_hrs'))
            data = combine_2_scope_lists_to_data_frame(hrs_list, lrs_list, scope_list, scope_app_list, sweep_list)
            iplots.updateline(data)

            ######################## Setting up scope  ################################################################################

            ttx.inputstate(3, True)
            ttx.inputstate(2, False)
            ttx.inputstate(1, True)
            ttx.inputstate(4, False)

            ttx.scale(1, scale)
            ttx.position(1, position)


            ttx.change_samplerate_and_recordlength(samplerate = 100e9, recordlength= recordlength)
            ttx.trigger_position(trigger_position)

            plt.pause(0.1)

            ttx.arm(source = 1, level = trigger_level, edge = edge)


            ################################ Applying pulse and reading scope data #############################################################
            if pg5_measurement:
                sympulsPG30.set_pulse_width(pulse_width)
            if not automatic_measurement:
                input('Connect the RF probes and press enter')
                plt.pause(0.5)
            else:
                plt.pause(1)
                
            if pg5_measurement:
                sympulsPG30.trigger()
            else:
                print('Apply pulse')
            plt.pause(0.1)
      
            plt.pause(0.2)
            status = ttx.triggerstate()

#############################################

            while ttx.triggerstate() == True:
                plt.pause(0.1)
                """if time.time() > timeout:
                                                                    ttx.disarm()
                                                                    timeout_bool = True
                                                                    break"""
                            
            plt.pause(0.5)
        ### data recorded from DUT, as a effective voltage (V_eff)
            ttx_eff = ttx.get_curve(3)

            ttx_eff['t_ttx'] = ttx_eff.pop('t_ttx')
            ttx_eff['V_ttx'] = ttx_eff.pop('V_ttx')

        ### data recorded directly from PG to OSCI (V_app)

            ttx_app = ttx.get_curve(1)

            ttx_app['t_ttx_app'] = ttx_app.pop('t_ttx')
            ttx_app['V_ttx_app'] = ttx_app.pop('V_ttx')

            scope_list.append(ttx_eff)
            scope_app_list.append(ttx_app)

            data = combine_2_scope_lists_to_data_frame(hrs_list, lrs_list, scope_list, scope_app_list, sweep_list)
            iplots.updateline(data)

            ### Reading LRS resistance #############################################################################

            if not automatic_measurement:
                input('Connect the DC probes and press enter')

            k.source_output(ch = 'A', state = True)
            k.source_level(source_val= V_read, source_func='v', ch='A')

            plt.pause(1)
            k._it_lua(sourceVA = V_read, sourceVB = 0, points = 10, interval = 0.1, rangeI = range_lrs, limitI = 1, nplc = nplc)
            while not k.done():
                plt.pause(0.1)
            k.source_output(ch = 'A', state = False)
            k.source_output(ch = 'B', state = False)
            lrs_data = k.get_data()
            lrs_list.append(add_suffix_to_dict(lrs_data,'_lrs'))
            data = combine_2_scope_lists_to_data_frame(hrs_list, lrs_list, scope_list, scope_app_list, sweep_list)
            iplots.updateline(data)

            ### performing sweep ###################################################################################
            if sweep:
                if two_sweeps:
                    dates_dict = defaultdict(list)
                    vlist1 = tri(v1 = v1, v2 = 0, step = step)
                    vlist2 = tri(v1 = 0, v2 = v2, step = step2)
                    k.iv(vlist1, measure_range = range_sweep, i_limit = limitI) 
                    while not k.done():
                        plt.pause(0.1)
                    sweep_data = k.get_data()
                    k.iv(vlist2, measure_range = range_sweep2, i_limit = limitI2) 
                    while not k.done():
                        plt.pause(0.1)
                    data_2nd_sweep = k.get_data()
                    for key in data_2nd_sweep:
                        data_to_append = data_2nd_sweep[key]
                        if not isinstance(data_to_append,dict) and not isinstance(data_to_append, str):
                            sweep_data[key] = np.append(sweep_data[key], data_to_append)
                else:  
                    k.iv(vlist, measure_range = range_sweep) 
                    while not k.done():
                        plt.pause(0.1)
                    k.source_output(ch = 'A', state = False)
                    k.source_output(ch = 'B', state = False)
                    sweep_data = k.get_data()
                sweep_list.append(add_suffix_to_dict(sweep_data,'_sweep'))
                data = combine_2_scope_lists_to_data_frame(hrs_list, lrs_list, scope_list, scope_app_list, sweep_list)
                iplots.updateline(data)
            if r_window:
                current_compliance = limitI2
                window_hit = False
                u=0
                d=0
                while not window_hit:
                    
                    k.source_output(chan = 'A', state = True)
                    k.source_level(source_val= V_read, source_func='v', ch='A')

                    plt.pause(1)
                    k._it_lua(sourceVA = V_read, sourceVB = 0, points = 5, interval = 0.1, measure_range = range_lrs, limitI = 1, nplc = nplc)
                    while not k.done():
                        plt.pause(0.1)
                    k.source_output(ch = 'A', state = False)
                    k.source_output(ch = 'B', state = False)
                    r_data = k.get_data()
                    resistance = np.mean(r_data['V']/r_data['I']) - 50
                    print('Compliance = ' + str(current_compliance))
                    print('Resistance = ' + str(resistance))

                    if resistance >= r_lower and resistance <= r_upper:
                        window_hit = True
                        break
                    elif resistance < r_lower:
                        current_compliance -= cc_step
                        u = 0
                        d += 1
                    elif resistance > 3.5e4 or u >=50:
                        vlist2 = tri(v1 = 0, v2 = -2, step = step2)
                        current_compliance = 2e-3
                    elif d >= 50:
                        vlist1 = tri(v1 = 2, v2 = 0, step = step)
                    else:
                        current_compliance += cc_step
                        u += 1
                        d = 0

                    if current_compliance < cc_step:
                        current_compliance =cc_step

                    if u > 51 or d > 51:
                        print('Failed hitting resistance window, aborting measurement')
                        window_hit = True
                        abort = True
                        break

                    k.iv(vlist1, measure_range = range_sweep, Ilimit = limitI) 
                    while not k.done():
                        plt.pause(0.1)
                    
                    k.iv(vlist2, measure_range = range_sweep2, Ilimit = current_compliance) 
                    while not k.done():
                        plt.pause(0.1)
                    vlist1 = tri(v1 = v1, v2 = 0, step = step)
                    vlist2 = tri(v1 = 0, v2 = v2, step = step2)

                    if current_compliance > 1e-3:
                        current_compliance = limitI2
            k.source_output(ch = 'A', state = False)
            k.source_output(ch = 'B', state = False)
  
    data['attenuation'] = attenuation
    data['pulse_width'] = pulse_width*widthFactor
    data['scale'] = scale
    data['position'] = position
    data['trigger_level'] = trigger_level
    data['positive_amplitude'] = positive_amplitude
    data['negative_amplitude'] = negative_amplitude
    #data['set_pattern'] = set_pattern
    data['set_pattern_format'] = set_pattern_format
    data['set_pattern_length'] = set_pattern_length
    data['set_ppg_channel'] = set_ppg_channel
    data['attenuation'] = attenuation
    data['recordlength'] = recordlength
    data['V_read'] = V_read
    data['delay'] = delay
    data['number_of_pulses'] = num_of_pulses 


    

    data['nplc'] = nplc
    data['cycles'] = cycles


    datafolder = os.path.join('C:\Messdaten', samplename, padname)
    subfolder = datestr
    file_exits = True
    i=1
    filepath = os.path.join(datafolder, subfolder, str(int(pulse_width*1e12*widthFactor)) + 'ps_'+str(int(num_of_pulses )) + 'pulses_'+str(int(delay)) + 'ps_delay_' 
        +str(pos_amp) + 'v_' +str(int(attenuation)) + 'dB_'+str(i))
    file_link = Path(filepath + '.df')
    while file_link.is_file():
        i +=1
        filepath = os.path.join(datafolder, subfolder, str(int(pulse_width*1e12*widthFactor)) + 'ps_'+str(int(num_of_pulses )) + 'pulses_'+str(int(delay)) + 'ps_delay_' 
            +str(pos_amp) + 'v_' +str(int(attenuation)) + 'dB_'+str(i))
        file_link = Path(filepath + '.df')
    io.write_pandas_pickle(meta.attach(data), filepath)

    print("number of pulses:", num_of_pulses)
    print("total_duration in ns:", total_pulse_duration *1e9)
    print("delay width in ps:", delay)


    return data, abort


def PPG30_pick_off_test(
    samplename: str,
    padname: str,
    # --- Sweep Voltage Configuration ---
    v1: float,
    v2: float,
    # --- Pulse Parameters ---
    positive_amplitude: float = 1.0,
    negative_amplitude: float = 0.6,
    pulse_width: float = 50e-12,
    set_pattern: list = None,
    # --- PPG30 Configuration ---
    set_pattern_format: str = 'LU',
    set_ppg_channel: str = 'BIP',
    set_trigger_soruce: str = 'IMM',
    attenuation: float = 0,
    # --- Keithley (SMU) Configuration ---
    V_read: float = 0.2,
    range_lrs: float = 1e-3,
    range_hrs: float = 1e-4,
    nplc: int = 10,
    # --- Oscilloscope Configuration ---
    recordlength: int = 250,
    trigger_position: int = 25,
    trigger_level: float = 0.05,
    scale: float = 0.12,
    position: int = -3,
    edge: str = 'r',
    # --- Measurement Flow Control ---
    cycles: int = 1,
    automatic_measurement: bool = True,
    pg5_measurement: bool = True,
    # --- Sweep Configuration ---
    sweep: bool = True,
    step: float = 0.01,
    limitI: float = 3e-4,
    two_sweeps: bool = False,
    step2: float = 0.01,
    range_sweep: float = 1e-2,
    range_sweep2: float = 1e-3,
    limitI2: float = 3e-4,
):
    """
    Performs a pick-off test using a Pulse Pattern Generator (PPG30), an
    oscilloscope (ttx), and a Source-Measure Unit (k).

    The sequence is:
    1. Measure initial High-Resistance State (HRS).
    2. Apply a voltage pulse from the PPG.
    3. Capture the pulse waveform on the oscilloscope.
    4. Measure the final Low-Resistance State (LRS).
    5. Optionally, perform a follow-up I-V sweep.
    This process is repeated for the specified number of cycles.
    """
    # Initialize plots for live visualization
    setup_vcm_pick_off_plots()

    # --- 1. Initial Setup and Parameter Storage ---
    # Store all measurement parameters in a dictionary for saving later.
    data = {
        'samplename': samplename,
        'padname': padname,
        'positive_amplitude': positive_amplitude,
        'negative_amplitude': negative_amplitude,
        'pulse_width': pulse_width,
        'set_pattern': set_pattern,
        'set_pattern_format': set_pattern_format,
        'set_ppg_channel': set_ppg_channel,
        'attenuation': attenuation,
        'V_read': V_read,
        'nplc': nplc,
        'cycles': cycles,
        'recordlength': recordlength,
        'scale': scale,
        'position': position,
        'trigger_level': trigger_level
    }

    # Initialize lists to store data from each cycle
    hrs_list = []
    lrs_list = []
    sweep_list = []
    scope_list = []
    scope_app_list = []

    # Configure the Pulse Pattern Generator (PPG)
    sympulsPG30.amplitude1(Amp1=positive_amplitude)
    sympulsPG30.amplitude2(Amp2=negative_amplitude)
    sympulsPG30.pattern(pattern=set_pattern_format)
    sympulsPG30.format(form=set_ppg_channel)
    sympulsPG30.set_lupattern(pattern=set_pattern)
    sympulsPG30.trigger_source(trig=set_trigger_soruce)
    
    vlist = tri(v1=v1, v2=v2, step=step)
    abort = False

    # --- 2. Main Measurement Loop ---
    for i in range(cycles):
        if abort:
            break

        # --- 2a. Measure High-Resistance State (HRS) ---
        RFswitches.write('X') # open RF2 switch
        plt.pause(1)

        k.source_output(ch='A', state=True)
        k.source_level(source_val=V_read, source_func='v', ch='A')
        plt.pause(1)  # Pause for SMU to settle
        k._it_lua(sourceVA=V_read, sourceVB=0, points=10, interval=0.1, rangeI=range_hrs, limitI=1, nplc=nplc)
        while not k.done():
            plt.pause(0.1)
        k.source_output(ch='A', state=False)
        hrs_data = k.get_data()
        hrs_list.append(add_suffix_to_dict(hrs_data, '_hrs'))
        
        # Update live plots with new data
        data = combine_2_scope_lists_to_data_frame(hrs_list, lrs_list, scope_list, scope_app_list, sweep_list)
        iplots.updateline(data)
        #RFswitches.write('X') # close RF2 switch

        # --- 2b. Configure Oscilloscope for Pulse Capture ---
        # Corrected to use positional arguments: (channel, state)
        ttx.inputstate(3, True)   # DUT signal
        ttx.inputstate(1, True)   # Applied pulse signal
        ttx.inputstate(2, False)
        ttx.inputstate(4, False)

        ttx.scale(1, scale)
        ttx.position(1, position)
        ttx.change_samplerate_and_recordlength(samplerate=100e9, recordlength=recordlength)
        ttx.trigger_position(trigger_position)
        plt.pause(0.1)
        ttx.arm(source=1, level=trigger_level, edge=edge)

        # --- 2c. Apply Pulse and Acquire Waveform ---
        if pg5_measurement:
            sympulsPG30.set_pulse_width(pulse_width)

        if not automatic_measurement:
            input('Connect the RF probes and press Enter to apply the pulse.')
        else:
            plt.pause(1)

        # Trigger the pulse and wait for the oscilloscope to capture it
        RFswitches.write('X') # open RF2 switch
        RFswitches.a_on() # open RF1 
        plt.pause(0.5)
        sympulsPG30.trigger() if pg5_measurement else print('Apply pulse manually.')
        plt.pause(0.5)
        RFswitches.a_off() # close RF1
        #RFswitches.write('X') # close RF2 switch

        while ttx.triggerstate():
            plt.pause(0.1)

        # Get captured waveform data from the oscilloscope
        ttx_eff = ttx.get_curve(3)  # Effective voltage on DUT
        ttx_eff['t_ttx'] = ttx_eff.pop('t_ttx')
        ttx_eff['V_ttx'] = ttx_eff.pop('V_ttx')

        ttx_app = ttx.get_curve(1)  # Applied voltage from PPG
        ttx_app['t_ttx_app'] = ttx_app.pop('t_ttx')
        ttx_app['V_ttx_app'] = ttx_app.pop('V_ttx')

        scope_list.append(ttx_eff)
        scope_app_list.append(ttx_app)

        # Update plots with scope data
        data = combine_2_scope_lists_to_data_frame(hrs_list, lrs_list, scope_list, scope_app_list, sweep_list)
        iplots.updateline(data)

        # --- 2d. Measure Low-Resistance State (LRS) ---
        if not automatic_measurement:
            input('Connect the DC probes and press Enter to measure LRS.')

        RFswitches.write('X') # open RF2 switch
        plt.pause(1)

        k.source_output(ch='A', state=True)
        k.source_level(source_val=V_read, source_func='v', ch='A')
        plt.pause(1)
        k._it_lua(sourceVA=V_read, sourceVB=0, points=10, interval=0.1, rangeI=range_lrs, limitI=1, nplc=nplc)
        while not k.done():
            plt.pause(0.1)
        k.source_output(ch='A', state=False)
        lrs_data = k.get_data()
        lrs_list.append(add_suffix_to_dict(lrs_data, '_lrs'))
        
        # Update plots with LRS data
        data = combine_2_scope_lists_to_data_frame(hrs_list, lrs_list, scope_list, scope_app_list, sweep_list)
        iplots.updateline(data)

        # --- 2e. (Optional) Perform Post-Pulse I-V Sweep ---
        if sweep:
            if two_sweeps:
                # Perform a two-part sweep (e.g., positive then negative)
                vlist1 = tri(v1=v1, v2=0, step=step)
                vlist2 = tri(v1=0, v2=v2, step=step2)
                
                k.iv(vlist1, measure_range=range_sweep, i_limit=limitI)
                while not k.done(): plt.pause(0.1)
                sweep_data_part1 = k.get_data()

                k.iv(vlist2, measure_range=range_sweep2, i_limit=limitI2)
                while not k.done(): plt.pause(0.1)
                sweep_data_part2 = k.get_data()

                # Combine the two sweep parts into a single dataset
                for key in sweep_data_part2:
                    if not isinstance(sweep_data_part2[key], (dict, str)):
                        sweep_data_part1[key] = np.append(sweep_data_part1[key], sweep_data_part2[key])
                sweep_data = sweep_data_part1
            else:
                # Perform a single continuous sweep
                k.iv(vlist, measure_range=range_sweep, i_limit=limitI)
                while not k.done():
                    plt.pause(0.1)
                sweep_data = k.get_data()
            
            k.source_output(ch='A', state=False)
            sweep_list.append(add_suffix_to_dict(sweep_data, '_sweep'))
            
            # Update plots with sweep data
            data = combine_2_scope_lists_to_data_frame(hrs_list, lrs_list, scope_list, scope_app_list, sweep_list)
            iplots.updateline(data)

    k.source_output(ch='A', state=False)
    k.source_output(ch='B', state=False)

    # --- 3. Save Data ---
    # Create a unique filename to avoid overwriting previous results.
    datafolder = os.path.join('C:\\Messdaten', samplename, padname)
    subfolder = datestr
    i = 1
    
    base_filename = f"{int(pulse_width*1e12)}ps_{int(attenuation)}dB"
    filepath = os.path.join(datafolder, subfolder, f"{base_filename}_{i}")
    
    while Path(filepath + '.df').is_file():
        i += 1
        filepath = os.path.join(datafolder, subfolder, f"{base_filename}_{i}")
    
    # Write the collected data to a pandas pickle file.
    io.write_pandas_pickle(meta.attach(data), filepath)

    RFswitches.all_off()       

    return data, abort

########################## ENDURANCE TESTING HELPER FUNCTIONS ##############################

def average_numerical_data(data_dict):
    """
    Averages all numerical values found within a dictionary.
    """
    numerical_values = []
    for key, value in data_dict.items():
        if isinstance(value, (int, float)):
            numerical_values.append(value)
        elif isinstance(value, np.ndarray):
            numerical_values.extend(value.flatten().tolist())  # Flatten and add array elements
        elif isinstance(value, list):
            numerical_values.extend([v for v in value if isinstance(v, (int, float))]) # Add list of numbers

    if numerical_values:
        return np.mean(numerical_values)
    else:
        return None

def average_resistance(data_dict):
    """
    Calculates the average resistance from 'Vmeasured' and 'I' arrays.
    """
    if 'Vmeasured' in data_dict and 'I' in data_dict:
        v_measured = data_dict['Vmeasured']
        current = data_dict['I']
        if isinstance(v_measured, np.ndarray) and isinstance(current, np.ndarray) and len(v_measured) == len(current):
            resistance_values = v_measured / current
            return np.mean(resistance_values)
    return None


def safe_updateline(d):
    if isinstance(d, dict):          # dict → DataFrame
        d = pd.DataFrame([d])
    iplots.updateline(d)


def PPG30_measurement_mod(samplename,
padname,
v1,
v2,
positive_amplitude = 1000e-3,
negative_amplitude = 600e-3,
set_pattern_format = 'LU',
pulse_block=[1],
delay_block=[],
repetitions=1,
final_block=[],
set_trigger_soruce = 'IMM',
set_ppg_channel = 'BIP',
step = 0.01,
step2 = 0.01,
V_read = 0.2,
range_lrs = 1e-3,
range_hrs = 1e-4,
range_sweep = 1e-2,
range_sweep2 = 1e-3,
cycles = 1,
pulse_width = 50e-12,
attenuation = 0,
automatic_measurement = True,
pg5_measurement = True,
recordlength = 250,
trigger_position = 25,
edge = 'r',
sweep = True,
two_sweeps = False,
scale = 0.12,
position = -3,
trigger_level = 0.05,
nplc = 10,
limitI = 3e-4,
limitI2 = 3e-4,
r_window = False,
r_lower = 1e3,
r_upper = 2e3,
cc_step = 25e-6):
    """
    Version of PPG30_measurement function modified for endurance testing purposes
    """
    
    # ----------------------- Setting the plots ----------------------------------

    setup_vcm_plots()

    # ----------------------- Create data dictionary -----------------------------

    data = {}
    data['samplename'] = samplename
    data['padname'] = padname
    data['positive_amplitude'] = positive_amplitude
    data['negative_amplitude'] = negative_amplitude
    data['pulse_block'] = pulse_block
    data['delay_block'] = delay_block
    data['repetitions'] = repetitions
    data['final_block'] = final_block
    data['set_pattern_format'] = set_pattern_format
    data['set_ppg_channel'] = set_ppg_channel
    data['pulse_width'] = pulse_width
    data['attenuation'] = attenuation
    data['recordlength'] = recordlength
    data['V_read'] = V_read 
    data['nplc'] = nplc
    data['cycles'] = cycles

    # ----------------------- Setting PPG30 -------------------------------------   

    sympulsPG30.amplitude1(Amp1 = positive_amplitude)
    sympulsPG30.amplitude2(Amp2 = negative_amplitude)
    sympulsPG30.pattern(pattern = set_pattern_format)
    sympulsPG30.format(form = set_ppg_channel)
    pattern_tuple = (pulse_block, delay_block, repetitions, final_block)         # here we compose all the variables into tuple, so that we can send it to new version of set_lupattern
    sympulsPG30.set_lupattern_new(pattern=pattern_tuple)                         # call the modified set_lupattern function    
    sympulsPG30.trigger_source(trig = set_trigger_soruce)

    # ----------------------- Setting lists for data storage --------------------

    hrs_list = []
    lrs_list = []
    avg_hrs_list = []
    avg_lrs_list = []
    sweep_list = []
    scope_list = []
    
    vlist = tri(v1 = v1, v2 = v2, step = step)

    processed_data = []                                                           # Initialize an empty list for processed data
    constant_params = {                                                           # Store the constant parameters once at the beginning
        'positive_amplitude': positive_amplitude,
        'negative_amplitude': negative_amplitude,
        'pulse_width': pulse_width
    }
    processed_data.append(constant_params)

    # ----------------------- Start of the actual measurement -------------------

    abort = False
    for i in range(cycles):
        if not abort:

            # ---------------- Reading HRS resistance ---------------------------

            k.source_output(ch = 'A', state = True)
            k.source_level(source_val= V_read, source_func='v', ch='A')

            plt.pause(1)

            k._it_lua(sourceVA = V_read, sourceVB = 0, points =10, interval = 0.1, rangeI = range_hrs , limitI = 1, nplc = nplc)
            while not k.done():
                plt.pause(0.1)
            k.source_output(ch = 'A', state = False)
            k.source_output(ch = 'B', state = False)
            hrs_data = k.get_data()
            hrs_list.append(add_suffix_to_dict(hrs_data,'_hrs'))

            avg_hrs_resistance = average_resistance(hrs_data)
            avg_hrs_list.append(avg_hrs_resistance)

            data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
            safe_updateline(data)   

            # ---------------- Setting up scope ---------------------------------

            ttx.inputstate(3, True)
            ttx.inputstate(2, False)
            ttx.inputstate(1, True)
            ttx.inputstate(4, False)

            ttx.scale(3, scale)
            ttx.position(3, position)


            ttx.change_samplerate_and_recordlength(samplerate = 100e9, recordlength= recordlength)
            ttx.trigger_position(trigger_position)

            plt.pause(0.1)

            ttx.arm(source = 3, level = trigger_level, edge = edge)

            # ---------------- Applying pulse and reading scope data ------------     

            if pg5_measurement:
                sympulsPG30.set_pulse_width(pulse_width)
            if not automatic_measurement:
                input('Connect the RF probes and press enter')
                plt.pause(0.5)
            else:
                plt.pause(1)
                
            if pg5_measurement:
                sympulsPG30.trigger()
            else:
                print('Apply pulse')
            plt.pause(0.1)
            plt.pause(0.2)
            status = ttx.triggerstate()
            while status == True:
                plt.pause(0.1)
                status = ttx.triggerstate()
            plt.pause(0.5)
            scope_list.append(ttx.get_curve(3))
            data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
            safe_updateline(data)   

            # ---------------- Reading LRS resistance --------------------------  

            if not automatic_measurement:
                input('Connect the DC probes and press enter')

            k.source_output(ch = 'A', state = True)
            k.source_level(source_val= V_read, source_func='v', ch='A')

            plt.pause(1)
            k._it_lua(sourceVA = V_read, sourceVB = 0, points = 10, interval = 0.1, rangeI = range_lrs, limitI = 1, nplc = nplc)
            while not k.done():
                plt.pause(0.1)
            k.source_output(ch = 'A', state = False)
            k.source_output(ch = 'B', state = False)
            lrs_data = k.get_data()
            lrs_list.append(add_suffix_to_dict(lrs_data,'_lrs'))
            
            avg_lrs_resistance = average_resistance(lrs_data)
            avg_lrs_list.append(avg_lrs_resistance)

            # Determine the 'switched' value
            switched_value = 0 # Default value
            if avg_lrs_resistance is not None:
                if avg_lrs_resistance < 3000:
                    switched_value = 1
                else:
                    switched_value = 0
            else:
                switched_value = None

            # Store the data for this cycle
            processed_data.append({
                'cycle': i + 1,
                'lrs': avg_lrs_resistance,
                'hrs': avg_hrs_resistance,
                'switched': switched_value
            })

            data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
            safe_updateline(data)   

            # ---------------- Performing sweep ------------------------------  
            if sweep:
                if two_sweeps:
                    dates_dict = defaultdict(list)
                    vlist1 = tri(v1 = v1, v2 = 0, step = step)
                    vlist2 = tri(v1 = 0, v2 = v2, step = step2)
                    k.iv(vlist1, measure_range = range_sweep, i_limit = limitI) 
                    while not k.done():
                        plt.pause(0.1)
                    sweep_data = k.get_data()
                    k.iv(vlist2, measure_range = range_sweep2, i_limit = limitI2) 
                    while not k.done():
                        plt.pause(0.1)
                    data_2nd_sweep = k.get_data()
                    for key in data_2nd_sweep:
                        data_to_append = data_2nd_sweep[key]
                        if not isinstance(data_to_append,dict) and not isinstance(data_to_append, str):
                            sweep_data[key] = np.append(sweep_data[key], data_to_append)
                else:  
                    k.iv(vlist, measure_range = range_sweep) 
                    while not k.done():
                        plt.pause(0.1)
                    k.source_output(ch = 'A', state = False)
                    k.source_output(ch = 'B', state = False)
                    sweep_data = k.get_data()
                sweep_list.append(add_suffix_to_dict(sweep_data,'_sweep'))
                data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
                safe_updateline(data)   
            k.source_output(ch = 'A', state = False)
            k.source_output(ch = 'B', state = False)
  
    data['attenuation'] = attenuation
    data['pulse_width'] = pulse_width
    data['scale'] = scale
    data['position'] = position
    data['trigger_level'] = trigger_level

    datafolder = os.path.join('C:\Messdaten', samplename, padname)
    subfolder = datestr
    file_exits = True
    i=1
    filepath = os.path.join(datafolder, subfolder, str(int(pulse_width*1e12)) + 'ps_'+str(int(attenuation)) + 'dB_'+str(i))
    file_link = Path(filepath + '.df')
    while file_link.is_file():
        i +=1
        filepath = os.path.join(datafolder, subfolder, str(int(pulse_width*1e12)) + 'ps_'+str(int(attenuation)) + 'dB_'+str(i))
        file_link = Path(filepath + '.df')
    io.write_pandas_pickle(meta.attach(data), filepath)

    # --- CSV Writing ---
    filepath_csv = filepath + '.csv' # Create a CSV filename based on the DataFrame filename

    # Extract headers from the first cycle's data and add constant parameters
    headers = ['cycle', 'lrs', 'hrs', 'switched', 'positive_amplitude', 'negative_amplitude', 'pulse_width']

    with open(filepath_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)

        writer.writeheader()
        writer.writerow(processed_data[0]) # Write the constant parameters

        # Write the data for each cycle
        for row_data in processed_data[1:]:
            writer.writerow(row_data)

    return avg_lrs_resistance, abort


def sweep_and_read(vset, vreset):
    """
    Perform SET/RESET sweeps and read resistance at V_read.
    Returns a finite resistance value (Ohms) or raises with a clear message.
    """
    V_read   = 0.2
    range_lrs = 1e-3
    nplc      = 10
    poll_dt   = 0.1
    timeout_s = 30.0  # safety timeout

    # Run sweeps
    kiv(tri(v1=vset,    step=0.02), measure_range=1e-3, i_limit=1e-3)
    kiv(tri(v1=-vreset, step=0.02), measure_range=1e-2, i_limit=1e-2)

    # Configure read
    time.sleep(0.1)
    k.source_output(ch='A', state=True)
    k.source_level(source_val=V_read, source_func='v', ch='A')

    # Start measurement
    k._it_lua(sourceVA=V_read, sourceVB=0, points=10, interval=0.1,
              rangeI=range_lrs, limitI=1, nplc=nplc)

    # Poll until done (do NOT toggle outputs inside the loop)
    t0 = time.time()
    while not k.done():
        if time.time() - t0 > timeout_s:
            # ensure we leave HW in a safe state
            k.source_output(ch='A', state=False)
            k.source_output(ch='B', state=False)
            raise TimeoutError("sweep_and_read: acquisition timed out")
        plt.pause(poll_dt)

    # Now it’s finished: turn outputs off once, then fetch data
    k.source_output(ch='A', state=False)
    k.source_output(ch='B', state=False)

    r_data = k.get_data()

    # ---- Robust resistance computation ----
    if not isinstance(r_data, dict) or 'V' not in r_data or 'I' not in r_data:
        raise RuntimeError("sweep_and_read: missing V/I in acquisition data")

    V = np.asarray(r_data['V'])
    I = np.asarray(r_data['I'])
    if V.size == 0 or I.size == 0:
        raise RuntimeError("sweep_and_read: empty V/I arrays")

    with np.errstate(divide='ignore', invalid='ignore'):
        R = V / I
    finite = np.isfinite(R)
    if not finite.any():
        raise RuntimeError("sweep_and_read: no finite resistance values (check I≈0)")

    resistance = float(np.mean(R[finite]) - 50.0)  # de-embed 50 Ω
    return resistance


def set_pattern_construction(package_size, ip_delay):
    """ 
    Function to create set pattern with packages
    """
    setpattern = ( [1] + [0]*ip_delay + [-1]*7 + [0]*ip_delay) * package_size
    return setpattern


def reset_pattern_construction(package_size, ip_delay):
    """ 
    Function to create reset pattern with packages
    """
    resetpattern = ( [-1]*7 + [0]*ip_delay + [1] + [0]*ip_delay ) * package_size
    return resetpattern


def _pn_pair(ip_delay: int):
    """ 
    Function to create set pattern without packages
    """
    return [1] + [0]*ip_delay + [-1]*7 + [0]*ip_delay


def _np_pair(ip_delay: int):
    """ 
    Function to create reset pattern without packages
    """
    return [-1]*7 + [0]*ip_delay + [1] + [0]*ip_delay


def calc_recordlength(n, ip_delay):
    """ 
    Function to calculate and set recordlength for oscilloscope 
    """
    # length = 1000 * (ip_delay/30) * 10**n
    length = 2000000 # hardcoded for testing
    return length


def call_ppg_with_params(pname, pulse_pattern, is_delay, repetitions, last_pulse, vset, vreset, record):
    """
    Function to fill PPG30_measurement_mod with testing parameters
    """
    print(f"--- Calling PPG30_measurement_mod, vset = {vset}, vreset = {vreset} ---")
    res2 = PPG30_measurement_mod(samplename ='Artem_RF33_023_',
    padname= pname,
    v1= 0,
    v2 = 0,
    positive_amplitude = vset,
    negative_amplitude = vreset,
    set_pattern_format = 'LU',
    pulse_block = pulse_pattern,
    delay_block = is_delay,
    repetitions = repetitions,
    final_block = last_pulse,
    pulse_width = 100e-12,
    set_trigger_soruce = 'IMM',
    set_ppg_channel = 'BIP',
    step = 0.02,
    step2 = 0.02,
    V_read = 0.2,
    range_lrs = 1e-3,
    range_hrs = 1e-2,
    range_sweep = 3e-4,
    range_sweep2 = 1e-2,
    cycles = 1,
    attenuation = 0,
    automatic_measurement = True,
    pg5_measurement = True,
    recordlength = record,
    trigger_position = 25,
    edge = 'r',
    two_sweeps = False,
    scale = 0.12,
    position = -1,
    trigger_level = 0.03,
    nplc = 1,
    limitI = 1e-3,
    limitI2 = 1e-2,)
    resistance = int(res2[0])
    return resistance

########################## LOGGING AND DATA SAVING UTILITIES ##############################

# --------------------- Catalogs ---------------------
BASE_DIR   = pathlib.Path(__file__).resolve().parent
LOG_DIR    = BASE_DIR / "logs"
DATA_DIR   = BASE_DIR / "data"
RUNS_DIR   = BASE_DIR / "runs"

for _p in (LOG_DIR, DATA_DIR, RUNS_DIR):
    _p.mkdir(exist_ok=True)

# --------------------- Logging parameters ---------------------
LOG_FILE = LOG_DIR / f"endurance_measurement{dt.datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(),                         # to console
        logging.FileHandler(LOG_FILE, encoding="utf-8")  # to file
    ]
)
logger = logging.getLogger("endurance measurement")
logging.getLogger("Plotter").setLevel(logging.ERROR)     # remove plot spam
logging.getLogger("visa_driver").setLevel(logging.INFO)  # remove debug spam

# --------------------- Filter string duplicates ---------------------
class DuplicateFilter(logging.Filter):
    last = None
    def filter(self, record):
        current = (record.name, record.levelno, record.getMessage())
        if current == self.last:        
            return False
        self.last = current
        return True

for h in logging.getLogger().handlers:
    h.addFilter(DuplicateFilter())

########################## MAIN ENDURANCE TESTING ALGORITHM ##############################

def endurance_testing(pname = 'R1_C5',
goal = 0,
ip_delay = 10,
package_size = 100,
is_delay = 400,
vset = 1.2,
vreset = 1.0,
correction_step = 0.025,
vsweepset = 1.2,
vsweepreset = 1.4,
cap_value = 7_500_000):
    """
    Perform endurance testing of a resistive device by repeatedly applying SET/RESET 
    pulse patterns until a defined total number of switching cycles is reached.

    The procedure includes:
    - Bootstrap stabilization: initial SET/RESET rounds with automatic 
      voltage adjustment to ensure the device can switch reliably.
    - Cycle-based accounting: one cycle is defined as a P→N transition 
      in the applied pulse sequence, counted analytically.
    - Exponential schedule: the number of package repetitions grows as 10^n 
      after each successful SET→RESET pair, accelerating the accumulation of cycles.
    - Cycle cap per round: each round is limited to at most "cap_value" cycles (default 7_500_000), 
      enforced by reducing repetitions if necessary.
    - Plateau mode: once the exponential schedule exceeds the cap, the maximum 
      legal round size is repeated until the total target is achieved.
    - Automatic correction: if SET or RESET does not reach the required state 
      (LRS/HRS), the respective voltage is incrementally increased by `correction_step`.

    Parameters:
        goal (int): Exponent for the total cycle target. 
                    The overall target is 10^goal × package_size cycles.
        ip_delay (int): Inter-pulse delay parameter.
        package_size (int): Number of cycles in one base pulse package.
        is_delay (int): Inter-package (inter-set) delay parameter.
        vset (float): Initial SET voltage.
        vreset (float): Initial RESET voltage.
        correction_step (float): Step size for adjusting voltages after failed switching.
        vsweepset (float): Voltage used for initial sweep in SET direction.
        vsweepreset (float): Voltage used for initial sweep in RESET direction.

    Example:
        goal=5, package_size=100  → target of 10,000,000 total cycles.

    Results:
        - Measurement data are logged at each round (timestamp, resistance, state, 
          applied voltages, repetition count, cycle totals).
        - Data are stored in a CSV file; run configuration is saved as JSON.
    """
    
    # ---------------- Constants and variables -----------------------------------------------------

    padname        = pname                                    # name of the device
    overall_cycles = 0                                        # running total of cycles (P→N), monotonic
    max_attempts   = 3                                        # maximal allowed sweep attempts before program break 
    attempts       = 0                                        # failed attempts counter
    set_failure    = 0                                        # failed sets counter
    max_set_fail   = 400                                      # maximal allowed set fails
    reset_failure  = 0                                        # failed resets counter
    max_reset_fail = 400                                      # maximal allowed reset fails
    n              = 0                                        # exponent for 10^n schedule (packages)
    failed         = False                                    # check for fail
    delay_block    = [0] * is_delay * 10                      # instrument delay block as list
    last_pulse     = [1]                                      # default last pulse
    records: List[Dict[str, Any]] = []                        # all measurements here
    cap_announced  = False                                    # log the plateau message only once
    ROUND_DIGITS   = 3                                        # mV roundings for adjusted values
    MAX_CYCLES_PER_ROUND = cap_value                          # max repetitions per round (each repetition contributes 'package_size' cycles)
    target_total_cycles = (10 ** int(goal)) * package_size    # TOTAL cycles target (overall, across all rounds)
    adjusted       = False                                    # Flag for adjusted values

    # ---------------- Safety measures --------------------------------------------------------------

    if package_size <= 0:
        raise ValueError("package_size must be positive.")
    max_repetitions = max(1, MAX_CYCLES_PER_ROUND // package_size) # 75_000
    if max_repetitions * package_size > MAX_CYCLES_PER_ROUND:
        max_repetitions -= 1
    if max_repetitions < 1:
        raise ValueError(
            f"package_size={package_size} exceeds instrument cap "
            f"({MAX_CYCLES_PER_ROUND}); decrease package_size."
        )

    # ---------------- Small helpers: count P→N cycles without expanding repetitions ----------------

    def _compress_events(seq):
        """
        Collapse contiguous non-zeros into 'P' or 'N' events.
        """
        ev = []
        for x in seq:
            if x == 0:
                continue
            s = 'P' if x > 0 else 'N'
            if not ev or ev[-1] != s:
                ev.append(s)
        return ev

    def _count_pn_in_events(ev):
        """
        Count P→N transitions inside a single event list.
        """
        return sum(1 for i in range(len(ev) - 1) if ev[i] == 'P' and ev[i + 1] == 'N')

    def count_pn_cycles(pulse_block, delay_block, repetitions, final_block):
        """
        Count P→N cycles for a tuple (pulse_block, delay_block, repetitions, final_block),
        without building the repeated sequence in Python.
        """
        ev_unit = _compress_events(pulse_block)
        cycles = 0

        if ev_unit:
            cycles_in_unit = _count_pn_in_events(ev_unit)
            cycles += repetitions * cycles_in_unit
            # P|N at boundaries between repeated units:
            if repetitions > 1 and ev_unit[-1] == 'P' and ev_unit[0] == 'N':
                cycles += (repetitions - 1)

        ev_final = _compress_events(final_block)
        if ev_unit and ev_final:
            # Boundary between the last unit and the final block
            if ev_unit[-1] == 'P' and ev_final[0] == 'N':
                cycles += 1
        if ev_final:
            cycles += _count_pn_in_events(ev_final)

        return cycles

    # ---------------- Log function config ----------------------------------------------------------

    cfg = {
        "padname": pname,
        "goal_exp": goal,
        "ip_delay": ip_delay,
        "package_size": package_size,
        "vset_start": vset,
        "vreset_start": vreset,
        "correction_step": correction_step,
        "vsweepset": vsweepset, 
        "vsweepreset": vsweepreset,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "max_cycles_per_round": MAX_CYCLES_PER_ROUND,
        "max_repetitions": int(max_repetitions),
        "target_total_cycles": int(target_total_cycles)
    }    
    run_hash = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    cfg_fname = RUNS_DIR / f"config_{cfg['timestamp'].replace(':','')}_{run_hash}.json"
    with cfg_fname.open("w", encoding="utf-8") as f_cfg:
        json.dump(cfg, f_cfg, indent=2, ensure_ascii=False)

    # ---------------- Start the endurance test -----------------------------------------------------

    logger.info("=== Endurance testing started ===")
    logger.info(
        f"Target total cycles = {target_total_cycles:,d}; "
        f"cap per round = {MAX_CYCLES_PER_ROUND:,d} "
        f"(max reps = {max_repetitions:d})"
    )
    logger.info("Config saved in %s", cfg_fname)

    # ---------------- Step 1: Initial sweep --------------------------------------------------------

    logger.info("--- Executing initial sweep ---")
    r = sweep_and_read(vsweepset, vsweepreset)
    logger.info("--- Measured r = %.0f Ω ---", r)

    # ---------------- Step 2: Start of the main loop -----------------------------------------------

    while True:
        if attempts >= max_attempts:
            logger.error("Reached max sweep attempts. Device not in HRS → STOP")
            break

        if r >= 5500:
            logger.info("Bootstrap rounds: P-N-P, N-P-N, (P-N)×10+P, (N-P)×10+N")
            recordlen_small = 1000

            def _log_round(phase: str, num_cycles: int, r_val: float, used_reps: int, capped: bool):
                """
                Consistent CSV row with cycle-based accounting.
                """
                records.append({
                    "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                    "cycle_exp": 0.0,                        # bootstrap marked as exp=0.0
                    "num_cycles": num_cycles,                # cycles this round
                    "phase": phase,
                    "state_after": ("LRS" if r_val <= 3500 else
                                    "HRS" if r_val >= 5500 else "MID"),
                    "resistance_ohm": r_val,
                    "vset": vset, 
                    "vreset": vreset,
                    "set_failure": set_failure,
                    "reset_failure": reset_failure,
                    "repetitions": used_reps,                # actual reps sent
                    "capped": bool(capped),                  # whether cap applied
                    "overall_cycles": overall_cycles         # monotonic total
                })

            def _run_pair(set_pat, set_last, reset_pat, reset_last):
                """
                Run SET until LRS, then RESET until HRS, with corrections.
                """
                nonlocal vset, vreset, set_failure, reset_failure, max_set_fail, max_reset_fail, overall_cycles, failed, r

                # ---- SET until LRS ----
                while True:
                    cycles_set = count_pn_cycles(set_pat, delay_block, 1, set_last)
                    logger.info("Bootstrap SET round: %d cycle(s)", cycles_set)
                    r = call_ppg_with_params(padname, set_pat, delay_block, 1, set_last, vset, vreset, recordlen_small)
                    overall_cycles += cycles_set
                    _log_round("SET", cycles_set, r, used_reps=1, capped=False)

                    if r > 40000:
                        if vreset > 0.6:
                            vreset = round(vreset - correction_step, ROUND_DIGITS)                            
                            logger.warning("High HRS (>40kΩ); vreset ↓ to %.3f", vreset)
                        else:
                            logger.warning("High HRS (>40kΩ); but vreset cant be lowered, capped at %.3f", vreset)

                    if r <= 3500:   # LRS reached
                        if r < 600:
                            if vset > 1:
                                vset = round(vset - correction_step, ROUND_DIGITS)
                                logger.warning("High LRS (<1kΩ) → vset ↓ to %.3f", vset)
                            else:
                                logger.warning("High LRS (<1kΩ), but vset cant be lowered, capped at %.3f", vset)
                        logger.info("Device in LRS")
                        break

                    if vset < 2:
                        vset = round(vset + correction_step, ROUND_DIGITS)
                        set_failure += 1
                        logger.warning("Bootstrap SET → not in LRS; vset ↑ to %.3f", vset)
                    else:
                        set_failure = 401
                        logger.warning("Bootstrap SET → not in LRS; max vset reached, vset %.3f", vset)
                    if set_failure >= max_set_fail:
                        logger.error("Bootstrap SET failed ≥ %d times → STOP", max_set_fail) 
                        failed = True
                        return

                # ---- RESET until HRS ----
                while r < 5500 and reset_failure < max_reset_fail:
                    cycles_reset = count_pn_cycles(reset_pat, delay_block, 1, reset_last)
                    logger.info("Bootstrap RESET round: %d cycle(s)", cycles_reset)
                    r = call_ppg_with_params(padname, reset_pat, delay_block, 1, reset_last, vset, vreset, recordlen_small)
                    overall_cycles += cycles_reset
                    _log_round("RESET", cycles_reset, r, used_reps=1, capped=False)
                    if r > 40000:
                        if vreset > 0.6:
                            vreset = round(vreset - correction_step, ROUND_DIGITS)                            
                            logger.warning("High HRS (>25kΩ); vreset ↓ to %.3f", vreset)
                        else:
                            logger.warning("High HRS (>25kΩ); but vreset cant be lowered, capped at %.3f", vreset)
                    if r < 5500:
                        if vreset < 1.2:
                            vreset = round(vreset + correction_step, ROUND_DIGITS)
                            reset_failure += 1
                            logger.warning("Bootstrap RESET → not back to HRS; vreset ↑ to %.3f", vreset)
                        else:
                            reset_failure = 401
                            logger.warning("Bootstrap RESET → not back to HRS; max vreset reached, vreset %.3f", vreset)
                        if reset_failure >= max_reset_fail:
                            logger.error("Bootstrap RESET failed ≥ %d times → STOP", max_reset_fail)
                            failed = True
                            return

            # ===================== MAIN SCHEDULE (with CAP & PLATEAU) =====================
            while not failed:

                adjusted = False

                # ---- stop when total cycles goal met ----
                if overall_cycles >= target_total_cycles:
                    logger.info("Total cycles goal reached %d → STOP", overall_cycles)
                    failed = True
                    break

                if reset_failure >= max_reset_fail:
                    logger.error("RESET failed ≥ %d times → STOP", max_reset_fail)
                    failed = True
                    break
                if set_failure >= max_set_fail:
                    logger.error("SET failed ≥ %d times → STOP", max_set_fail)
                    failed = True
                    break

                if n < 1:    
                    # 1) P-N-P ↔ N-P-N
                    if not failed:
                        _run_pair(_pn_pair(ip_delay), [1], _np_pair(ip_delay), [-1]*7)
                    if failed: break
                    # 2) (P-N)×10+P ↔ (N-P)×10+N
                    if not failed:
                        _run_pair(_pn_pair(ip_delay) * 10, [1], _np_pair(ip_delay) * 10, [-1]*7)
                    if failed: break

                # Desired repetitions for this exponent step
                desired_reps = int(10 ** n)
                # Apply cap (plateau if needed)
                capped = False
                if desired_reps > max_repetitions:
                    desired_reps = int(max_repetitions)
                    capped = True
                    if not cap_announced:
                        logger.info("Plateau mode: repetitions capped at %d (%d cycles/round).",
                                    desired_reps, desired_reps * package_size)
                        cap_announced = True

                # --- SET round ---
                pulse_pattern = set_pattern_construction(package_size, ip_delay)
                recordlen     = calc_recordlength(n, ip_delay)
                last_pulse    = [1]

                cycles_this_round = count_pn_cycles(pulse_pattern, delay_block, desired_reps, last_pulse)
                logger.info(
                    f"HRS → SET: {desired_reps:d} rep(s) × {package_size:d} cycles = {cycles_this_round:,d} cycle(s)"
                    f"{' [CAPPED]' if capped else ''}"
                )
                r = call_ppg_with_params(padname, pulse_pattern, delay_block, desired_reps, last_pulse, vset, vreset, recordlen)
                logger.info("Measured r = %.0f Ω", r)
                overall_cycles += cycles_this_round

                records.append({
                    "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                    "cycle_exp": float(n),
                    "num_cycles": int(cycles_this_round),
                    "phase": "SET",
                    "state_after": "LRS" if r <= 3500 else "HRS",
                    "resistance_ohm": float(r),
                    "vset": float(vset), "vreset": float(vreset),
                    "set_failure": int(set_failure), "reset_failure": int(reset_failure),
                    "repetitions": int(desired_reps),     
                    "capped": bool(capped),              
                    "overall_cycles": int(overall_cycles)
                })

                if 600 < r <= 3500:                    
                    vreset_saved = vreset # saving last vreset value for future usage
                    reset_failure = 0
                    # --- RESET round(s) until back to HRS ---
                    while r < 5500 and reset_failure < max_reset_fail:

                        pulse_pattern = reset_pattern_construction(package_size, ip_delay)
                        recordlen     = calc_recordlength(n, ip_delay)
                        last_pulse    = [-1] * 7

                        cycles_this_round = count_pn_cycles(pulse_pattern, delay_block, desired_reps, last_pulse)
                        logger.info(
                            f"LRS → RESET: {desired_reps:d} rep(s) × {package_size:d} cycles = {cycles_this_round:,d} cycle(s)"
                            f"{' [CAPPED]' if capped else ''}"
                        )
                        r = call_ppg_with_params(padname, pulse_pattern, delay_block, desired_reps, last_pulse, vset, vreset, recordlen)
                        logger.info("Measured r = %.0f Ω", r)
                        overall_cycles += cycles_this_round

                        records.append({
                            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                            "cycle_exp": float(n),
                            "num_cycles": int(cycles_this_round),
                            "phase": "RESET",
                            "state_after": "HRS" if r >= 5500 else "LRS",
                            "resistance_ohm": float(r),
                            "vset": float(vset), "vreset": float(vreset),
                            "set_failure": int(set_failure), "reset_failure": int(reset_failure),
                            "repetitions": int(desired_reps),  
                            "capped": bool(capped),            
                            "overall_cycles": int(overall_cycles)
                        })

                        if r < 5500:
                            n = 0.0  # restart exponent when RESET underperforms
                            adjusted = True
                            reset_failure += 1
                            if vreset < 1.2:
                                vreset = round(vreset + correction_step, ROUND_DIGITS)
                                logger.warning("Not back to HRS → vreset ↑ to %.3f; cycle_exp reset to 0.0", vreset)
                            else:
                                logger.warning("Not back to HRS, no increase of vreset possible, vreset capped at %.3f; cycle_exp reset to 0.0", vreset)                          
                                

                    if r >= 5500:    
                        # Successful full SET→RESET completed.
                        # Advance exponent ONLY if we are not on plateau. Otherwise stay and keep repeating.
                        if not capped:
                            n += 1
                            logger.info("Successful step; advance to exponent n=%s.", n)
                        else:
                            logger.info("Successful step at plateau; staying at capped repetitions=%d.", desired_reps)
                        if r > 40000 and vreset > 0.6:
                            n = 0.0  # restart exponent when RESET overperforms
                            vreset = round(vreset - correction_step, ROUND_DIGITS)
                            logger.warning("RESET too hard → vreset ↓ to %.3f", vreset)
                        if adjusted:
                            n = 0.0
                            logger.info("Vreset value was adjusted, back to the bootstrap with new value")

                    elif reset_failure >= max_reset_fail:
                        logger.error("RESET failed ≥ %d times → STOP", max_reset_fail)
                        failed = True
                    else:
                        logger.warning("RESET not back to HRS yet (fail #%d) → continuing with adjusted vreset", reset_failure)
                        continue

                else:
                    # SET didn’t reach LRS/LRS value too low — correct and retry from 10^0
                    if r < 600:
                        if vset > 1:
                            vset = round(vset - correction_step, ROUND_DIGITS)
                            n = 0.0
                            logger.warning("High LRS (<1kΩ) → vset ↓ to %.3f", vset)
                        else:
                            logger.warning("High LRS (<1kΩ), but vset cant be lowered, capped at %.3f", vset)
                    if vset < 2:
                        vset = round(vset + correction_step, ROUND_DIGITS)
                        logger.warning("Not in LRS after SET → vset ↑ to %.3f; cycle_exp reset to 0.0", vset)
                    else:
                        logger.warning("Not in LRS after SET, no increase of vset possible, vset capped at %.3f; cycle_exp reset to 0.0", vset)
                    set_failure += 1
                    n = 0.0

            break  # exit outer HRS-check loop

        else:
            # Not in HRS at the start — sweep SET/RESET and retry
            attempts += 1
            vsweepreset = round(vsweepreset + 0.1, ROUND_DIGITS)
            logger.warning("Not in HRS at start. Increasing sweep reset to %d. Sweep attempt %d", vsweepreset, attempts)
            r = sweep_and_read(vsweepset, vsweepreset)
            logger.info("Measured r = %.0f Ω", r)

    # ---------------- Step 3: Save data to CSV -----------------------------------------------------

    try:
        if records:
            df = pd.DataFrame(records)
            csv_path = DATA_DIR / f"endurance_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
            df.to_csv(csv_path, index=False)
            logger.info("Saved %d records to %s", len(df), csv_path)
        else:
            logger.warning("No records collected — nothing to save.")
    finally:
        logger.info("=== Endurance testing finished ===")



def _timed_prompt(prompt: str, timeout_s: int = 30):
    """
    Show a prompt and wait for user input for up to `timeout_s` seconds.
    Displays a live countdown. Returns the entered string (stripped),
    or None if timed out / input unavailable.
    """
    q: queue.Queue[str | None] = queue.Queue()

    def _reader():
        try:
            # Using input() in a daemon thread; fine for CLI use.
            s = input()
            q.put(None if s is None else s.strip())
        except Exception:
            q.put(None)

    print(f"{prompt} (y/n) — waiting {timeout_s}s...", flush=True)
    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    for remaining in range(timeout_s, 0, -1):
        # Inline countdown on one line
        print(f"\rTime left: {remaining:2d}s  ", end="", flush=True)
        time.sleep(1)
        if not q.empty():
            resp = q.get_nowait()
            print()  # newline after countdown line
            return resp

    # One last check after loop
    print("\rTime left:  0s  ", end="", flush=True)
    print()  # newline
    if not q.empty():
        return q.get_nowait()
    return None


def run_endurance_campaign(initial_kwargs: dict,
ask_timeout_s: int = 30,
on_timeout_increment: bool = True):
    """
    Repeatedly run `endurance_testing(**kwargs)`.
    After each run, ask the user if they want to stop.
    - If the user types 'y' (case-insensitive) → stop.
    - If 'n' → continue with goal incremented by 1.
    - If no input within `ask_timeout_s` and `on_timeout_increment=True`
      → continue with goal incremented by 1 (default behavior).
      Otherwise (on_timeout_increment=False) re-run with the SAME goal.

    `initial_kwargs` should contain all arguments for `endurance_testing`,
    including 'goal'. This function mutates a local copy only.
    """
    kwargs = dict(initial_kwargs)  # local copy we can mutate
    if 'goal' not in kwargs:
        raise ValueError("initial_kwargs must include 'goal'")

    while True:
        # --- Run one endurance pass ---
        print(f"\n=== Running endurance_testing(goal={kwargs['goal']}) ===", flush=True)
        endurance_testing(**kwargs)

        # --- Ask whether to stop ---
        resp = _timed_prompt("Stop the campaign now?", timeout_s=ask_timeout_s)

        if resp is not None and resp.lower().startswith('y'):
            print("Stopping campaign on user request.")
            break

        # Decide next goal
        if resp is None:
            # Timeout path
            if on_timeout_increment:
                kwargs['goal'] = int(kwargs['goal']) + 1
                print(f"No input. Auto-continuing with goal={kwargs['goal']}.\n")
            else:
                print(f"No input. Auto-continuing with same goal={kwargs['goal']}.\n")
        else:
            # User explicitly said something (treat anything not starting with 'y' as 'no/continue')
            kwargs['goal'] = int(kwargs['goal']) + 1
            print(f"User chose to continue. Next goal={kwargs['goal']}.\n")

##################################### FUNCTIONS FOR EFFECT TESTING #####################################

import os
import time
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional  # <= compatible with Python <3.9

import matplotlib
import matplotlib.pyplot as plt

# --- Thresholds for deciding switched/not switched ---
LRS_MAX_OHM = 3500.0   # < 3500 Ω → LRS
HRS_MIN_OHM = 5500.0   # > 5500 Ω → HRS


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _save_all_figs(run_dir: str, stem: str) -> List[str]:
    """
    Save all currently open Matplotlib figures into run_dir with a given stem.
    Returns the list of saved file paths.
    """
    saved: List[str] = []
    # grab all figure managers (works even if figures created elsewhere)
    for m in matplotlib._pylab_helpers.Gcf.get_all_fig_managers():
        fig = m.canvas.figure
        out = os.path.join(run_dir, f"{stem}_fig{m.num}.png")
        try:
            fig.savefig(out, dpi=150, bbox_inches="tight")
            saved.append(out)
        except Exception as e:
            print(f"[warn] Could not save figure {m.num}: {e}")
    return saved


def _newest_df_since(base_dir: str, samplename: str, padname: str, since_ts: float) -> Optional[str]:
    """
    Find the newest *.df created/modified under C:\\Messdaten\\<samplename>\\<padname>\\** after since_ts.
    Returns the basename (e.g., '100ps_0dB_19.df') or None if not found.
    """
    root = os.path.join(base_dir, samplename, padname)
    newest = None
    newest_mtime = since_ts
    if not os.path.isdir(root):
        return None

    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".df"):
                full = os.path.join(r, f)
                try:
                    m = os.path.getmtime(full)
                except OSError:
                    continue
                # small grace offset to tolerate clock jitter
                if m >= since_ts - 0.5 and m >= newest_mtime:
                    newest_mtime = m
                    newest = full
    return os.path.basename(newest) if newest else None


# ------------------------ Pattern builders (exact spec) ------------------------

def _ip_seq(ip_delay: int) -> List[int]:
    return [0] * int(ip_delay)

def _build_pnp_pattern(ip_delay: int, amount_of_cycles: int) -> List[int]:
    """
    Build: ([1] + [0]*ip_delay + [-1]*7 + [0]*ip_delay) * amount_of_cycles
    """
    ip = _ip_seq(ip_delay)
    one_block = [1] + ip + ([-1] * 7) + ip
    return one_block * int(amount_of_cycles)

def _build_npn_pattern(ip_delay: int, amount_of_cycles: int) -> List[int]:
    """
    Build: ([-1]*7 + [0]*ip_delay + [1] + [0]*ip_delay) * amount_of_cycles
    """
    ip = _ip_seq(ip_delay)
    one_block = ([-1] * 7) + ip + [1] + ip
    return one_block * int(amount_of_cycles)


# ------------------------ Main runner ------------------------

def run_effect_test(
    pname: str,
    ip_delay: int,
    is_delay: int,
    amount_of_cycles: int,
    vset: float,
    vreset: float,
    *,
    runs: int = 10,
    record: int = 250,
    vsweepset: float,
    vsweepreset: float,
    samplename: str = "Artem_RF33_023_",
    base_dir: str = r"C:\Messdaten",
    figure_root: str = "effect_testing",
) -> dict:
    """
    Executes the endurance effect experiment:
      PNP block → read & judge (expect LRS),
      NPN block → read & judge (expect HRS, else sweep),
    repeated `runs` times.

    Side effects:
      - Creates `C:\Messdaten\<samplename>\effect_testing\<ip>_IP_<cycles>_cycles\`
      - Saves all open Matplotlib figures as PNG after each block
      - Writes a CSV log with the newest *.df file name for each block

    Returns a dict with p_LRS, p_HRS, p_overall and paths.
    """

    # Output directories/files
    test_dir = _ensure_dir(
        #os.path.join(base_dir, samplename, figure_root, f"{ip_delay}_IP_{amount_of_cycles}_cycles") # for ip_delay tests
        os.path.join(base_dir, samplename, figure_root, f"{is_delay}_IS_{amount_of_cycles}_cycles") # for is_delay tests
    )
    log_csv = os.path.join(test_dir, "log.csv")
    meta_json = os.path.join(test_dir, "meta.json")

    # Prepare CSV header
    header = [
        "timestamp_iso",
        "run_idx",
        "step",                  # "PNP" or "NPN"
        "resistance_ohm",
        "switched",              # 1/0 against the expected state
        "ppg_log_file",          # e.g. 100ps_0dB_19.df
        "figures_saved_count",
        "recovered_by_sweep",    # True/False
        "sweep_resistance_after" # float or empty
    ]
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(header)

    # Build patterns exactly per your spec
    pnp_pattern = _build_pnp_pattern(ip_delay, amount_of_cycles)
    npn_pattern = _build_npn_pattern(ip_delay, amount_of_cycles)

    # Per your requirement:
    is_delay_singleton = [0] * is_delay * 10   # for ip_delay tests set to [0], for is_delay test set to desired amount
    repetitions_once   = 50     # for ip_delay tests set to 1, for is_delay test set to desired amount
    pnp_final          = [1]
    npn_final          = [-1] * 7

    # Collect stats
    pnp_success: List[int] = []
    npn_success: List[int] = []

    for run_idx in range(1, runs + 1):
        #print(f"\n=== RUN {run_idx}/{runs} | ip_delay={ip_delay}, cycles={amount_of_cycles} ===")                   # for is_delay tests
        print(f"\n=== RUN {run_idx}/{runs} | is_delay={is_delay}, amount of sets={repetitions_once} ===")           # for ip_delay tests

        # ---- STEP A: PNP → expect LRS ----
        t_before = time.time()
        res_pnp = call_ppg_with_params(
            pname=pname,
            pulse_pattern=pnp_pattern,     # ([1]+ip+[−1]*7+ip)*amount
            is_delay=is_delay_singleton,   # [0]
            repetitions=repetitions_once,  # 1
            last_pulse=pnp_final,          # [1]
            vset=vset,
            vreset=vreset,
            record=record,
        )
        ppg_file_a = _newest_df_since(base_dir, samplename, pname, since_ts=t_before)
        switched_a = 1 if (res_pnp is not None and float(res_pnp) < LRS_MAX_OHM) else 0
        figs_a = _save_all_figs(test_dir, f"run{run_idx:02d}_PNP")

        # Log step A
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(timespec="seconds"),
                run_idx,
                "PNP",
                float(res_pnp) if res_pnp is not None else "",
                switched_a,
                ppg_file_a or "",
                len(figs_a),
                False,
                ""
            ])
        pnp_success.append(switched_a)

        # ---- STEP B: NPN → expect HRS ----
        t_before = time.time()
        res_npn = call_ppg_with_params(
            pname=pname,
            pulse_pattern=npn_pattern,     # ([-1]*7+ip+[1]+ip)*amount
            is_delay=is_delay_singleton,   # [0]
            repetitions=repetitions_once,  # 1
            last_pulse=npn_final,          # [-1]*7
            vset=vset,
            vreset=vreset,
            record=record,
        )
        ppg_file_b = _newest_df_since(base_dir, samplename, pname, since_ts=t_before)
        switched_b = 1 if (res_npn is not None and float(res_npn) > HRS_MIN_OHM) else 0

        recovered = False
        sweep_R = ""
        if switched_b == 0:
            # Device likely stuck in LRS → recover
            try:
                sweep_R_val = sweep_and_read(vset=vsweepset, vreset=vsweepreset)
                sweep_R = float(sweep_R_val)
                recovered = True
            except Exception as e:
                print(f"[warn] sweep_and_read failed: {e}")

        figs_b = _save_all_figs(test_dir, f"run{run_idx:02d}_NPN")

        # Log step B
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(timespec="seconds"),
                run_idx,
                "NPN",
                float(res_npn) if res_npn is not None else "",
                switched_b,
                ppg_file_b or "",
                len(figs_b),
                recovered,
                sweep_R
            ])
        npn_success.append(switched_b)

    # --- Compute stats ---
    def _mean(xs: List[int]) -> float:
        xs = [int(x) for x in xs if x in (0, 1)]
        return (sum(xs) / len(xs)) if xs else 0.0

    p_LRS = _mean(pnp_success)     # success ratio for PNP → LRS
    p_HRS = _mean(npn_success)     # success ratio for NPN → HRS
    p_overall = _mean(pnp_success + npn_success)

    summary = {
        "pname": pname,
        "ip_delay": ip_delay,
        "is_delay": is_delay,
        "amount_of_cycles": amount_of_cycles,
        "vset": vset,
        "vreset": vreset,
        "runs": runs,
        "log_csv": log_csv,
        "output_dir": test_dir,
        "p_LRS": p_LRS,
        "p_HRS": p_HRS,
        "p_overall": p_overall,
        "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
    }

    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"PNP → LRS success (p_LRS): {p_LRS:.3f}")
    print(f"NPN → HRS success (p_HRS): {p_HRS:.3f}")
    print(f"Overall success (both steps combined): {p_overall:.3f}")
    print(f"Log file: {log_csv}")
    print(f"Figures dir: {test_dir}")

    return summary
