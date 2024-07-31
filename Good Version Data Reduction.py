# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:02:32 2024

@author: kellyobj
"""

#import modules
import pandas as pd
import numpy as np
import h5py as hp
import re
import array as arr
from array import *
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import tkinter as tk
import sys
from PyQt5 import QtWidgets, uic
import csv
from scipy.optimize import curve_fit 
import seaborn as sns



########################################################

#Make an array for the change of days based on the file_number - Temp
start_date_file_numbers = np.array([62,178,318, 0, 0, 0, 0])

#change the column width to allow all data to be extracted
pd.set_option('display.max_colwidth', None)

########## Variable Information ################
variable_names = np.array ([])
temp_gui_check = False
HKL_gui_check = False
sort_values = True
wrong_name_initial_file = False

###############################################


#Create the various dataframes that will be used throughout, as well as create the titles
def create_various_dataframes():
    global titles_df, non_sorted_df, depolarised_df, organised_df_dict, organised_df_base
    
    # Define empty databases for the seperate data
    #if instrument_type == 'Taipan' or instrument_type == 'Sika':  
        
    if len(variable_names) == 0:
        titles_df = pd.Series(['scan','Elapsed Hours', 'Flip', 'Beam Mon',
                              'Counts/Beam Mon', 'Counts Relative Unpol', 'Counts Silicon Corrected', 'Magnitude','Phe Calc','Trans Calc','Error^2'])
    
    elif len(variable_names) == 1:
        titles_df = pd.Series(['scan','Elapsed Hours', variable_names[0], 'Flip', 'Beam Mon',
                              'Counts/Beam Mon', 'Counts Relative Unpol', 'Counts Silicon Corrected', 'Magnitude','Phe Calc','Trans Calc','Error^2'])
    elif len(variable_names) == 2:
        titles_df = pd.Series(['scan','Elapsed Hours', variable_names[0], variable_names[1], 'Flip', 'Beam Mon',
                              'Counts/Beam Mon', 'Counts Relative Unpol', 'Counts Silicon Corrected', 'Magnitude','Phe Calc','Trans Calc','Error^2'])
    elif len(variable_names) == 3:
        titles_df = pd.Series(['scan','Elapsed Hours', variable_names[0], variable_names[1],variable_names[2], 'Flip', 'Beam Mon',
                              'Counts/Beam Mon', 'Counts Relative Unpol', 'Counts Silicon Corrected', 'Magnitude','Phe Calc','Trans Calc','Error^2'])
    elif len(variable_names) == 4:
             titles_df = pd.Series(['scan','Elapsed Hours', variable_names[0], variable_names[1],variable_names[2], variable_names[3], 'Flip', 'Beam Mon',
                                   'Counts/Beam Mon', 'Counts Relative Unpol', 'Counts Silicon Corrected', 'Magnitude','Phe Calc','Trans Calc','Error^2'])
    if instrument_type =='Quokka':
      titles_df = pd.Series(['scan','Elapsed Hours', 'Flip', 'Beam Monitor1', 'Beam Monitor 2', 'Beam Monitor 3',
                              'wavelength','Magnitude','Total Counts','Phe Calc','Trans Calc','Error^2'])
    if instrument_type == 'Wombat':
       titles_df = pd.Series(['scan','run_count','Elapsed Hours Per Scan', 'Flip', 'Beam Monitor1','Elapsed Time', 'Beam Monitor 2', 'Elapsed Time',
                             'Total Counts','Phe Calc','Trans Calc','Error^2'])
    
    #Create a database for the required data before organisation, as well as for the depoalrized data
    non_sorted_df = pd.DataFrame([list(titles_df)])
    depolarised_df = pd.DataFrame([list(titles_df)])
    df_count_number = len(variable_names)**6
    df_count = np.arange(df_count_number)
    organised_df_dict = {name: pd.DataFrame([list(titles_df)]) for name in df_count}
    organised_df_base = pd.DataFrame([list(titles_df)])
    if len(variable_names) == 0:
        df_count_number = 3
        df_count = np.arange(df_count_number)
        organised_df_dict = {name: pd.DataFrame([list(titles_df)]) for name in df_count}


############# Define constants ################

pressure = 1
thickness = 10
wavelength = 4.04
t_si = 0.9809
opacity_calc = 0.0732 * pressure * thickness * wavelength
beam_mon_count = 30000
counts_per_bm_intensity_check = 13.8210
good_data = 2
bad_data = 1
minus_organise_range = 1
bounds = [(1, 150), (0.0, 1)]
spinflip_state = 'normal'
broken_files_str = ''
error_message = "New Errors Can be seen on page 7 - Errors"
polarisation_method = 'NSF'
instrument_type = ''
depolarised_scans_manual_input = ''
spin_flip = 'error'
popt_array = arr.array('d'[0])
error_array = arr.array('d'[0])
##############################################


############### Set True parameters #################

variables = True
organise = True
calc_polarise = True

############ Formula Definitions #############
def calc_t_si(wavelength):
    global t_si
    t_si = (-0.0047*(wavelength)) + 0.9998
    
def opacity_helium_depol_formula(depol_count):
    global opacity_depolarised
    opacity_depolarised = -0.5 * math.log(depol_count)
    
def trans_intensity_nsf(Phe):
   global trans_intensity
   trans_intensity = 0.5*np.exp(-2*opacity_depolarised*(1-Phe)) + 0.5*np.exp(-2*opacity_depolarised*(1+Phe))

def gauss(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def gauss_quokka(x, A, mu, sigma, bg):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2)) + bg

def trans_intensity_sf(Phe):
    global trans_intensity
    trans_intensity = 0.5*np.exp(-opacity_depolarised*(1-Phe)) * np.exp(-opacity_depolarised*(1+Phe))
    
def error_nsf_total(params, opacity_depolarised, t_values, counts_silicon_corrected_values):
    T, Phe0 = params
    total_error = 0    
    
    for t, counts_silicon_corrected in zip(t_values, counts_silicon_corrected_values):
        Phe = np.exp(-t/T) * Phe0
        trans_calc = (0.5*np.exp(-2*opacity_depolarised * (1 - Phe)))+ 0.5 * np.exp(-2* opacity_depolarised * (1+Phe))
        total_error += (trans_calc - counts_silicon_corrected) ** 2
    return total_error

def error_sf_total(params, opacity_depolarised, t_values, counts_silicon_corrected_values):
    global Phe0, T
    T, Phe0 = params
    total_error = 0
    for t, counts_silicon_corrected in zip(t_values, counts_silicon_corrected_values):
        total_error += ((0.5 * np.exp(-opacity_depolarised * (1 - np.exp(-t / T) * Phe0))) * 
                        0.5 * np.exp(-opacity_depolarised * (1 + (np.exp(-t / T) * Phe0))) - 
                        counts_silicon_corrected) ** 2
    return total_error

def Phe_calc(Phe0, T, t):
    global Phe
    Phe = np.exp(-t/T) * Phe0
    return Phe
    
def trans_intensity_nsf_error_Phe_calc(Phe0,T,t):
    global error_nsf
    error_nsf = ((0.5*np.exp(-2*opacity_depolarised*(1-(np.exp(-t/T) * Phe0))) + 0.5*np.exp(-2*opacity_depolarised(1+(np.exp(-t/T) * Phe0)))) - counts_silicon_corrected)**2

def trans_intensity_sf_error_Phe_calc(Phe0,T,t):
    global error_sf
    error_sf = ((0.5*np.exp(-opacity_depolarised*(1-(np.exp(-t/T) * Phe0))) * np.exp(-opacity_depolarised*(1+(np.exp(-t/T) * Phe0)))) - counts_silicon_corrected)**2


############################Plot Data ###################################

def plot_data_taipan_dat(input_data):
    global min_input, max_input, mean_input, sig_input, error, pcov, popt, intensity
    min_input = np.min(input_data)
    max_input = np.max(input_data)
    mean_input = np.mean(input_data)
    sig_input = np.std(input_data)
    x = np.linspace(min_input, max_input, len(input_data))
    y = input_data
    try: 
        popt, pcov = curve_fit(gauss, x, y, p0 = [max_input, mean_input, sig_input], maxfev=5000)
        error = np.sqrt(np.diagonal(pcov))

    except (RuntimeError, TypeError):
        popt = np.array([0,0,0,0])
        pcov = np.array([0,0,0,0])
    #error = np.sqrt(np.diagonal(pcov))
    intensity = popt[0]
    #plt.plot(x, gauss( x, *popt))
    
def plot_data_hdf(test, bins):    
    global n, patches, x, y, tester, min_test, max_test,mean, sig, popt, pcov, error, intensity
    tester = np.nonzero(test.flatten()) #flatten the data, then take the zeros out of the data
#    bins = 100                           #Set the bin count
    min_test = (np.min(tester))           #Define variables (minimum, maximum, mean and std)
    max_test = (np.max(tester))
    mean = np.mean(tester)
    sig = np.std(tester)
    n,rand, patches = plt.hist(tester, bins)   #plot the histogram
    x = np.linspace(min_test, max_test, bins)
    y = n
    popt, pcov = curve_fit(gauss_quokka,x, y, p0 = [max_test, mean, sig, sum(tester[0])]) #Curve fit the histogram
    error = np.sqrt(np.diagonal(pcov))
    intensity = popt[0]
    #plt.plot(x, gauss(x,*popt), color ='r')     #plot the histogram and the curve fitted line
    
def plot_hdf_data():
    y = np.ravel(hmm_xy)
    sum_y = len(y) 
    x = np.asarray(range(sum_y))
    plt.plot (x,y)
    plt.xlabel ('Spectrum')
    plt.ylabel ('Count')
    plt.title('Count for Data')
    max_y = np.max(y)

def plot_heat_map_wombat(data):
    heatmap_plot = sns.heatmap(data)
    plt.show()
    
#####################Load the data#########################

#extract the different variables from the datafiles
def variables_datfile_sika(file):
    global intensity, position, title, date, time, scan_number, data_titles
    intensity = float(str(file.loc[len(df)-2]).split('=')[2].split('N')[0][1:-1])
    position = float(str(df.loc[len(df)-3]).split('=')[2].split('N')[0][1:-1])
    title = str(df.loc[9]).split('scan_title = ')[1].split('Name')[0][0:-1]
    time = str(df.loc[1]).split(' time = ')[1].split('Name')[0][0:-1]
    date = str(df.loc[0]).split(' date = ')[1].split('Name')[0]
    data_titles = str(df.loc[33]).split(' ')

#extract the different variables from the datafiles
def variables_datfile_taipan(file):
    global position, title, date, time, scan_number, data_titles
    title = str(df.loc[10]).split('scan_title = ')[1].split('Name')[0][0:-1]
    time = str(df.loc[2]).split('time = ')[1].split('Name')[0].split(' ')[1][0:-1]
    date = str(df.loc[2]).split('time = ')[1].split('Name')[0].split(' ')[0]
    data_titles = str(df.loc[29]).split(' ')
    
def load_raw_data_hdf(fn):
    global group_1, hmm_xy, hdf, dataget1
    hdf = hp.File(fn, 'r')
    first_section = list(hdf.keys())           #save a list of whats in the file
    first_section_name = first_section[0]
    group_1 = hdf.get(first_section_name)      #Open up the first group, usually the title
    group_1_items = list(group_1.items())     #List the items in the first group
    dataget1 = group_1.get('data')       #Open up desired subgroup and list these items
    dataget1_items = list(dataget1.items())
    hmm_xy =np.array( dataget1.get ('hmm_xy')) #pull out the data and turn into an array
    
def load_other_data_hdf_wombat():
    global bm1_time, bm2_time,title,experiment_get, run_number, total_counts, date_time, monitor_get, bm1_counts, bm2_counts
    run_number = np.array(dataget1.get('run_number'))
    total_counts = np.array(dataget1.get('total_counts'))
    date_time = group_1.get('start_time')
    date_time = ''.join(map(str,date_time))[2:-1]
    monitor_get = group_1.get('monitor')
    bm1_counts = np.array(monitor_get.get('bm1_counts'))
    bm1_time = np.array(monitor_get.get('bm1_time'))
    bm2_counts = np.array(monitor_get.get('bm2_counts'))
    bm2_time = np.array(monitor_get.get('bm2_time'))
    experiment_get = group_1.get("experiment")
    title = experiment_get.get('title')
    title = ''.join(map(str,title))[2:-1]
    
def load_other_data_hdf_quokka():
    global run_number, total_counts, date_time, monitor_get, bm1_counts, bm2_counts, bm3_counts, wavelength
    run_number = np.array(dataget1.get('run_number'))
    total_counts = np.array(dataget1.get('total_counts'))
    wavelength = np.array(dataget1.get('wavelength'))
    date_time = group_1.get('start_time')
    date_time = ''.join(map(str,date_time))[2:-1]
    monitor_get = group_1.get('monitor')
    bm1_counts = np.array(monitor_get.get('bm1_counts'))
    bm2_counts = np.array(monitor_get.get('bm2_counts')) 
    bm3_counts = np.array(monitor_get.get('bm3_counts'))
    experiment_get = group_1.get("experiment")
    title = experiment_get.get('title')
    title = ''.join(map(str,title))[2:-1]
    plot_data_hdf(hmm_xy, 100)


        
def determine_starting_time_hdf():
    global start_time_bm1, start_time_bm2, date_time, start_hour, start_minute, start_second, start_time, start_date, start_year, start_month, start_day, date, start_time_in_seconds, time
    date_time = date_time.split(' ')
    time = date_time[1]
    start_time = time.split(':')
    start_hour = int(start_time[0])
    start_minute = int(start_time[1])
    start_second = float(start_time[2])
    date = date_time[0]
    start_date = date.split('-')
    start_year = int(start_date[0])
    start_month = int(start_date[1])
    start_day = int(start_date[2])
    start_time_in_seconds = (start_hour * 3600) + \
        (start_minute * 60) + start_second

    if instrument_type == 'Wombat':
        start_time_bm1 = bm1_time
        start_time_bm2 = bm2_time
        
#Determine the current time
def determine_time_hdf():
    global hour, minute, second, date, year, month, day, date, time_in_seconds, date_time
    date_time = date_time.split(' ')
    time = date_time[1]
    current_time = time.split(':')
    hour = int(current_time[0])
    minute = int(current_time[1])
    second = float(current_time[2])
    date = date_time[0]
    date = date.split('-')
    year = int(date[0])
    month = int(date[1])
    day = int(date[2])
    if day == start_day:
        time_in_seconds = (hour * 3600) + (minute * 60) + second
    elif day == start_day + 1:
        time_in_seconds = (hour * 3600) + (minute * 60) + second + 86400
        
#determine the initial starting time
def determine_starting_time_dat():
    global start_hour, start_minute, start_second, start_time, start_date, start_year, start_month, start_day, date, start_time_in_seconds, time
        
    start_time = time.split(':')
    start_hour = int(start_time[0])
    start_minute = int(start_time[1])
    start_second = float(start_time[2])
    start_date = date.split('-')
    start_year = int(start_date[0])
    start_month = int(start_date[1])
    start_day = int(start_date[2])
    start_time_in_seconds = (start_hour * 3600) + \
        (start_minute * 60) + start_second

def determine_bm_time_per_run():
    global bm_elapsed_time_df, time_in_seconds, start_time_in_seconds
    bm_elapsed_titles = pd.Series(['bm1_elapsed_time', 'bm2_elapsed_time'])
    bm_elapsed_time_df = pd.DataFrame([list(bm_elapsed_titles)])
    try:
        start_time_in_seconds
    except NameError:
        start_time_in_seconds = 0
    
    try:
        time_in_seconds
    except NameError:
        time_in_seconds = start_time_in_seconds

        
    bm1_time_total =(time_in_seconds - start_time_in_seconds) + bm1_time[0]
    bm2_time_total =(time_in_seconds - start_time_in_seconds) + bm2_time[0]
    bm_combine_time = pd.Series([bm1_time_total, bm2_time_total])
    bm_elapsed_time_df.loc[0] = bm_combine_time
    z = 1
    for x in range(len(run_number)):
       if z == len(run_number):
           break
       else: 
            bm1_time_total += bm1_time[z]
            bm2_time_total += bm2_time[z]
            bm_combine_time = pd.Series([bm1_time_total, bm2_time_total])
            bm_elapsed_time_df.loc[z] = bm_combine_time
            z += 1
    
    bm_elapsed_time_df.columns = bm_elapsed_titles


#Determine the current time
def determine_time():
    global hour, minute, second, date, year, month, day, date, time_in_seconds, time
    current_time = time.split(':')
    hour = int(current_time[0])
    minute = int(current_time[1])
    second = float(current_time[2])
    date = date.split('-')
    year = int(date[0])
    month = int(date[1])
    day = int(date[2])
    if day == start_day:
        time_in_seconds = (hour * 3600) + (minute * 60) + second
    elif day == start_day + 1:
        time_in_seconds = (hour * 3600) + (minute * 60) + second + 86400

#Determine the elapsed time in both seconds and hours
def elapsed_time():
    global elapsed_time_in_seconds, elapsed_time_in_hrs
    elapsed_time_in_seconds = time_in_seconds - start_time_in_seconds
    
    elapsed_time_in_hrs = elapsed_time_in_seconds / 3600

#Determine the spin flip
def determine_spinflip_dat_normal():
    global spin_flip

    if str(re.findall(" SF", title))[2:-2] == " SF":
        spin_flip = 'SF'
    elif str(re.findall("NSF", title))[2:-2] == "NSF":
        spin_flip = "NSF"
    else:
        spin_flip = "error"

def determine_spinflip_dat_flip():
    global spin_flip

    if str(re.findall(" SF", title))[2:-2] == " SF":
        spin_flip = 'NSF'
    elif str(re.findall("NSF", title))[2:-2] == "NSF":
        spin_flip = "SF"
    else:
        spin_flip = "error"
        
        
#Determine the temperature from the Title
def temp_from_name(temp_input):
    global temp
    temp_range_input = 0

    for x in range(len(temp_input)):
        if str(re.findall(temp_input[temp_range_input], title))[2:-2] == temp_input[temp_range_input]:
            temp = temp_input[temp_range_input]
            temp_range_input = temp_range_input + 1
        else:
            temp_range_input = temp_range_input + 1


#Determine the HKL from the title
def HKL_from_name(variable_2_input):
    global HKL
    HKL_range_input = 0

    for x in range(len(variable_2_input)):
        if str(re.findall(variable_2_input[HKL_range_input], title))[2:-2] == variable_2_input[HKL_range_input]:
            HKL = variable_2_input[HKL_range_input]
            HKL_range_input = HKL_range_input + 1
        else:
            HKL_range_input = HKL_range_input + 1

#Determine the exp from name (pzz, pyy etc) - Minimal use going forward
def exp_from_name(exp_input):
    global exp
    exp_range_input = 0

    for x in range(len(exp_input)):
        if str(re.findall(exp_input[exp_range_input], title))[2:-2] == exp_input[exp_range_input]:
            exp = exp_input[exp_range_input]
            exp_range_input = exp_range_input + 1
        else:
            exp_range_input = exp_range_input + 1

#determine the start of a new experiment based on start days 
def determine_start_of_new_experiment():
       if file_number == start_date_file_numbers[0] or file_number== start_date_file_numbers[1] or file_number == start_date_file_numbers[2] or file_number == start_date_file_numbers[3]:
           determine_starting_time_dat()    

#Determine the counts per beam monitor
def det_counts_per_beammonitor():
    global counts_per_bm, counts_relative_unpol, counts_silicon_corrected
    counts_per_bm = intensity/beam_mon_count
    counts_relative_unpol = counts_per_bm/counts_per_bm_intensity_check
    counts_silicon_corrected = counts_relative_unpol * (t_si**-4)


######################## Organise Data ##########################
#Save the original data into a pulled dataframe prior to organisation
def save_data_non_sorted_df_no_variables():
    newline_df = pd.Series([file_number, elapsed_time_in_hrs, spin_flip, beam_mon_count,
                               counts_per_bm, counts_relative_unpol, counts_silicon_corrected, intensity])
    non_sorted_df.loc[inte] = newline_df

def save_data_non_sorted_hdf():
    if instrument_type == 'Wombat':
        newline_df = pd.Series([file_number, run_number[run] ,elapsed_time_in_hrs, spin_flip, bm1_counts[run],bm1_elapsed_time[run],bm2_counts[run],bm2_elapsed_time[run],
                               wavelength, total_counts[run]])
        non_sorted_df.loc[inte] = newline_df
    elif instrument_type == "Quokka":
        newline_df = pd.Series([file_number, elapsed_time_in_hrs, spin_flip, int(bm1_counts), int(bm2_counts), int(bm3_counts), 
                               float(wavelength), intensity, int(total_counts)])
        non_sorted_df.loc[inte] = newline_df

def save_data_non_sorted_df():
    newline_df = pd.Series([file_number, elapsed_time_in_hrs, HKL, temp, spin_flip, beam_mon_count,
                               counts_per_bm, counts_relative_unpol, counts_silicon_corrected, intensity])
    non_sorted_df.loc[inte] = newline_df

def save_data_organise_df_no_variables(df_organise):
    newline_df = pd.Series([file_number, elapsed_time_in_hrs, spin_flip,beam_mon_count,
                               counts_per_bm, counts_relative_unpol, counts_silicon_corrected, intensity])
    df_organise.loc[len(df_organise)] = newline_df

#Save the organised data to a organised dataframe
def save_data_organise_df(df_organise):
    
    newline_df = pd.Series([file_number, elapsed_time_in_hrs, HKL, temp, spin_flip,beam_mon_count,
                               counts_per_bm, counts_relative_unpol, counts_silicon_corrected, intensity])
    df_organise.loc[len(df_organise)] = newline_df
        
#Save the data again to a organised dictionary
def save_data_organise_df2(df_organise):
    df_organise.loc[len(df_organise)] = non_sorted_df.loc[search]       

#A function which creates all the required data from the opened files .DAT FILES ONLY
def create_data():
        if instrument_type == 'Sika':
               variables_datfile_sika(df)
        elif instrument_type == 'Taipan':
           variables_datfile_taipan(df)
        if spinflip_state == 'normal':
               determine_spinflip_dat_normal()
        elif spinflip_state == 'flip':
                determine_spinflip_dat_flip()
            
        if len(variable_names) < 0 and any(variable_names == 'Temp') == True:    
            temp_from_name(variable_1_input)
        if len(variable_names) < 0 and any(variable_names == 'HKL') == True:    
            HKL_from_name(variable_2_input)
        #exp_from_name(exp_input)
        determine_start_of_new_experiment()
        determine_time()
        elapsed_time()
        det_counts_per_beammonitor()
        

        
def create_data_taipan_dat():
        variables_datfile_taipan(df)
        if spinflip_state == 'normal':
            determine_spinflip_dat_normal()
        elif spinflip_state == 'flip':
            determine_spinflip_dat_flip()
            
        if len(variable_names) < 0 and any(variable_names == 'Temp') == True:    
            temp_from_name(variable_1_input)
        if len(variable_names) < 0 and any(variable_names == 'HKL') == True:    
            HKL_from_name(variable_2_input)
        #exp_from_name(exp_input)
        determine_start_of_new_experiment()
        determine_time()
        elapsed_time()
#        det_counts_per_beammonitor()

#Finds the depolarised data by searching for the word 'depolarised' in the titles. 
def find_depolarised_data():
    global depolarised_scans_manual_input
    if len(depolarised_scans_manual_input) == 0:
        if str(re.findall('Depolarized',title))[2:-2] == 'Depolarized' or str(re.findall('d...',title))[2:-2] == 'd...':
            if len(variable_names) == 0:    
                save_data_organise_df_no_variables(depolarised_df)
            else:
                save_data_organise_df(depolarised_df)
                
    elif len(depolarised_scans_manual_input) >= 1:       
        if file_number == int(depolarised_scans_manual_input[0]):
            if len(variable_names) == 0:
                save_data_organise_df_no_variables(depolarised_df)
            else:
                save_data_organise_df(depolarised_df)
#Goes through the original organised data and organises again into groups based on variables used
def organise_data2():
    global search, save_og

    search_inte = 100000
    search = 1
    save_og = organised_df_dict[0]
    save_og.loc[1] = non_sorted_df.loc[1]    
    organised_df_dict[0] = save_og
    loca = 0
    for i in range(search_inte):
        checked_file = organised_df_dict[loca]

        if len(checked_file) == 1:
            save_data_organise_df2(organised_df_dict[loca])
            search = search + 1
            loca = 0
        
        elif checked_file.loc[1][0] == non_sorted_df.loc[search][0]:
            search = search + 1
            loca = 0 
            
        elif checked_file.loc[1][2] == non_sorted_df.loc[search][2] and checked_file.loc[1][3] == non_sorted_df.loc[search][3] and checked_file.loc[1][4] == non_sorted_df.loc[search][4] and checked_file.loc[1][5] == non_sorted_df.loc[search][5]:
            save_data_organise_df2(organised_df_dict[loca])
            search = search + 1
            loca = 0         
        
        elif search == file_range - minus_organise_range or search > file_range - 1:
            break
        
        else:
            loca = loca + 1
         
def organise_data_HDF_non_sorted():
    global file_number, non_sorted_df, elapsed_time_in_hrs
    if instrument_type == 'Wombat':
        newline_df = pd.Series([file_number, run_number[run] ,elapsed_time_in_hrs, spin_flip, bm1_counts[run],bm1_elapsed_time[run],bm2_counts[run],bm2_elapsed_time[run],
                               total_counts[run]])
        non_sorted_df.loc[inte] = newline_df
        
    elif instrument_type == "Quokka":
        newline_df = pd.Series('d',[])
        try:
            elapsed_time_in_hrs
        except NameError:
            elapsed_time_in_hrs = 0
        newline_df = pd.Series([file_number, elapsed_time_in_hrs, spin_flip, int(bm1_counts), int(bm2_counts), int(bm3_counts), 
                               float(wavelength), intensity, int(total_counts)])
        #non_sorted_df = non_sorted_df.append(newline_df, ignore_index = True)
        non_sorted_df.loc[inte] = newline_df
        
def organise_data_no_variables():
    global search, checked_file   
    search_inte = 100000
    search = 1
    save_og = organised_df_dict[0]
    save_og.loc[1] = non_sorted_df.loc[1]    
    organised_df_dict[0] = save_og
    loca = 0   
    for i in range(search_inte):
        if len(organised_df_dict[0]) + len(organised_df_dict[1]) == file_range - 1:
            break
        else:
            checked_file = organised_df_dict[loca]

            if len(checked_file) == 1:
                save_data_organise_df2(organised_df_dict[loca])
                search = search + 1
                loca = 0
            elif len(non_sorted_df) == search:
                break
            
            elif checked_file.loc[1][0] == non_sorted_df.loc[search][0]:
                search = search + 1
                loca = 0
            
            elif checked_file.loc[1][2] == non_sorted_df.loc[search][2]:
                save_data_organise_df2(organised_df_dict[loca])
                search = search + 1
                loca = 0         
            
            else:
                loca = loca + 1    
    
#organises the data into the puleld dataframes
def organise_data():
    global minus_organise_range, non_sorted_df
    create_data()
    load_raw_data_sika_dat()
    compare_data_to_expected()
    if acceptable_data % 2 == 0:
        if len(variable_names) > 0:
            save_data_non_sorted_df()
        elif len(variable_names) == 0:
            save_data_non_sorted_df_no_variables()
    else:
        #print (file_name + 'is broken')       
        minus_organise_range = minus_organise_range + 1

    non_sorted_df = non_sorted_df.reset_index (drop = True)
    find_depolarised_data()

def organise_data_taipan():
    global minus_organise_range, non_sorted_df
    create_data()
    load_raw_data_taipan_dat()
    compare_data_to_expected()
    if acceptable_data % 2 == 0:
        if len(variable_names) > 0:
            save_data_non_sorted_df()
        elif len(variable_names) == 0:
            save_data_non_sorted_df_no_variables()
    else:
        #print (file_name + 'is broken')       
        minus_organise_range = minus_organise_range + 1

    non_sorted_df = non_sorted_df.reset_index (drop = True)
    find_depolarised_data()
    
#loads the data and organises it the initial time, scrolls through all the organised data.
def load_input_variables_sika():
    global file_name, df, file_range, file_number, inte, data_raw_datfile
    file_name = str(experiment) + '_0000' + str(file_number) + '.dat'
    if os.path.isfile(file_name) == True:
        df = pd.read_csv(file_name, delimiter='\t')
        create_various_dataframes()
        variables_datfile_sika(df)
    
        determine_starting_time_dat()
        file_number = int(file_number) + 1
        
        inte = 1
    else:
        wrong_name_initial_file = True
    for x in range(file_range):
        if int(file_number) < 100:
            file_name = str(experiment) + '_0000' + str(file_number) + '.dat'
            if os.path.isfile(file_name) == True :
                
                df = pd.read_csv(file_name, delimiter='\t')
                data_raw_datfile = np.loadtxt(file_name)
                organise_data()
                find_depolarised_data()
                file_number = int(file_number) + 1
                
                inte = inte + 1
            else:
                file_number = int(file_number)+ 1

        else:
            file_name = str(experiment) + '_000' + str(file_number) + '.dat'
            if os.path.isfile(file_name) == True:
                
                df = pd.read_csv(file_name, delimiter='\t')
                data_raw_datfile = np.loadtxt(file_name)
                organise_data()
                file_number = int(file_number) + 1
                inte = inte + 1
            else:
                file_number = int(file_number)+1
                
def load_input_variables_taipan_dat():
    global file_name, df, file_range, file_number, inte, data_raw_datfile
    #file_number = str('62')  #61
    #experiment = '0311'
            #Set range to 116 for base tests
    
    file_name = 'TAIPAN_exp' + str(experiment) + '_scan' + str(file_number) + '.dat'
    df = pd.read_csv(file_name, delimiter='\t')
    create_various_dataframes()
    variables_datfile_taipan(df)
    determine_starting_time_dat()
    file_number = int(file_number) + 1
    
    inte = 1

    for x in range(file_range):
        if int(file_number) < 100:
            file_name = 'TAIPAN_exp' + str(experiment) + '_scan' + str(file_number) + '.dat'
            if os.path.isfile(file_name) == True :
                
                df = pd.read_csv(file_name, delimiter='\t')
                data_raw_datfile = np.loadtxt(file_name)
                variables_datfile_taipan(df)
                load_raw_data_taipan_dat()
                pull_intensity_data_taipan_dat()
                organise_data_taipan()
                find_depolarised_data()
                file_number = int(file_number) + 1
                
                inte = inte + 1
            else:
                file_number = int(file_number)+ 1

        else:
            file_name = 'TAIPAN_exp' + str(experiment) + '_scan' + str(file_number) + '.dat'
            if os.path.isfile(file_name) == True:
                
                df = pd.read_csv(file_name, delimiter='\t')
                data_raw_datfile = np.loadtxt(file_name)
                variables_datfile_taipan(df)
                load_raw_data_taipan_dat()
                pull_intensity_data_taipan_dat()
                organise_data_taipan()
                create_data_taipan_dat()
                
                file_number = int(file_number) + 1
                inte = inte + 1
            else:
                file_number = int(file_number)+1


def load_input_variables_quokka_HDF():
    global file_name, df, file_range, file_number, inte, data_raw_datfile  , inte
    file_name = 'QKK' + str(experiment)  + str(file_number) + '.nx.hdf'
    load_raw_data_hdf(file_name)
    create_various_dataframes()
    load_other_data_hdf_quokka()
    determine_starting_time_hdf()
    inte = 1
    organise_data_HDF_non_sorted()
    file_number = int(file_number) + 1
    inte += 1
    
    for x in range(file_range):
        if int(file_number) < 100:
            file_name = 'QKK' + str(experiment) + '0' + str(file_number) + '.nx.hdf'
            if os.path.isfile(file_name) == True :
                combine_functions_pre_QUOKKA(file_name)
                organise_data_HDF_non_sorted()
                file_number = int(file_number) + 1
                
                inte = inte + 1
            else:
                file_number = int(file_number)+ 1

        else:
            file_name = 'QKK' + str(experiment)  + str(file_number) + '.nx.hdf'
            if os.path.isfile(file_name) == True:
                combine_functions_pre_QUOKKA(file_name)
                organise_data_HDF_non_sorted()
                file_number = int(file_number) + 1
                
                inte = inte + 1
            else:
                file_number = int(file_number)+1

#Seperately loads the raw numerical data
def load_raw_data_sika_dat():
    global data_raw_datfile, data_titles
    data_raw_datfile = pd.DataFrame(data_raw_datfile)
    data_titles = [x for x in data_titles if x]
    del data_titles[:5]
    del data_titles[-3:]
    if len(data_raw_datfile) > 50:
        data_raw_datfile = data_raw_datfile.T
    data_raw_datfile.columns = data_titles
    
def load_raw_data_taipan_dat():
    global data_raw_datfile, data_titles
    data_raw_datfile = pd.DataFrame(data_raw_datfile)
    data_titles = [x for x in data_titles if x]
    del data_titles[:5]
    del data_titles[-3:]
    try:
        data_raw_datfile.columns = data_titles
    except ValueError:
        data_raw_datfile = data_raw_datfile.T
        data_raw_datfile.columns = data_titles
    
def determine_title_pos_time(inter):
    global time_pos
    dataset_check = organised_df_dict[inter]
    x = 100
    checker = 0
    for x in range(x): 
        if dataset_check[checker][0] == "Elapsed Hours":
           time_pos = checker
           break
        else:
            checker += 1  
            
def determine_title_pos_counts_sil_corrected(inter):
    global counts_sil_corrected_pos
    dataset_check = organised_df_dict[inter]
    x = 100
    checker = 0
    for x in range(x):
        if dataset_check[checker][0] == "Counts Silicon Corrected":
           counts_sil_corrected_pos = checker
           break
        else:
            checker += 1
        
        
#Pull the wanted datasets to complete the Phe Calculations on
def pull_datasets_for_Phe_calc(inter):
    global dataset1, t_values, counts_silicon_corrected_values
    dataset1 = organised_df_dict[inter]
    determine_title_pos_counts_sil_corrected(1)
    determine_title_pos_time(1)
    t_values = np.array(dataset1[time_pos][1:])
    counts_silicon_corrected_values = np.array(dataset1[counts_sil_corrected_pos][1:])
    ##If two arrays need to be conjoined - np.concatenate((array1, array2), axis = None)
    
#Pull the depolarised data and set for the opacity_depolarised
def get_opacity_depol(file):
    global opacity_depolarised
    depolarised_counts_silicon = depolarised_df.loc[1]
    depolarised_counts_silicon = depolarised_counts_silicon[counts_sil_corrected_pos]
    opacity_helium_depol_formula(depolarised_counts_silicon)

#Calcualte the Phe and the Time
def calculate_phe_and_T():
    global Phe0, T, result, min_error_found

    pull_datasets_for_Phe_calc(1)
    get_opacity_depol(1)
    result = differential_evolution(error_nsf_total, bounds = bounds, args=(opacity_depolarised, t_values, counts_silicon_corrected_values))
    Phe0 = result.x[1]
    T = result.x[0]
    min_error_found = result.fun
    #print("Minimum value found at T =", result.x[0], "and Phe0 =", result.x[1])
    #print("Minimum total error:", result.fun)    

#Check for blank databases in the dictionary and ignore them as too many may have been created
def check_for_blank_df_dict():
    global non_blank_dict_count   
    non_blank_dict_count = 0
    num = 0
    for x in range(len(organised_df_dict)):
        if len(organised_df_dict[num]) > 1:
            non_blank_dict_count = non_blank_dict_count + 1
            num = num + 1
        else:
            num = num + 1

#Update the Phe and T sections of all the databases
def update_phe_and_t_in_df():

    check_for_blank_df_dict()
    num = 0
    for x in range(non_blank_dict_count):
        pulled_new_df = organised_df_dict[num]
        Phe_values = np.array(pulled_new_df[11])
        t_values = np.array(pulled_new_df[1])
        trans_calc = np.array(pulled_new_df[12])
        num2 = 1
        num3 = 1
        spin_flip = pulled_new_df[4]
        num = num + 1
        for x in range(len(Phe_values)-1):
            Phe_calc(Phe0, T, t_values[num2])
            Phe_values[num2] = Phe
            num2 = num2 + 1
            if spin_flip[1] == "NSF":
                trans_intensity_nsf(Phe_values[num3])
                trans_calc[num3] = trans_intensity
                num3 = num3 + 1
            else:
                trans_intensity_sf(Phe_values[num3])
                trans_calc[num3] = trans_intensity
                num3 = num3 + 1
        pulled_new_df[11] = Phe_values
        pulled_new_df[12] = trans_calc



#Organise the HKL data from the raw data, then produce a BOOL of if the expected data matches the actual data
def organise_hkl_data():
    global HKL_accept, acceptable_data, HKL, broken_files_str
    h_latticeplane = data_raw_datfile.filter('h')
    k_latticeplane = data_raw_datfile.filter('k')
    l_latticeplane = data_raw_datfile.filter('l')
    h_mean = round(float(h_latticeplane.mean()))
    k_mean = round(float(k_latticeplane.mean()))
    l_mean = round(float(l_latticeplane.mean()))
    inter = 0
    
    
    for x in range(len(variable_2_input)):
            HKL_expect = variable_2_input[inter]
            H_expect = round(float(HKL_expect[0]))
            K_expect = round(float(HKL_expect[1]))
            L_expect = round(float(HKL_expect[2]))
            
            if abs(h_mean) == H_expect and abs(k_mean) == K_expect and abs(l_mean) == L_expect:
                HKL = HKL_expect
                h_accept = True
                k_accept = True
                l_accept = True
            else:
                inter = inter + 1
                h_accept = False
                k_accept = False
                l_accept = False
    
    
    if h_accept == True and k_accept == True and l_accept == True:
        HKL_accept = True
    else:
        HKL_accept = False
        broken_files_str = broken_files_str + (file_name +  " is broken according to HKL matching. ")
       
    if HKL_accept == True:
        acceptable_data = acceptable_data + good_data
    else:
        acceptable_data = acceptable_data + bad_data    

#Organise the Temperature data from the raw data, then produce a BOOL of if the expected data matches the actual data
def organise_temp_data():
    global temp, temp_accept, acceptable_data, broken_files_str
    temp_from_rawdata = data_raw_datfile.filter(like ='tc1a')
    temp_mean = float(temp_from_rawdata.mean())
    inter = 0
    for x in range(len(variable_1_input)):
        temp_expect = float(variable_1_input[inter])  
        temp_difference = temp_mean - temp_expect
        accept_dif = temp_expect * 0.1
        statement = abs(temp_difference) <= accept_dif
        if statement == True:
            temp_accept = True
            temp = temp_expect
            break
        else:
            inter= inter + 1
            temp_accept = False
            
    if temp_accept == True:
        acceptable_data = acceptable_data + good_data
    else:
        acceptable_data = acceptable_data + bad_data
        broken_files_str = broken_files_str + ('Failed on file ' + file_name + " because of temperature matching. ")
        
        
def pull_detector_data_for_analysis():
    global detector_data
    detector_data = data_raw_datfile.filter(like ='count')
    detector_data = detector_data.to_numpy().flatten()

#Organise the sample Rotation (s1) data from the raw data
def sample_rotation_data():
    global sample_rotation
    sample_rotation_rawdata = data_raw_datfile.filter( items = ['s1'])
    sample_rotation = float(sample_rotation_rawdata.mean())
    
#Organise the sample Rotation 2 Theta (s2) data from the raw data
def sample_rotation_2theta_data():
    global sample_rotation_2theta
    sample_rotation_rawdata = data_raw_datfile.filter( items = ['s2'])
    sample_rotation_2theta = float(sample_rotation_rawdata.mean())

#Organise the wave Vector (q) data from the raw data
def wave_vector_data():
    global wave_vector
    wave_vector_rawdata = data_raw_datfile.filter( items = ['q'])
    wave_vector = float(wave_vector_rawdata.mean())

#Define a function that takes the search variables and compares them then organises the data
def compare_data_to_expected():
    global acceptable_data
    acceptable_data = 2
    search_variable = variable_names
    if str(re.findall('Temp',str(search_variable)))[2:-2] == 'Temp':
        organise_temp_data()
        
    if str(re.findall('HKL', str(search_variable)))[2:-2] == 'HKL':
        organise_hkl_data()
        
    if str(re.findall('Sample Rotation', str(search_variable)))[2:-2] == 'Sample Rotatation':
        sample_rotation_data()
        
    if str(re.findall('Sample Rotation 2Theta', str(search_variable)))[2:-2] == 'Sample Rotatation 2Theta':
        sample_rotation_2theta_data()
        
    if str(re.findall('Wave Vector', str(search_variable)))[2:-2] == 'Wave Vector':
        wave_vector_data()


###################Combine Functions #################################
#Combine all the required functions to produce one function to do all tasks - Later can be assigned to a button to do all
def combine_all_functions_multiple_variables():
    load_input_variables_sika()
    organise_data2()
    pull_datasets_for_Phe_calc(1)
    calculate_phe_and_T()
    check_for_blank_df_dict()
    update_phe_and_t_in_df()
    
def combine_functions_pre_QUOKKA(file):
    global time_in_seconds
    load_raw_data_hdf(file)
    load_other_data_hdf_quokka()
    determine_time_hdf()
    elapsed_time()
    organise_data_HDF_non_sorted()

def combine_all_functions_no_variables():
    load_input_variables_sika()
    organise_data_no_variables()

def pull_intensity_data_taipan_dat():
    global intensity, detector_data
    detector_data = data_raw_datfile.filter(like ='detector')
    detector_data = detector_data.to_numpy().flatten()
    plot_data_taipan_dat(detector_data)
    
    
def save_raw_data(save_name):
    non_sorted_df.to_excel(save_name)
    
def organised_dict_to_organised_df():
    global organised_df_base
    organised_df_base = pd.DataFrame([list(titles_df)])
    size_required = organised_df_base.shape
    empty_array = size_required[1] * ['']
    organised_df_base.loc[1] = empty_array
    search_num = 1000
    search_param = 0
    for x in range(search_num):
        search_df_from_dict = organised_df_dict[search_param]
        df_size = search_df_from_dict.shape
        if df_size[0] > 1:
            search_df_from_dict.loc[df_size[0]+1] = empty_array
            combined = [organised_df_base, search_df_from_dict]
            organised_df_base = pd.concat (combined)
            search_param += 1
        elif df_size[0] <= 1:
            break
        
########################Save to Excel Code##############################
def save_organised_data(save_name):
    organised_dict_to_organised_df()
    organised_df_base.to_excel(save_name)

def save_results_only(save_name):
    global results_df
    results = {"Phe0": result.x[1], "T0":result.x[0], "Minimum Squared": result.fun}

    results_df = pd.DataFrame(data=results, index=[0])
    results_df = (results_df.T)
    results_df.to_excel(save_name)

def save_raw_data_and_results(save_name):
    global raw_data_results_df
    results = {"Phe0": result.x[1], "T0":result.x[0], "Minimum Squared": result.fun}

    results_df = pd.DataFrame(data=results, index=[0])
    results_df = (results_df.T)
    combine = [results_df, non_sorted_df]
    raw_data_results_df = pd.concat(combine)
    raw_data_results_df.to_excel(save_name)
    
def save_organised_data_and_results(save_name):
    global organised_data_results_df
    organised_dict_to_organised_df()

    results = {"Phe0": result.x[1], "T0":result.x[0], "Minimum Squared": result.fun}

    results_df = pd.DataFrame(data=results, index=[0])
    results_df = (results_df.T)
    combine = [results_df, organised_df_base]
    organised_data_results_df = pd.concat(combine)
    organised_data_results_df.to_excel(save_name)
    
def cal_error_between_intensity():
    global error
    error = (intensity - popt[0])/popt[0]
#################################################################################################
######################################### PYQT / Interface Code #################################

qtcreator_file  = "testv2_gui_he3.ui" # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)

    
class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.btn_homepage.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_1_home))
        self.btn_options_page.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_2_options))
        self.btn_variables_page.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_3_variables))
        self.btn_parameterspage.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_4_parameters))
        self.btn_view_page.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_5_view))
        self.btn_save_page.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_6_save))
        self.btn_errors_page.clicked.connect(lambda:self.stackedWidget.setCurrentWidget(self.page_7_errors))
        self.btn_display_page.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_8_display_variables))
        self.btn_display_page.clicked.connect(self.update_display_page)
        self.btn_edit_data_page.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_9_edit_data))
        self.btn_update_parameters_page4.clicked.connect(self.update_parameters)
        self.btn_update_variables.clicked.connect(self.find_variables)
        self.btn_reset_pg3.clicked.connect(self.reset_page3_variables)
        self.btn_submit_page1.clicked.connect(self.submit_page_1)
        self.btn_calc_silicon_trans.clicked.connect(self.calculate_silicon_trans)
        self.btn_update_options_pg2.clicked.connect(self.update_options_pg2)
        self.btn_reset_p2.clicked.connect(self.reset_options_pg2)
        self.btn_clear_error_textbrowser.clicked.connect(self.clear_errorsbrowser)
        self.checkbox_dont_calculate_phe_and_time.clicked.connect(self.uncheck_opposite)
        self.checkbox_calculate_phe_and_time.clicked.connect(self.uncheck_opposite2)  
        self.checkbox_sort_values.clicked.connect(self.uncheck_opposite)
        self.checkbox_dont_sort_values.clicked.connect(self.uncheck_opposite2)
        self.checkbox_nsf_polarise_method.clicked.connect(self.uncheck_opposite)
        self.checkbox_sf_polarise_method.clicked.connect(self.uncheck_opposite2)
        self.checkbox_raw_data.clicked.connect(self.uncheck_opposite)
        self.checkbox_organised_data.clicked.connect(self.uncheck_opposite2)
        self.btn_reset_pg8.clicked.connect(self.clear_display_variables)
        self.btn_save_to_csv.clicked.connect(self.save_to_csv)
        
        if str(self.combobox_instrument_select.currentText()) == 'Sika':
            self.stackedWidget_pg_8.setCurrentWidget(self.sika_page)
        
        if str(self.combobox_instrument_select.currentText()) == 'Taipan':
            self.stackedWidget_pg_8.setCurrentWidget(self.taipan_page)
        if str(self.combobox_instrument_select.currentText()) == 'Wombat':
            self.stackedWidget_pg_8.setCurrentWidget(self.wombat_page)
        if str(self.combobox_instrument_select.currentText())== 'Quokka':
            self.stackedWidget_pg8.setCurrentWidget(self.quokka_page)
            
    def update_display_page(self):
        if str(self.combobox_instrument_select.currentText()) == 'Sika':
            self.stackedWidget_pg_8.setCurrentWidget(self.sika_page)
        
        if str(self.combobox_instrument_select.currentText()) == 'Taipan':
            self.stackedWidget_pg_8.setCurrentWidget(self.taipan_page)
        if str(self.combobox_instrument_select.currentText()) == 'Wombat':
            self.stackedWidget_pg_8.setCurrentWidget(self.wombat_page)
        if str(self.combobox_instrument_select.currentText())== 'Quokka':
            self.stackedWidget_pg8.setCurrentWidget(self.quokka_page)
            
    def save_to_csv(self):
        global save_name
        error_message_pg6 = "Error: Values are not loaded and cannot be saved"
        if len(self.textedit_enter_desired_name.toPlainText()) > 1:
            
            desired_name = self.textedit_enter_desired_name.toPlainText()
            save_name = desired_name + ".xlsx" 
        
        elif len(self.textedit_enter_desired_name.toPlainText()) < 1:
            self.textbrowser_pg6.setText("Please enter a name to save the file as")

        
        if self.checkbox_raw_data.isChecked() and self.checkbox_save_results_only.isChecked():
           if 'non_sorted_df' and 'save_name' in globals():
                  try:
                      save_raw_data_and_results(save_name)
                      self.textbrowser_pg6.setText("Saved as " + save_name)
                  except NameError:
                    self.textbrowser_pg6.setText("No Results To Save")

           else:
                self.textbrowser_pg6.setText(error_message_pg6)
                
        if self.checkbox_organised_data.isChecked() and self.checkbox_save_results_only.isChecked():
           if 'non_sorted_df' and 'save_name' in globals():
                  try: 
                      save_organised_data_and_results(save_name)
                      self.textbrowser_pg6.setText("Saved as " + save_name)
                  except NameError: 
                      self.textbrowser_pg6.setText("No Results To Save")

           else:
                self.textbrowser_pg6.setText(error_message_pg6)                


        if self.checkbox_raw_data.isChecked():
            if 'non_sorted_df' and 'save_name' in globals():
                save_raw_data(save_name)
                self.textbrowser_pg6.setText("Saved as " + save_name)
            else:
                self.textbrowser_pg6.setText(error_message_pg6)
        if self.checkbox_organised_data.isChecked():
            if 'titles_df' and 'save_name' in globals():
                save_organised_data(save_name)
                self.textbrowser_pg6.setText("Saved as " + save_name)
            else:
                self.textbrowser_pg6.setText(error_message_pg6)
                
            
        if self.checkbox_save_results_only.isChecked():
            if 'results' and 'save_name' in globals():
                try:
                    save_results_only(save_name)
                    self.textbrowser_pg6.setText("Saved as " + save_name)
                except NameError: 
                    self.textbrowser_pg6.setText("No Results To Save")

            else:
                self.textbrowser_pg6.setText(error_message_pg6)

            
            
    def uncheck_opposite(self):
        if self.checkbox_dont_calculate_phe_and_time.isChecked():
           self.checkbox_calculate_phe_and_time.setChecked(False)
           
        if self.checkbox_sort_values.isChecked():
            self.checkbox_dont_sort_values.setChecked(False)
        
        if self.checkbox_nsf_polarise_method.isChecked():
            self.checkbox_sf_polarise_method.setChecked(False)
        
        if self.checkbox_raw_data.isChecked():
            self.checkbox_organised_data.setChecked(False)
            
    def uncheck_opposite2(self):
        if self.checkbox_calculate_phe_and_time.isChecked():
            self.checkbox_dont_calculate_phe_and_time.setChecked(False)
            
        if self.checkbox_dont_sort_values.isChecked():
            self.checkbox_sort_values.setChecked(False)
        
        if self.checkbox_sf_polarise_method.isChecked():
            self.checkbox_nsf_polarise_method.setChecked(False)
            
        if self.checkbox_organised_data.isChecked():
            self.checkbox_raw_data.setChecked(False)
            
    def update_parameters(self):
        global cell_thickness, pressure, wavelength, beam_mon_count, t_si
        cell_thickness = float(self.textedit_cell_thickness_input.toPlainText())
        pressure = float(self.textedit_pressure_input.toPlainText())
        wavelength = float(self.textedit_wavelength_input.toPlainText())
        beam_mon_count = float(self.textedit_beam_monitor.toPlainText())
        t_si = float(self.textedit_trans_silicon.toPlainText())
        self.textbrowser_pg4.setText("The Cell thickness has been changed to: " + str(cell_thickness) + ". The pressure has been changed to: " + str(pressure)+ ". The wavelength has been changed to: " + str(wavelength))
    
    def update_options_pg2(self):
        global spinflip_state, sort_values, options_statement,depolarised_scans_manual_input ,options_error_message, polarisation_method
        
        for x in range(1):
            options_error_message = ""
            options_statement_warning = ""
            options_statement = "Options Updated: "
            if self.checkbox_depolarised_scan_manual_input.isChecked():
                depolarised_scans_manual_input = self.textedit_depolarised_scan_number.toPlainText()
                depolarised_scans_manual_input = depolarised_scans_manual_input.split(',')
                options_statement += 'The depolarised files have been manually input. '
            if self.checkbox_switch_spin_flips.isChecked():
                spinflip_state = 'flip'
                options_statement += "The spinflip state will be flipped for all loaded files. "
            
            if self.checkbox_sort_values.isChecked() and self.checkbox_dont_sort_values.isChecked():
                options_error_message += "Please only Choose 1 option for the Sorting of the Values"
                self.textbrowser_pg2.append(options_error_message)
                self.textbrowser_errors_page7.append(options_error_message)
            
            elif self.checkbox_sort_values.isChecked():
                sort_values = True
                self.checkbox_dont_sort_values.setChecked(False)
                options_statement += 'The values will be sorted. '
                options_statement_warning = ""
                
            elif self.checkbox_dont_sort_values.isChecked():
                sort_values = False
                self.checkbox_sort_values.setChecked(False)
                options_statement += "The values won't be sorted."
                options_statement_warning += " WARNING- Values will NOT be sorted and no calculations will be done. All other options will be ignored. "
            
            if self.checkbox_nsf_polarise_method.isChecked():
                options_statement += "The non-spin flip polarization method will be used for this data set. "
                polarisation_method = "NSF"
            
            if self.checkbox_sf_polarise_method.isChecked():
                options_statement += "The spin flip polarisation method will be used for this data set. "
                polarisation_method = "SF"
            
            if len(options_error_message) < 2 and len(options_statement_warning) < 2:
                self.textbrowser_pg2.setText(options_statement)
                        
            elif len(options_statement_warning) > 2:
                self.textbrowser_pg2.setText(options_statement_warning)

            else:
                self.textbrowser_pg2.append("Please Check Errors before Proceeding")
            
    def reset_options_pg2(self):
        global sort_values, spinflip_state, polarisation_method
        self.checkbox_sf_polarise_method.setChecked(False)
        polarisation_method = 'NSF'
        self.checkbox_nsf_polarise_method.setChecked(False)
        self.checkbox_dont_sort_values.setChecked(False)
        self.checkbox_sort_values.setChecked(False)
        sort_values = True
        self.checkbox_switch_spin_flips.setChecked(False)
        spinflip_state = 'normal'
        self.textedit_depolarised_scan_number.setText('')
        
        self.textbrowser_pg2.setText("All Options have been reset prior to editing.")
    def clear_errorsbrowser(self):
        self.textbrowser_errors_page7.setText('')
    
    def calculate_silicon_trans(self): 
        calc_t_si(wavelength)
        self.textedit_trans_silicon.setText(str(t_si))
        
    def clear_display_variables(self):
        self.checkbox_beam_monitor.setChecked(False)
        self.checkbox_counts_per_beam_monitor.setChecked(False)
        self.checkbox_counts_relative.setChecked(False)
        self.checkbox_counts_silicon.setChecked(False)
        self.checkbox_elapsed_hrs.setChecked(False)
        self.checkbox_errors_squared.setChecked(False)
        self.checkbox_flip_state.setChecked(False)
        self.checkbox_magnitude.setChecked(False)
        self.checkbox_phe_calc.setChecked(False)
        self.checkbox_scan.setChecked(False)
        self.checkbox_scan_title.setChecked(False)
        self.checkbox_trans_calc.setChecked(False)
        self.textbrowser_pg8.setText("All Display Variables have been reset - Please Make a selection and set again")
    def find_variables(self):
        global variable_names, temp_gui_check,variable_1_input, HKL_gui_check, variable_2_input
        if self.checkbox_temp.isChecked() and temp_gui_check == False:
            variable_names = np.append(variable_names, 'Temp')
            temp_gui_check = True
            variable_1_input = str(self.textedit_temp_values.toPlainText()).split(',')
            
        if self.checkbox_HKL.isChecked() and HKL_gui_check == False:
            variable_names = np.append(variable_names, 'HKL')
            HKL_gui_check = True
            variable_2_input = str(self.textedit_HKL_values.toPlainText()).split(',')
            self.textbrowser_pg3.setText("The variables have been updated in the Database")
        
        if len(self.textedit_HKL_values.toPlainText()) > 1 and not self.checkbox_HKL.isChecked() or len(self.textedit_temp_values.toPlainText()) > 1 and not self.checkbox_temp.isChecked():
            self.textbrowser_pg3.setText("Please select the matching variable for the data")
        create_various_dataframes()

     
    def reset_page3_variables(self):
        self.checkbox_temp.setChecked(False)
        self.checkbox_HKL.setChecked(False)
        self.textedit_HKL_values.clear()
        self.textedit_temp_values.clear()
        self.textbrowser_pg3.setText('Variables Reset')
        
    def submit_page_1(self):
########################### SIKA ##################################
        
        global file_number, experiment, file_range, instrument_type
        if len(self.textedit_first_file.toPlainText()) > 1 and len(variable_names) > 0 and sort_values == True :
            file_number = str(self.textedit_first_file.toPlainText())
            experiment = str(self.textedit_experiment_number.toPlainText())
            file_range = int(self.textedit_file_range.toPlainText())
            if str(self.combobox_instrument_select.currentText()) == 'Sika':
                instrument_type = 'Sika'
                if str(self.combobox_file_type.currentText())=='.dat':
                    
                    combine_all_functions_multiple_variables()
                    display_phe0 = "Minimum value found at T =" +  str(result.x[0]) + "and Phe0 =" + str(result.x[1]) + " with the minimum error being " + str(result.fun)
                    self.textbrowser_pg1.setText(display_phe0)
                    if len(broken_files_str) > 1:
                        self.textbrowser_errors_page7.append(broken_files_str)
                        self.textbrowser_pg1.append (error_message)
                    
        elif len(self.textedit_first_file.toPlainText()) < 10 and len(variable_names) > 0 and sort_values == True :
            file_number = '0' + str(self.textedit_first_file.toPlainText())
            experiment = str(self.textedit_experiment_number.toPlainText())
            file_range = int(self.textedit_file_range.toPlainText())
            if str(self.combobox_instrument_select.currentText()) == 'Sika':
                instrument_type = 'Sika'
                if str(self.combobox_file_type.currentText())=='.dat':
    
                    combine_all_functions_multiple_variables()
                    display_phe0 = "Minimum value found at T =" +  str(result.x[0]) + "and Phe0 =" + str(result.x[1]) + " with the minimum error being " + str(result.fun)
                    self.textbrowser_pg1.setText(display_phe0)
                    if len(broken_files_str) > 1:
                        self.textbrowser_errors_page7.append(broken_files_str)
                        self.textbrowser_pg1.append (error_message) 
         
        elif len(self.textedit_first_file.toPlainText()) > 1 and len(variable_names) > 0 and sort_values == False :
            file_number = str(self.textedit_first_file.toPlainText())
            experiment = str(self.textedit_experiment_number.toPlainText())
            file_range = int(self.textedit_file_range.toPlainText())
            if str(self.combobox_instrument_select.currentText()) == 'Sika':
                instrument_type = 'Sika'
                if str(self.combobox_file_type.currentText())=='.dat':

                    load_input_variables_sika()
                    self.textbrowser_pg1.setText("The data has been Created")

        elif len(self.textedit_first_file.toPlainText()) < 10 and len(variable_names) > 0 and sort_values == False :
            file_number = '0' + str(self.textedit_first_file.toPlainText())
            experiment = str(self.textedit_experiment_number.toPlainText())
            file_range = int(self.textedit_file_range.toPlainText())
            if str(self.combobox_instrument_select.currentText()) == 'Sika':
                instrument_type = 'Sika'
                if str(self.combobox_file_type.currentText())=='.dat':
                    load_input_variables_sika()
                    self.textbrowser_pg1.setText("The data has been Created")
                    
                    
        elif len(self.textedit_first_file.toPlainText()) > 1 and len(variable_names) == 0:
            file_number = str(self.textedit_first_file.toPlainText())
            experiment = str(self.textedit_experiment_number.toPlainText())
            file_range = int(self.textedit_file_range.toPlainText())
            if str(self.combobox_instrument_select.currentText()) == 'Sika':
                instrument_type = 'Sika'
                if str(self.combobox_file_type.currentText())=='.dat':

                    combine_all_functions_no_variables()
                    self.textbrowser_pg1.setText("Dun")
                
        elif len(self.textedit_first_file.toPlainText()) < 10 and len(variable_names) == 0:
            file_number ='0' + str(self.textedit_first_file.toPlainText())
            experiment = str(self.textedit_experiment_number.toPlainText())
            file_range = int(self.textedit_file_range.toPlainText())
            if str(self.combobox_instrument_select.currentText()) == 'Sika':
                instrument_type = 'Sika'
                if str(self.combobox_file_type.currentText())=='.dat':
                    combine_all_functions_no_variables()
                    self.textbrowser_pg1.setText("Dun")

########################### TAIPAN ##################################
        if len(self.textedit_first_file.toPlainText()) > 1 and len(variable_names) > 0 and sort_values == True :
            file_number = str(self.textedit_first_file.toPlainText())
            experiment = str(self.textedit_experiment_number.toPlainText())
            file_range = int(self.textedit_file_range.toPlainText())
            if str(self.combobox_instrument_select.currentText()) == 'Taipan':
                instrument_type = 'Taipan'

                if str(self.combobox_file_type.currentText())=='.dat':
                    load_input_variables_taipan_dat()
                    load_raw_data_taipan_dat
                    #combine_all_functions_multiple_variables()
                    #display_phe0 = "Minimum value found at T =" +  str(result.x[0]) + "and Phe0 =" + str(result.x[1]) + " with the minimum error being " + str(result.fun)
                    #self.textbrowser_pg1.setText(display_phe0)
                if len(broken_files_str) > 1:
                    self.textbrowser_errors_page7.append(broken_files_str)
                    self.textbrowser_pg1.append (error_message)
                    
        elif len(self.textedit_first_file.toPlainText()) > 1 and len(variable_names) == 0:
            file_number = str(self.textedit_first_file.toPlainText())
            experiment = str(self.textedit_experiment_number.toPlainText())
            file_range = int(self.textedit_file_range.toPlainText())
            if str(self.combobox_instrument_select.currentText()) == 'Taipan':
                instrument_type = 'Taipan'
                if str(self.combobox_file_type.currentText())=='.dat':

                    load_input_variables_taipan_dat()
                    load_raw_data_taipan_dat
                    self.textbrowser_pg1.setText("Dun")
                    
########################## Wombat ###############################
        if len(self.textedit_first_file.toPlainText()) > 1 and len(variable_names) > 0 and sort_values == True :
             file_number = str(self.textedit_first_file.toPlainText())
             experiment = str(self.textedit_experiment_number.toPlainText())
             file_range = int(self.textedit_file_range.toPlainText())
             if str(self.combobox_instrument_select.currentText()) == 'Wombat':
                 instrument_type = 'Wombat'
                 
        
             if len(broken_files_str) > 1:
                 self.textbrowser_errors_page7.append(broken_files_str)
                 self.textbrowser_pg1.append (error_message)
                     
        elif len(self.textedit_first_file.toPlainText()) > 1 and len(variable_names) == 0:
             file_number = str(self.textedit_first_file.toPlainText())
             experiment = str(self.textedit_experiment_number.toPlainText())
             file_range = int(self.textedit_file_range.toPlainText())
             if str(self.combobox_instrument_select.currentText()) == 'Taipan':
                 instrument_type = 'Taipan'
                 if str(self.combobox_file_type.currentText())=='.dat':
        
                     load_input_variables_taipan_dat()
                     load_raw_data_taipan_dat
                     self.textbrowser_pg1.setText("Dun")
                     
        

########################## Quokka ##################################
        if len(self.textedit_first_file.toPlainText()) > 1 and len(variable_names) > 0 and sort_values == True :
             file_number = str(self.textedit_first_file.toPlainText())
             experiment = str(self.textedit_experiment_number.toPlainText())
             file_range = int(self.textedit_file_range.toPlainText())
             if str(self.combobox_instrument_select.currentText()) == 'Quokka':
                 instrument_type = 'Quokka'
                 load_input_variables_quokka_HDF()
                 self.textbrowser_pg1.setText("Dun 1")

             if len(broken_files_str) > 1:
                 self.textbrowser_errors_page7.append(broken_files_str)
                 self.textbrowser_pg1.append (error_message)
                     
        elif len(self.textedit_first_file.toPlainText()) > 1 and len(variable_names) == 0:
             file_number = str(self.textedit_first_file.toPlainText())
             experiment = str(self.textedit_experiment_number.toPlainText())
             file_range = int(self.textedit_file_range.toPlainText())
             if str(self.combobox_instrument_select.currentText()) == 'Quokka':
                 instrument_type = 'Quokka'
                 if str(self.combobox_file_type.currentText())=='HDF':
                     load_input_variables_quokka_HDF()
                     self.textbrowser_pg1.setText("Dun 2")
                     
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())