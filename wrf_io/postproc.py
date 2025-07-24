import gc
import re
import os
import math
import time
import glob
import joblib
import pickle
import pprint
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from wrf_io import sweep
from pathlib import Path
from wrf_io import preproc
from scipy.io import loadmat
from multiprocessing import Pool
from rich.console import Console
from numpy.typing import ArrayLike
from matplotlib.gridspec import GridSpec
from typing import Dict, Any, Optional, List, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, Kernel, WhiteKernel, Sum, Product, DotProduct, RationalQuadratic, ConstantKernel as C

def madsen_ref() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a reference curve of axial induction factor (a) versus thrust coefficient (Ct)
    using the polynomial model from Madsen et al. (2020).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - a (np.ndarray): Induction factor values
            - ct (np.ndarray): Thrust coefficient values
    """
    ct = np.linspace(0.0, 1.5, 100)

    k1 = 0.2460
    k2 = 0.0586
    k3 = 0.0883

    a = k3 * ct**3 + k2 * ct**2 + k1 * ct

    return a, ct


def classical_ref() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a reference curve of axial induction factor (a) versus thrust coefficient (Ct)
    using 1D momentum theory.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - a (np.ndarray): Induction factor values
            - ct (np.ndarray): Thrust coefficient values
    """
    a = np.linspace(0,1.0,100)

    ct = 4 * a * (1 - a)

    return a, ct


# def load_train_data(path: str):



def load_wrfout(top_dir: str) -> List[Dict[str, Any]]:
    """
    Load processed WRF output data from a set of .npz files in the specified directory.

    Args:
        top_dir (str): Path to the top-level directory containing WRF output subdirectories.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing data loaded from a .npz file.
    """
    files, cases = get_dirs(top_dir)

    wrf_data: List[Dict[str, Any]] = []
    for file, case in zip(files, cases):
        wrf_data.append(dict(np.load(f"{file}{case}_lite.npz")))

    return wrf_data

def get_dirs(top_dir: str):
    # Look one level deeper for s*_v* entries
    path_template = os.path.join(top_dir, '*', 's*_v*')
    all_matches = glob.glob(path_template)

    # Filter only directories
    dir_list = [path for path in all_matches if os.path.isdir(path)]

    # Extract only the last part of each path (e.g., "s2_v3")
    last_parts = [os.path.basename(path) for path in dir_list]

    # Sort both lists based on folder name
    sorted_pairs = sorted(zip(last_parts, dir_list))
    last_parts, dir_list = zip(*sorted_pairs) if sorted_pairs else ([], [])

    return list(dir_list), list(last_parts)

def rmsd_window(data: ArrayLike, window: int, interval: int) -> ArrayLike:
    """
    Compute the root mean square deviation (RMSD) over a rolling window.

    Args:
        data (Union[np.ndarray, list]): Input data as a 1D array or list.
        window (int): Window size in seconds.
        interval (int): Data sampling interval in seconds.

    Returns:
        np.ndarray: RMSD values computed over the rolling window.
    """
    # Flatten the input data to ensure it's 1D
    data = np.ravel(data)

    # Convert input data to Pandas Series
    data = pd.Series(data)

    window_size = max(1, window // interval)  # Convert seconds to number of samples

    def rmsd(series):
        mean_val = series.mean()
        return np.sqrt(((series - mean_val) ** 2).mean())

    return data.rolling(window=window_size, min_periods=1).apply(rmsd, raw=True)


def convergence(params: Dict[str, Any], D: Optional[float] = None) -> None:
    """
    Generate timeseries plots of power, thrust, CP, and CT for a series of runs

    Args:
        params (Dict): A dictionary of settings
    """

    combs     = sweep.get_combinations(params, D=D)
    formats   = sweep.determine_format(combs)
    casenames = sweep.return_case_strings(combs,formats)

    model = params['rotor_model'].lower() + f'_sweep'

    remove_data = params['exclude_time']
    save_period = params['save_interval']

    save_dir = f"{params['base_dir']}/{model}/figs/convergence"

    os.makedirs(save_dir, exist_ok=True)

    for case in casenames:

        print(f'Working on {case}...')

        case_base_path = f"{params['base_dir']}/{model}/{case}"

        file2read = netCDF4.Dataset(f'{case_base_path}/wrfout_d02_0001-01-01_00_00_00','r',mmap=False) # type: ignore # Read Netcdf-type WRF output file
        file2read.variables.keys()

        # Field variables
        Nt = file2read.variables['Times'].shape[0]

        if(remove_data == 0.0):
            save_period_new = 0.0
        else:
            save_period_new = (remove_data * 60 / save_period) + 1 # first xxx timesteps are not included in analysis
        process_period  = Nt - int(save_period_new) # consider only xxx timesteps in analysis

        Ts = Nt - int(process_period)
        Te = Nt
        Nt = Te - Ts

        # Wind turbine variables
        thrust      = file2read.variables['WTP_THRUST'      ][1:-1,:]
        power_aero  = file2read.variables['WTP_POWER'       ][1:-1,:]
        power_mech  = file2read.variables['WTP_POWER_MECH'  ][1:-1,:]
        torque_aero = file2read.variables['WTP_TORQUE'      ][1:-1,:]
        ct          = file2read.variables['WTP_THRUST_COEFF'][1:-1,:]
        cp          = file2read.variables['WTP_POWER_COEFF' ][1:-1,:]
        v0          = file2read.variables['WTP_V0_FST_AVE'  ][1:-1,:]

        timeseries = np.arange(len(thrust)) * save_period / 60

        fig = plt.figure(figsize=(16, 11))

        gs = GridSpec(4, 3, width_ratios=[1, 1, 1], height_ratios=[2,1,2,1])
        ax1  = fig.add_subplot(gs[0, 0])
        ax2  = fig.add_subplot(gs[0, 1])
        ax3  = fig.add_subplot(gs[0, 2])
        ax4  = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax5  = fig.add_subplot(gs[1, 1], sharex=ax2)
        ax6  = fig.add_subplot(gs[1, 2], sharex=ax3)
        ax7  = fig.add_subplot(gs[2, 0], sharex=ax1)
        ax8  = fig.add_subplot(gs[2, 1], sharex=ax2)
        ax9  = fig.add_subplot(gs[2, 2], sharex=ax3)
        ax10 = fig.add_subplot(gs[3, 0], sharex=ax1)
        ax11 = fig.add_subplot(gs[3, 1], sharex=ax2)
        ax12 = fig.add_subplot(gs[3, 2], sharex=ax3)

        ax1.plot(timeseries,thrust / 1000,linestyle='solid',linewidth=2)
        ax1.set_ylabel(r'Thrust [kN]')
        # ax1.set_ylim([1300,1900])

        ax2.plot(timeseries,torque_aero / 1000,linestyle='solid',linewidth=2)
        ax2.set_ylabel(r'Torque [kN m]')
        # ax2.set_ylim([13000,19000])

        ax3.plot(timeseries,power_aero / 1000,linestyle='solid',linewidth=2,label='aero')
        ax3.plot(timeseries,power_mech / 1000,linestyle='solid',linewidth=2,label='mech')
        ax3.set_ylabel(r'Power [kW]')
        # ax3.set_ylim([6000,10000])
        ax3.legend(loc="upper right", fancybox=True, shadow=False, ncol=3, fontsize=8)

        rmsd = rmsd_window(thrust.filled(np.nan)/1000, 300, 10)

        ax4.plot(timeseries,rmsd,linestyle='solid',linewidth=2)
        ax4.set_yscale('log')
        ax4.set_ylabel(r'5-min RMSD')

        rmsd = rmsd_window(torque_aero.filled(np.nan)/1000, 300, 10)

        ax5.plot(timeseries,rmsd,linestyle='solid',linewidth=2)
        ax5.set_yscale('log')
        # ax5.set_ylabel(r'relative change [-]')

        rmsd_a = rmsd_window(power_aero.filled(np.nan)/1000, 300, 10)
        rmsd_m = rmsd_window(power_mech.filled(np.nan)/1000, 300, 10)

        ax6.plot(timeseries,rmsd_a,linestyle='-',linewidth=2)
        ax6.plot(timeseries,rmsd_m,linestyle='-',linewidth=2)
        ax6.set_yscale('log')
        # ax6.set_ylabel(r'relative change [-]')

        ax7.plot(timeseries,v0,linestyle='solid',linewidth=2)
        ax7.set_ylabel(r'$V_0$ [m/s]')
        ax7.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))

        ax8.plot(timeseries,ct,linestyle='solid',linewidth=2)
        ax8.set_ylabel(r'$C_T$ [-]')
        ax8.set_ylim([0.75,1.1])

        ax9.plot(timeseries,cp,linestyle='solid',linewidth=2)
        ax9.set_ylabel(r'$C_P$ [-]')
        ax9.set_ylim([0.45,0.65])
        ax9.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

        rmsd = rmsd_window(v0.filled(np.nan), 300, 10)

        ax10.plot(timeseries,rmsd,linestyle='solid',linewidth=2)
        ax10.set_yscale('log')
        ax10.set_ylabel(r'5-min RMSD')
        ax10.set_xlabel(r'Time [min]')

        rmsd = rmsd_window(ct.filled(np.nan), 300, 10)

        ax11.plot(timeseries,rmsd,linestyle='solid',linewidth=2)
        ax11.set_yscale('log')
        # ax8.set_ylabel(r'relative change [-]')
        ax11.set_xlabel(r'Time [min]')

        rmsd = rmsd_window(cp.filled(np.nan), 300, 10)

        ax12.plot(timeseries,rmsd,linestyle='solid',linewidth=2)
        ax12.set_yscale('log')
        # ax8.set_ylabel(r'relative change [-]')
        ax12.set_xlabel(r'Time [min]')

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax3.get_xticklabels(), visible=False)
        plt.setp(ax4.get_xticklabels(), visible=False)
        plt.setp(ax5.get_xticklabels(), visible=False)
        plt.setp(ax6.get_xticklabels(), visible=False)
        plt.setp(ax7.get_xticklabels(), visible=False)
        plt.setp(ax8.get_xticklabels(), visible=False)
        plt.setp(ax9.get_xticklabels(), visible=False)

        plt.savefig(f"{save_dir}/{case}.png", bbox_inches="tight", dpi=600)

    print('\nDone.')


def fast_process(file: str, static_args: Dict[str, Any]) -> bool:
    """
    Fast process of WRF output data limited to turbine data

    Args:
        file (str): the path of the WRFout file to process
        static_args (ArrayLike): Azimuthal locations over which to average
    """

    save_period = static_args['save_period'] # in seconds
    remove_data = static_args['remove_data'] # in minutes;  discard first xxx minutes (e.g., ~2 flow-through times)

    diameter   = static_args['diameter']
    dhub       = static_args['dhub']
    Nsct       = static_args['Nsct']
    Nelm       = static_args['Nelm']

    #============================================================================================================
    # Main logic [generally no edits beyond this point]
    #============================================================================================================

    dir_path, _ = os.path.split(file)

    case = os.path.basename(dir_path)

    print(f'\nWorking on {case}...')

    file2read = netCDF4.Dataset(file,'r',mmap=False) # type: ignore # Read Netcdf-type WRF output file
    file2read.variables.keys()

    # Field variables
    dx = file2read.getncattr('DX')
    dy = file2read.getncattr('DY')
    dt = file2read.getncattr('DT')
    Nx = file2read.getncattr('WEST-EAST_PATCH_END_UNSTAG')
    Ny = file2read.getncattr('SOUTH-NORTH_PATCH_END_UNSTAG')
    Nz = file2read.getncattr('BOTTOM-TOP_PATCH_END_UNSTAG')
    Nt = file2read.variables['Times'].shape[0]

    if(remove_data == 0.0):
        save_period_new = 0.0
    else:
        save_period_new = (remove_data * 60 / save_period) + 1 # first xxx timesteps are not included in analysis
    process_period  = Nt - int(save_period_new) # consider only xxx timesteps in analysis

    Ts = Nt - int(process_period)
    Te = Nt
    Nt = Te - Ts

    # Wind turbine variables
    thrust      = file2read.variables['WTP_THRUST'      ][Ts:Te,:]
    power_aero  = file2read.variables['WTP_POWER'       ][Ts:Te,:]
    power_mech  = file2read.variables['WTP_POWER_MECH'  ][Ts:Te,:]
    power_gen   = file2read.variables['WTP_POWER_GEN'   ][Ts:Te,:]
    torque_aero = file2read.variables['WTP_TORQUE'      ][Ts:Te,:]
    ct          = file2read.variables['WTP_THRUST_COEFF'][Ts:Te,:]
    cp          = file2read.variables['WTP_POWER_COEFF' ][Ts:Te,:]
    v0          = file2read.variables['WTP_V0_FST_AVE'  ][Ts:Te,:]
    rotspeed    = file2read.variables['WTP_OMEGA'       ][Ts:Te,:] * (30.0 / np.pi) # convert rad/s to rpm

    # Wind turbine blade-element variables
    f   = (file2read.variables['WTP_F'            ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    fn  = (file2read.variables['WTP_FN'           ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    ft  = (file2read.variables['WTP_FT'           ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    l   = (file2read.variables['WTP_L'            ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    d   = (file2read.variables['WTP_D'            ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    cl  = (file2read.variables['WTP_CL'           ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    cd  = (file2read.variables['WTP_CD'           ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    aoa = (file2read.variables['WTP_ALPHA'        ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    v1  = (file2read.variables['WTP_V1'           ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    bpx = (file2read.variables['WTP_BLADEPOINTS_X'][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    bpy = (file2read.variables['WTP_BLADEPOINTS_Y'][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    bpz = (file2read.variables['WTP_BLADEPOINTS_Z'][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    vrel = (file2read.variables['WTP_VREL'        ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    phi = (file2read.variables['WTP_PHI'          ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)

    u    = (file2read.variables['WTP_U'           ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    v    = (file2read.variables['WTP_V'           ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    w    = (file2read.variables['WTP_W'           ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    vtan = (file2read.variables['WTP_VT'          ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)

    file2read.close()

    rhub = dhub/2
    dist = 0.0
    dr = np.zeros(Nelm)
    for i in range(0,Nelm):
        dist = dist + 0.5*((diameter/2 - rhub)/Nelm)
        dr[i] = rhub + dist
        dist = dist + 0.5*((diameter/2 - rhub)/Nelm)

    rOverR = dr/(diameter/2)

    var_holder = {}

    var_holder['diameter']     = diameter
    var_holder['radius']       = diameter/2
    var_holder['hub_diameter'] = dhub
    var_holder['hub_radius']   = dhub/2
    var_holder['hub_height']  = static_args['hub_height']
    var_holder['rOverR']      = rOverR
    var_holder['dx']          = dx
    var_holder['dy']          = dy
    var_holder['dz']          = static_args['dz']

    var_holder['dt']          = dt
    var_holder['Nx']          = Nx
    var_holder['Ny']          = Ny
    var_holder['Nz']          = Nz
    var_holder['tower_xloc']  = static_args['tower_xloc']
    var_holder['tower_yloc']  = static_args['tower_yloc']
    var_holder['Nsct']        = Nsct
    var_holder['Nelm']        = Nelm

    var_holder['uinf']        = static_args['uinf']
    var_holder['omega']       = rotspeed
    var_holder['thrust']      = thrust
    var_holder['power_aero']  = power_aero
    var_holder['power_mech']  = power_mech
    var_holder['power_gen']   = power_gen
    var_holder['torque_aero'] = torque_aero
    var_holder['ct']          = ct
    var_holder['cp']          = cp
    var_holder['v0']          = v0
    var_holder['f']           = f
    var_holder['fn']          = fn
    var_holder['ft']          = ft
    var_holder['l']           = l
    var_holder['d']           = d
    var_holder['cl']          = cl
    var_holder['cd']          = cd
    var_holder['aoa']         = aoa
    var_holder['v1']          = v1

    var_holder['u']           = u
    var_holder['v']           = v
    var_holder['w']           = w
    var_holder['v_tan']       = vtan

    var_holder['vrel']        = vrel
    var_holder['phi']         = phi

    var_holder['bpx']         = bpx
    var_holder['bpy']         = bpy
    var_holder['bpz']         = bpz

    np.savez( os.path.join(f'{dir_path}/{case}_lite.npz'),**var_holder)
    
    del var_holder

    return True


def full_process(file: str, static_args: Dict[str, Any]) -> bool:
    """
    Large process of WRF output data including sampling u, v, w, and p fields.

    Args:
        file (str): the path of the WRFout file to process
        static_args (ArrayLike): Azimuthal locations over which to average
    """

    save_period = static_args['save_period'] # in seconds
    remove_data = static_args['remove_data'] # in minutes;  discard first xxx minutes (e.g., ~2 flow-through times)

    diameter   = static_args['diameter']
    dhub       = static_args['dhub']
    Nsct       = static_args['Nsct']
    Nelm       = static_args['Nelm']

    #============================================================================================================
    # Main logic [generally no edits beyond this point]
    #============================================================================================================

    dir_path, _ = os.path.split(file)

    case = os.path.basename(dir_path)

    print(f'\nWorking on {case}...')

    file2read = netCDF4.Dataset(file,'r',mmap=False) # type: ignore # Read Netcdf-type WRF output file
    file2read.variables.keys()

    # Field variables
    dx = file2read.getncattr('DX')
    dy = file2read.getncattr('DY')
    dt = file2read.getncattr('DT')
    Nx = file2read.getncattr('WEST-EAST_PATCH_END_UNSTAG')
    Ny = file2read.getncattr('SOUTH-NORTH_PATCH_END_UNSTAG')
    Nz = file2read.getncattr('BOTTOM-TOP_PATCH_END_UNSTAG')
    Nt = file2read.variables['Times'].shape[0]

    if(remove_data == 0.0):
        save_period_new = 0.0
    else:
        save_period_new = (remove_data * 60 / save_period) + 1 # first xxx timesteps are not included in analysis
    process_period  = Nt - int(save_period_new) # consider only xxx timesteps in analysis

    Ts = Nt - int(process_period)
    Te = Nt
    Nt = Te - Ts

    # # Wind turbine variables
    thrust      = file2read.variables['WTP_THRUST'      ][Ts:Te,:]
    power_aero  = file2read.variables['WTP_POWER'       ][Ts:Te,:]
    power_mech  = file2read.variables['WTP_POWER_MECH'  ][Ts:Te,:]
    power_gen   = file2read.variables['WTP_POWER_GEN'   ][Ts:Te,:]
    torque_aero = file2read.variables['WTP_TORQUE'      ][Ts:Te,:]
    ct          = file2read.variables['WTP_THRUST_COEFF'][Ts:Te,:]
    cp          = file2read.variables['WTP_POWER_COEFF' ][Ts:Te,:]
    v0          = file2read.variables['WTP_V0_FST_AVE'  ][Ts:Te,:]
    rotspeed    = file2read.variables['WTP_OMEGA'       ][Ts:Te,:] * (30.0 / np.pi) # convert rad/s to rpm
    rotorApex_x = file2read.variables['WTP_ROTORAPEX_X' ][Ts:Te,:]
    rotorApex_y = file2read.variables['WTP_ROTORAPEX_Y' ][Ts:Te,:]
    rotorApex_z = file2read.variables['WTP_ROTORAPEX_Z' ][Ts:Te,:]

    # # Wind turbine blade-element variables
    f   = (file2read.variables['WTP_F'            ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    fn  = (file2read.variables['WTP_FN'           ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    ft  = (file2read.variables['WTP_FT'           ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    l   = (file2read.variables['WTP_L'            ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    d   = (file2read.variables['WTP_D'            ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    cl  = (file2read.variables['WTP_CL'           ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    cd  = (file2read.variables['WTP_CD'           ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    aoa = (file2read.variables['WTP_ALPHA'        ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    v1  = (file2read.variables['WTP_V1'           ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    bpx = (file2read.variables['WTP_BLADEPOINTS_X'][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    bpy = (file2read.variables['WTP_BLADEPOINTS_Y'][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    bpz = (file2read.variables['WTP_BLADEPOINTS_Z'][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    vrel = (file2read.variables['WTP_VREL'        ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    phi = (file2read.variables['WTP_PHI'          ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)

    u_WTP = (file2read.variables['WTP_U'              ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    v_WTP = (file2read.variables['WTP_V'              ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    w_WTP = (file2read.variables['WTP_W'              ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)

    var_holder = {}

    tower_xloc  = static_args['tower_xloc']
    tower_yloc  = static_args['tower_yloc']
    hub_height  = static_args['hub_height']

    ###########################################################################
    rotor_xloc = np.mean(rotorApex_x)                 # Rotor x-position in meters
    rotor_yloc = np.mean(rotorApex_y)                 # Rotor y-position in meters
    rotor_zloc = np.mean(rotorApex_z)                 # Rotor z-position in meters
    ###########################################################################
    distances = {f"dist_{i}D": rotor_xloc + (i * diameter) for i in range(0, static_args['sample_distances'] + 1)}

    lat_distances = {
        f"lat_{i}": int(np.floor((dist + (0.5 * dx)) / dx))
        for i, dist in distances.items()
    }

    nrow_vect = np.arange(0,Nx)
    ncol_vect = np.arange(0,Ny)
    neta_vect = np.arange(0,Nz)

# ================================================================================================================================
    # u-velocity component at different downstream locations, y-z plots:
    u   = file2read.variables['U'][Ts:Te,:,:,:]
    u4d = 0.5*(  u[:,:,:,nrow_vect] +   u[:,:,:,nrow_vect+1] ) # x-component of wind speed in 4d
    del u
    gc.collect()

    var_holder.update({
        f"ux_{i}D": u4d[:, :, :, lat_dist] +
        (u4d[:, :, :, lat_dist + 1] - u4d[:, :, :, lat_dist]) * (distances[f"{i.replace('lat_', '')}"] - lat_dist * dx) / dx
        for i, lat_dist in lat_distances.items()
    })

    del u4d
    gc.collect()

# ================================================================================================================================
    # v-velocity component at different downstream locations, y-z plots:
    v   = file2read.variables['V'][Ts:Te,:,:,:]
    v4d = 0.5*(  v[:,:,ncol_vect,:] +   v[:,:,ncol_vect+1,:] ) # y-component of wind speed in 4d
    del v
    gc.collect()

    var_holder.update({
        f"vx_{i}D": v4d[:, :, :, lat_dist] +
        (v4d[:, :, :, lat_dist + 1] - v4d[:, :, :, lat_dist]) * (distances[f"{i.replace('lat_', '')}"] - lat_dist * dx) / dx
        for i, lat_dist in lat_distances.items()
    })

    del v4d
    gc.collect()

# ================================================================================================================================
    # v-velocity component at different downstream locations, y-z plots:
    w   = file2read.variables['W'][Ts:Te,:,:,:]
    w4d = 0.5*(  w[:,neta_vect,:,:] +   w[:,neta_vect+1,:,:] ) # z-component of wind speed in 4d
    del w
    gc.collect()

    var_holder.update({
        f"wx_{i}D": w4d[:, :, :, lat_dist] +
        (w4d[:, :, :, lat_dist + 1] - w4d[:, :, :, lat_dist]) * (distances[f"{i.replace('lat_', '')}"] - lat_dist * dx) / dx
        for i, lat_dist in lat_distances.items()
    })

    del w4d
    gc.collect()

# ================================================================================================================================
    # pressure at different downstream locations, y-z plots:

    p   = file2read.variables['P'][Ts:Te,:,:,:]
    pb  = file2read.variables['PB'][Ts:Te,:,:,:]

    p4d = p + pb # total pressure in Pa in 4d (perturbation pressure + base state pressure)
    del p,pb
    gc.collect()

    var_holder.update({
        f"p_{i}D": p4d[:, :, :, lat_dist] +
        (p4d[:, :, :, lat_dist + 1] - p4d[:, :, :, lat_dist]) * (distances[f"{i.replace('lat_', '')}"] - lat_dist * dx) / dx
        for i, lat_dist in lat_distances.items()
    })

    del p4d
    gc.collect()

    file2read.close()

    ###########################################################################
    rhub = dhub/2
    dist = 0.0
    dr = np.zeros(Nelm)
    for i in range(0,Nelm):
        dist = dist + 0.5*((diameter/2 - rhub)/Nelm)
        dr[i] = rhub + dist
        dist = dist + 0.5*((diameter/2 - rhub)/Nelm)

    rOverR = dr/(diameter/2)

    var_holder['diameter']     = diameter
    var_holder['radius']       = diameter/2
    var_holder['hub_diameter'] = dhub
    var_holder['hub_height']  = static_args['hub_height']
    var_holder['rOverR']      = rOverR
    var_holder['dx']          = dx
    var_holder['dy']          = dy
    var_holder['dt']          = dt
    var_holder['Nx']          = Nx
    var_holder['Ny']          = Ny
    var_holder['Nz']          = Nz
    var_holder['tower_xloc']  = static_args['tower_xloc']
    var_holder['tower_yloc']  = static_args['tower_yloc']
    var_holder['Nsct']        = Nsct
    var_holder['Nelm']        = Nelm

    var_holder['uinf']        = static_args['uinf']
    var_holder['omega']       = rotspeed
    var_holder['thrust']      = thrust
    var_holder['power_aero']  = power_aero
    var_holder['power_mech']  = power_mech
    var_holder['power_gen']   = power_gen
    var_holder['torque_aero'] = torque_aero
    var_holder['ct']          = ct
    var_holder['cp']          = cp
    var_holder['v0']          = v0
    var_holder['f']           = f
    var_holder['fn']          = fn
    var_holder['ft']          = ft
    var_holder['l']           = l
    var_holder['d']           = d
    var_holder['cl']          = cl
    var_holder['cd']          = cd
    var_holder['aoa']         = aoa
    var_holder['v1']          = v1

    var_holder['u']           = u_WTP
    var_holder['v']           = v_WTP
    var_holder['w']           = w_WTP

    var_holder['vrel']        = vrel
    var_holder['phi']         = phi

    var_holder['bpx']         = bpx
    var_holder['bpy']         = bpy
    var_holder['bpz']         = bpz

    var_holder['trb_x']       = rotor_xloc
    var_holder['trb_y']       = rotor_yloc
    var_holder['trb_z']       = rotor_zloc

    np.savez( os.path.join(f'{dir_path}/{case}_full.npz'),**var_holder)
    
    del var_holder

    return True


def parproc(processes: int, params: Dict[str, Any], procType: str) -> None:
    """
    Processes the WRF output files in parallel

    Args:
        processes (int): The number of parallel processes to use
        opt_params (Dict): A dictionary of settings including sample locations if desired
        procType (str): Tell the function to do a fast process or full process
    """
    full_base_path = params['base_dir'] + params['rotor_model'].lower() + f'_sweep'

    # filelist = glob.glob(params['base_dir'] + '/**/wrfout_d02_0001-01-01_00_00_00', recursive=True)
    filelist, _ = get_dirs(full_base_path)

    sample_namelist = filelist[0] + '/namelist.input'
    sample_turbdb = os.path.join(filelist[0], 'windTurbines', params['turb_model'], 'turbineProperties.tbl')
    sample_turbloc = filelist[0] + '/windturbines-ij.dat'

    console = Console()

    output_lines = []
    output_lines.append('Preparing to process these files:\n')

    # Print files to be processed
    for i in range(len(filelist)):
        # Extract the part after 's<digit>_v<something>/', dynamically
        prefix = filelist[i].split('/')[-2]  # Get the second-to-last part
        output_lines.append(f"{i+1:2}. {prefix}/{os.path.basename(filelist[i]).split(prefix)[-1]}")

    console.print("\n".join(output_lines), highlight=False)
    console.print(f"Processing WRF outputs with [bold][bright_red]{processes}[/bright_red][/bold] parallel processes...")

    namelist, turbprops = preproc.load_variables(preproc.parse_namelist(params,sample_namelist), preproc.parse_turbine_properties(params,sample_turbdb),preproc.parse_turbine_location(params,sample_turbloc))

    static_args = {}

    static_args['save_period']      = params['save_interval']
    static_args['remove_data']      = params['exclude_time']
    static_args['diameter']         = turbprops.turb_diameter
    static_args['dhub']             = turbprops.hub_diameter
    static_args['Nsct']             = namelist.nSections
    static_args['Nelm']             = namelist.nElements
    static_args['hub_height']       = turbprops.hubheight
    static_args['tower_xloc']       = turbprops.turb_x
    static_args['tower_yloc']       = turbprops.turb_y
    static_args['dz']               = namelist.e_vert
    static_args['uinf']             = params['Ufst']
    static_args['sample_distances'] = params['slice_loc']

    # Process the files
    start_time = time.time()
    
    if procType == 'fast':

        # Parallel pool
        with Pool(processes) as pool:
            pool.starmap(fast_process, [(file, static_args) for file in filelist])

    elif procType == 'full':  

        # Parallel pool
        with Pool(processes) as pool:
            pool.starmap(full_process, [(file, static_args) for file in filelist])

    else:
        print(f"Error: Invalid procType '{procType}'. Expected 'fast' or 'full'.")
        return False

    end_time = time.time()

    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    console.print(f"Finished in [bold][green3]{minutes}[/green3][/bold] min and [bold][green3]{seconds}[/green3][/bold] sec.")

def annulus_average(f: ArrayLike, theta: ArrayLike) -> ArrayLike:
    """
    Compute annulus average of spatial quantity f(r,theta) over rotor

    Args:
        f (ArrayLike): Spatial quantity in (r, theta)
        theta (ArrayLike): Azimuthal locations over which to average
    
    Returns:
        ArrayLike: array of annulus averages
    """

    dtheta  = np.gradient(theta)
    weights = dtheta / np.sum(dtheta)
    X_azim  = np.sum(f * weights, axis=1)

    return X_azim

def annulus_avg_to_rotor_avg(f: ArrayLike, r: ArrayLike) -> ArrayLike:
    """
    Compute rotor average of annulus-averaged spatial quantity f(r) over rotor

    Args:
        f (ArrayLike): Annulus-averaged spatial quantity in (r)
        theta (ArrayLike): Azimuthal locations over which to average
    
    Returns:
        ArrayLike: array of annulus averages
    """

    dr = np.gradient(r)
    area_elements = 2 * np.pi * r * dr  # differential area for annular rings
    integrand = f * area_elements

    A = np.pi * (1 - r[0]**2)  # total area of annulus from R1 to R2

    X_rotor =  np.sum(integrand) / A

    return X_rotor

def rotor_average(f: ArrayLike, r: ArrayLike, theta: ArrayLike) -> float:
    """
    Compute rotor average of spatial quantity f(r,theta) over rotor

    Args:
        f (ArrayLike): Spatial quantity in (r,theta)
        r (ArrayLike): Normalized radial points over rotor [0,1]
        theta (ArrayLike): Azimuthal points over rotor
    
    Returns:
        float: rotor-averaged quantity
    """

    dr     = np.gradient(r)
    dtheta = np.gradient(theta)

    area_elements = np.outer(dr * r, dtheta)
    integrand     = f * area_elements

    A = np.pi * (1 - r[0]**2)

    X_rotor = np.sum(integrand) / A

    return X_rotor

def shapiro_correction(ind: ArrayLike, r: ArrayLike, theta: ArrayLike, delta: float, R: float) -> float:
    """
    Compute Shapiro correction a posteriori.

    Args:
        ind (ArrayLike): Local induction in polar coordinates r and theta
        r (ArrayLike): Radial points over rotor
        theta (ArrayLike): Azimuthal points over rotor
        delta (float): delta = a * sqrt(dx^2 + dy^2 + dz^2)
        R (float): Turbine radius
    
    Returns
        float: correction factor
    """

    a_bar = rotor_average(ind, r, theta)

    CT_prime_bar = 4 * a_bar / (1 - a_bar)

    M = (1 + CT_prime_bar/4 * 1/(np.sqrt(np.pi * 3)) * delta/R)**(-1)

    return M

def per_error(A: float, E: float) -> float:
    """
    Returns the error between from reference value A and another value E

    Args:
        A (float): Comparison value
        E (float): Reference value
    
    Returns:
        float: percent error [%]
    """

    A = np.asarray(A)
    E = np.asarray(E)

    with np.errstate(divide='ignore', invalid='ignore'):
        error = np.where(E == 0, np.inf, ((A - E) / E) * 100)

    return error if error.shape != () else error.item()


def extract_sounding(params: Dict[str, Any], local: bool, path: Optional[str] = None) -> Dict[float, Any]:
    """
    Extracts u and v velocity components from wrf sounding file

    Args:
        file_list (list): List of file paths to sounding files
    """

    if local:
        files, _ = get_dirs(path)
    else:
        files, _ = get_dirs(params['base_dir'])

    file_list = [path + '/input_sounding' for path in files]

    data_dict = {}
    
    for file in file_list:
        with open(file, 'r') as f:
            lines = f.readlines()
            
            # Skip the header line and extract columns 1, 4, and 5
            data = np.loadtxt(lines[1:])[:, [0, 3, 4]]

            key = Path(file).parent.name
            data_dict[key] = data
    
    return data_dict


def rotGlobalToLocal(Nelm,Nsct,u_rotor,v_rotor,w_rotor):

    precone = 0
    tilt    = 0
    trbYaw  = 0

    psi = 0.0
    angle = 2 * np.pi / Nsct

    Ux   = np.zeros_like(u_rotor,dtype='float')
    Utau = np.zeros_like(u_rotor,dtype='float')
    Ur   = np.zeros_like(u_rotor,dtype='float')

    for i in range(Nelm):
        for j in range(Nsct):
            transposePrecone = np.array([[np.cos(precone), 0, np.sin(precone)],[0,1,0],[-np.sin(precone), 0, np.cos(precone)]])
            transposeAzimuth = np.array([[1,0,0],[0,np.cos(psi),np.sin(psi)],[0,-np.sin(psi),np.cos(psi)]])
            transposeTilt    = np.array([[np.cos(tilt), 0, -np.sin(tilt)],[0,1,0],[np.sin(tilt), 0, np.cos(tilt)]])
            transposeYaw     = np.array([[np.cos(trbYaw), np.sin(trbYaw), 0],[-np.sin(trbYaw), np.cos(trbYaw), 0],[0,0,1]])

            psi = psi + angle

            PreconeAzimuth = np.matmul(transposePrecone, transposeAzimuth)
            PreconeAzimuthTilt = np.matmul(PreconeAzimuth, transposeTilt)
            PreconeAzimuthTiltYaw = np.matmul(PreconeAzimuthTilt, transposeYaw)

            local = np.matmul(PreconeAzimuthTiltYaw, np.array([[u_rotor[i,j]], [v_rotor[i,j]], [w_rotor[i,j]]]))

            Ux[i,j]   = local[0][0]
            Utau[i,j] = local[1][0]
            Ur[i,j]   = local[2][0]

    return Ux, Utau, Ur


def load_params(path: str) -> Dict[str, Any]:

    with open(path, 'rb') as f:
        opt_params = pickle.load(f)

    max_key_len = max(len(str(key)) for key in opt_params)

    output_lines = []
    for key, value in opt_params.items():
        output_lines.append(f"{key:<{max_key_len}} : {value}")

    print("\n".join(output_lines))

    return opt_params


def load_data(params: Dict[str, Any], casenames: Dict[str, Any], local: bool, path: Optional[str] = None) -> Dict[str, Any]:

    model_str = params['rotor_model'].lower() + f'_sweep'

    if local:
        base = path
    else:
        base = params['base_dir']

    wrfles_bem = []
    for i in range(len(casenames)):
        wrfles_bem.append(dict(np.load(os.path.join(base, model_str, casenames[i]+'_lite.npz'))))

    return wrfles_bem

# ------------------------ GP search functions

def build_kernel_from_search(params, noise: bool):

    file_path = params['base_path'] + params['kernel_file']
    # Extract kernel line
    kernel_line = extract_kernel_line(file_path)
    print("Kernel line:")
    print(kernel_line)

    # Extract noise
    noise_log = extract_noise_parameter(file_path)
    print("\nNoise:")
    print(noise_log)

    # Extract the kernel string argument (remove wrapper like k_opt=..., nll=)
    kernel_str_match = re.search(r'k_opt=(.+\)), nll=', kernel_line)
    kernel_str = kernel_str_match.group(1)
    print("\nKernel specification string:")
    print(kernel_str)

    # Parse
    parsed_kernel = parse_kernel_expr(kernel_str)
    print("\nKernel structure:")
    pprint.pprint(parsed_kernel)

    if noise is False:
        noise_log = None

    # Build
    my_kernel_obj = build_sklearn_kernel(parsed_kernel, noise_log=noise_log)
    print("\nFinal Scikit-learn kernel:")
    print(my_kernel_obj)

    save_directory = Path(file_path).parent
    temp_filename = Path(file_path).name

    # Strip suffix and append new one
    obj_filename = temp_filename.replace("_result.txt", "_obj.pkl")

    save_path = save_directory / obj_filename

    joblib.dump(my_kernel_obj, save_path)
    print(f"\nKernel saved at: {save_path}")

    gpr = GaussianProcessRegressor(kernel=my_kernel_obj, optimizer=None, normalize_y=False)

    return my_kernel_obj, gpr

def extract_kernel_line(file_path: str) -> str:
    with open(file_path, 'r') as file:
        for line in file:
            if "Kernel(" in line:
                return line.strip()
    raise ValueError("No kernel specification found in the file.")

def extract_noise_parameter(file_path: str) -> float:
    with open(file_path, 'r') as file:
        for line in file:
            if "Kernel(" in line or "ScoredKernel(" in line:
                match = re.search(r'noise=\[([-0-9eE\.\+]+)\]', line)
                if match:
                    return float(match.group(1))
    raise ValueError("No noise parameter found in the file.")

def parse_kernel_expr(expr: str) -> dict:
    expr = expr.strip()
    # SumKernel([ ... ])
    if expr.startswith('SumKernel(['):
        inner = expr[len('SumKernel(['):-1]
        children = split_kernels(inner)
        return {'type': 'sum', 'children': [parse_kernel_expr(child) for child in children]}
    # ProductKernel([ ... ])
    elif expr.startswith('ProductKernel(['):
        inner = expr[len('ProductKernel(['):-1]
        children = split_kernels(inner)
        return {'type': 'product', 'children': [parse_kernel_expr(child) for child in children]}
    # MaskKernel(ndim=..., active_dimension=..., base_kernel=...)
    elif expr.startswith('MaskKernel('):
        ndim = int(re.search(r'ndim=(\d+)', expr).group(1))
        active_dimension = int(re.search(r'active_dimension=(\d+)', expr).group(1))
        # Find base_kernel= using parenthesis-matching (since it's not always at the end)
        m = re.search(r'base_kernel=', expr)
        if not m:
            raise ValueError(f"base_kernel= not found in MaskKernel: {expr}")
        start = m.end()
        # Starting from start, extract kernel expr up to matching parentheses.
        depth = 0
        i = start
        # Find where the inner kernel ends
        while i < len(expr):
            c = expr[i]
            if c == '(':
                depth += 1
            elif c == ')':
                if depth == 0:
                    break
                depth -= 1
            i += 1
        base_kernel_str = expr[start:i]
        return {
            'type': 'mask',
            'ndim': ndim,
            'active_dimension': active_dimension,
            'base_kernel': parse_kernel_expr(base_kernel_str.strip())
        }
    # SqExpKernel(lengthscale=..., output_variance=...)
    elif expr.startswith('SqExpKernel('):
        params = dict(re.findall(r'(\w+)=([-\d\.]+)', expr))
        return {
            'type': 'sqexp', 
            'lengthscale': float(params['lengthscale']),
            'variance': float(params['output_variance'])
        }
    # SqExpPeriodicKernel(lengthscale=..., period=..., output_variance=...)
    elif expr.startswith('SqExpPeriodicKernel('):
        params = dict(re.findall(r'(\w+)=([-\d\.]+)', expr))
        return {
            'type': 'sqexpperiodic', 
            'lengthscale': float(params['lengthscale']),
            'period': float(params['period']),
            'variance': float(params['output_variance'])
        }
    # LinKernel(offset=..., lengthscale=..., location=...)
    elif expr.startswith('LinKernel('):
        params = dict(re.findall(r'(\w+)=([-\d\.]+)', expr))
        return {
            'type': 'lin',
            'offset': float(params['offset']),
            'lengthscale': float(params['lengthscale']),
            'location': float(params['location'])
        }
    elif expr.startswith('RQKernel('):
        params = dict(re.findall(r'(\w+)=([-\d\.]+)', expr))
        return {
            'type': 'rq',
            'lengthscale': float(params['lengthscale']),
            'variance': float(params['output_variance']),
            'alpha': float(params['alpha'])
        }
    else:
        raise ValueError(f"Unknown kernel: {expr}")

def split_kernels(s: str) -> List[str]:
    children = []
    depth = 0
    last = 0
    for i, c in enumerate(s):
        if c in '([':
            depth += 1
        elif c in ')]':
            depth -= 1
        elif c == ',' and depth == 0:
            children.append(s[last:i].strip())
            last = i+1
    children.append(s[last:].strip())
    return [child for child in children if child]

class MaskedKernel(Kernel):
    def __init__(self, base_kernel: Kernel, active_dims: list):
        self.base_kernel = base_kernel
        self.active_dims = active_dims  # Always keep as list, not np.array

    def __call__(self, X, Y=None, eval_gradient=False):
        X_sub = X[:, self.active_dims]
        Y_sub = Y[:, self.active_dims] if Y is not None else None
        return self.base_kernel(X_sub, Y_sub, eval_gradient=eval_gradient)

    def diag(self, X):
        return self.base_kernel.diag(X[:, self.active_dims])

    def is_stationary(self):
        return self.base_kernel.is_stationary()

    def __repr__(self):
        return f"MaskedKernel({repr(self.base_kernel)}, active_dims={self.active_dims})"

    def get_params(self, deep=True):
        # This is key for scikit-learn compatibility!
        return {'base_kernel': self.base_kernel, 'active_dims': self.active_dims}

def build_sklearn_kernel(parsed: dict, noise_log: Optional[float] = None) -> Kernel:
    if parsed['type'] == 'sum':
        result = None
        for child in parsed['children']:
            term = build_sklearn_kernel(child)
            result = term if result is None else result + term
    elif parsed['type'] == 'product':
        result = None
        for child in parsed['children']:
            term = build_sklearn_kernel(child)
            result = term if result is None else result * term
    elif parsed['type'] == 'mask':
        active_dim = [parsed['active_dimension']]
        base = build_sklearn_kernel(parsed['base_kernel'])
        result = MaskedKernel(base, active_dim)
    elif parsed['type'] == 'sqexp':
        lengthscale = np.exp(parsed['lengthscale'])
        variance = np.exp(parsed['variance'])
        result = C(variance) * RBF(length_scale=lengthscale)
    elif parsed['type'] == 'sqexpperiodic':
        lengthscale = np.exp(parsed['lengthscale'])
        period = np.exp(parsed['period'])
        variance = np.exp(parsed['variance'])
        result = C(variance) * RBF(length_scale=lengthscale) * ExpSineSquared(length_scale=lengthscale, periodicity=period)
    elif parsed['type'] == 'lin':
        # Use offset as sigma_0 -- exponential because that's how your other params are handled
        sigma_0 = np.exp(parsed['offset'])
        result = DotProduct(sigma_0=sigma_0)
    elif parsed['type'] == 'rq':
        lengthscale = np.exp(parsed['lengthscale'])
        variance = np.exp(parsed['variance'])
        alpha = np.exp(parsed['alpha'])
        result = C(variance) * RationalQuadratic(length_scale=lengthscale, alpha=alpha)
    else:
        raise ValueError(f"Unknown kernel type: {parsed['type']}")

    # If noise_log is provided, add WhiteKernel
    if noise_log is not None:
        noise_level = np.exp(noise_log)
        result = result + WhiteKernel(noise_level=noise_level)
    return result

def set_active_dims(kernel: Kernel, active_dims: List[int]) -> Kernel:
    """Helper to patch active_dims attribute on kernels for scikit-learn ≤1.2 compatibility."""
    try:
        return kernel.clone_with_theta(kernel.theta, active_dims=active_dims)
    except Exception:
        kernel.active_dims = active_dims  # For kernels where .active_dims exists
        return kernel

def extract_data(params, transform: bool):

    train_path = params['base_path'] + params['train_data']
    test_path  = params['base_path'] + params['test_data']

    X_train_mat = loadmat(train_path)
    X_test_mat  = loadmat(test_path)

    X_train = X_train_mat['X']
    y_train = X_train_mat['y'].ravel()

    X_test = X_test_mat['X']
    y_test = X_test_mat['y'].ravel()

    return X_train, y_train, X_test, y_test

def load_scalars(params):

    with open(params['base_path'] + params['cot_scalar'], 'rb') as f:
        scalar_cot = pickle.load(f)

    with open(params['base_path'] + params['shear_scalar'], 'rb') as f:
        scalar_shear = pickle.load(f)

    with open(params['base_path'] + params['veer_scalar'], 'rb') as f:
        scalar_veer = pickle.load(f)

    with open(params['base_path'] + params['ind_scalar'], 'rb') as f:
        scalar_ind = pickle.load(f)

    with open(params['base_path'] + params['enc_scalar'], 'rb') as f:
        scalar_enc = pickle.load(f)

    return scalar_cot, scalar_ind, scalar_shear, scalar_veer, scalar_enc

def gpr_predict(params, gpr, set: str, std: bool):

    if set == 'test':
        X_train, y_train, X_test, _ = extract_data(params, transform = False)

    gpr.fit(X_train, y_train)

    A_pred, sigma = gpr.predict(X_test, return_std=True)

    _, ind_scalar, _, _, _ = load_scalars(params)

    a_pred = ind_scalar.inverse_transform(A_pred.reshape(-1, 1))

    if std:
        return a_pred, sigma
    else:
        return a_pred