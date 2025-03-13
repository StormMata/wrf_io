import gc
import os
# import wrf
import time
import glob
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from wrf_io import sweep
from pathlib import Path
from wrf_io import preproc
from typing import Dict, Any
from multiprocessing import Pool
from rich.console import Console
from numpy.typing import ArrayLike
from matplotlib.gridspec import GridSpec


def convergence(params):
    """
    Generate timeseries plots of power, thrust, CP, and CT for a series of runs

    Args:
        params (Dict): A dictionary of settings
    """
    combs     = sweep.get_combinations(params)
    formats   = sweep.determine_format(combs)
    casenames = sweep.return_case_strings(combs,formats)

    model = params['rotor_model'].lower() + f'_sweep'

    remove_data = params['exclude_time']
    save_period = params['save_interval']

    save_dir = f"{params['base_dir']}/{model}/figs/convergence"

    os.makedirs(save_dir, exist_ok=False)

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
        ax1.set_ylim([1300,1900])

        ax2.plot(timeseries,torque_aero / 1000,linestyle='solid',linewidth=2)
        ax2.set_ylabel(r'Torque [kN m]')
        ax2.set_ylim([13000,19000])

        ax3.plot(timeseries,power_aero / 1000,linestyle='solid',linewidth=2,label='aero')
        ax3.plot(timeseries,power_mech / 1000,linestyle='solid',linewidth=2,label='mech')
        ax3.set_ylabel(r'Power [kW]')
        ax3.set_ylim([6000,10000])
        ax3.legend(loc="upper right", fancybox=True, shadow=False, ncol=3, fontsize=8)

        error = np.zeros(len(thrust) - 1)
        for i in range(len(thrust) - 1):
            error[i] = np.abs(thrust[i] - thrust[i + 1]) / np.abs(thrust[i])

        ax4.plot(timeseries[:-1],error,linestyle='solid',linewidth=2)
        ax4.set_yscale('log')
        ax4.set_ylabel(r'relative change [-]')

        error = np.zeros(len(torque_aero) - 1)
        for i in range(len(torque_aero) - 1):
            error[i] = np.abs(torque_aero[i] - torque_aero[i + 1]) / np.abs(torque_aero[i])

        ax5.plot(timeseries[:-1],error,linestyle='solid',linewidth=2)
        ax5.set_yscale('log')
        # ax5.set_ylabel(r'relative change [-]')

        error_a = np.zeros(len(power_aero) - 1)
        error_m = np.zeros(len(power_mech) - 1)
        for i in range(len(power_aero) - 1):
            error_a[i] = np.abs(power_aero[i] - power_aero[i + 1]) / np.abs(power_aero[i])
            error_m[i] = np.abs(power_mech[i] - power_mech[i + 1]) / np.abs(power_mech[i])

        # ax6.plot(timeseries[:-1],error_a,linestyle='-',linewidth=2, marker='.')
        # ax6.plot(timeseries[:-1],error_m,linestyle='-',linewidth=2, marker='x')
        ax6.plot(timeseries[:-1],error_a,linestyle='-',linewidth=2)
        ax6.plot(timeseries[:-1],error_m,linestyle='-',linewidth=2)
        ax6.set_yscale('log')
        # ax6.set_ylabel(r'relative change [-]')

        ax7.plot(timeseries,v0,linestyle='solid',linewidth=2)
        ax7.set_ylabel(r'$V_0$ [m/s]')
        ax7.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))

        ax8.plot(timeseries,ct,linestyle='solid',linewidth=2)
        ax8.set_ylabel(r'$C_T$ [-]')
        ax8.set_ylim([0.7,1])

        ax9.plot(timeseries,cp,linestyle='solid',linewidth=2)
        ax9.set_ylabel(r'$C_P$ [-]')
        ax9.set_ylim([0.5,0.9])
        ax9.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

        error = np.zeros(len(v0) - 1)
        for i in range(len(v0) - 1):
            error[i] = np.abs(v0[i] - v0[i + 1]) / np.abs(v0[i])

        ax10.plot(timeseries[:-1],error,linestyle='solid',linewidth=2)
        ax10.set_yscale('log')
        ax10.set_ylabel(r'relative change [-]')
        ax10.set_xlabel(r'Time [min]')

        error = np.zeros(len(ct) - 1)
        for i in range(len(ct) - 1):
            error[i] = np.abs(ct[i] - ct[i + 1]) / np.abs(ct[i])

        ax11.plot(timeseries[:-1],error,linestyle='solid',linewidth=2)
        ax11.set_yscale('log')
        # ax8.set_ylabel(r'relative change [-]')
        ax11.set_xlabel(r'Time [min]')

        error = np.zeros(len(cp) - 1)
        for i in range(len(cp) - 1):
            error[i] = np.abs(cp[i] - cp[i + 1]) / np.abs(cp[i])

        ax12.plot(timeseries[:-1],error,linestyle='solid',linewidth=2)
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

    print('Done.')

def fast_process(file, static_args):

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

    print(f'Working on {case}...')

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

    u = (file2read.variables['WTP_U'              ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    v = (file2read.variables['WTP_V'              ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)
    w = (file2read.variables['WTP_W'              ][Ts:Te,:]).reshape(Nt,Nelm,Nsct)

    rhub = dhub/2
    dist = 0.0
    dr = np.zeros(Nelm)
    for i in range(0,Nelm):
        dist = dist + 0.5*((diameter/2 - rhub)/Nelm)
        dr[i] = rhub + dist
        dist = dist + 0.5*((diameter/2 - rhub)/Nelm)

    rOverR = dr/(diameter/2)

    var_holder = {}

    var_holder['diameter']    = diameter
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

    var_holder['vrel']        = vrel
    var_holder['phi']         = phi

    var_holder['bpx']         = bpx
    var_holder['bpy']         = bpy
    var_holder['bpz']         = bpz

    np.savez( os.path.join(f'{dir_path}/{case}_lite.npz'),**var_holder)
    
    del var_holder


def parproc(processes: int, params: Dict[str, Any]) -> None:
    """
    Processes the WRF output files in parallel

    Args:
        processes (int): The number of parallel processes to use
        opt_params (Dict): A dictionary of settings including sample locations if desired
    """

    filelist = sorted(glob.glob(params['base_dir'] + '/*/wrfout_d02_0001-01-01_00_00_00'))

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

    namelist  = preproc.parse_namelist(params)
    turbprops = preproc.parse_turbine_properties(params)
    turbloc   = preproc.parse_turbine_location(params)

    static_args = {}

    static_args['save_period'] = params['save_period']
    static_args['remove_data'] = params['remove_data']
    static_args['diameter']    = float(turbprops['Rotor diameter [m]'])
    static_args['dhub']        = float(turbprops['Hub diameter [m]'])
    static_args['Nsct']        = int(namelist['physics'].get('wind_wtp_nSections', None))
    static_args['Nelm']        = int(namelist['physics'].get('wind_wtp_nElements', None))
    static_args['hub_height']  = float(turbprops['Hub height [m]'])
    static_args['tower_xloc']  = float(turbloc[0])
    static_args['tower_yloc']  = float(turbloc[1])
    static_args['uinf']        = params['Ufst']

    # Process the files
    start_time = time.time()
    
    # Parallel pool
    with Pool(processes) as pool:
        pool.starmap(fast_process, [(file, static_args) for file in filelist])

    end_time = time.time()

    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    console.print(f"Finished in [bold][green3]{minutes}[/green3][/bold] min and [bold][green3]{seconds}[/green3][/bold] sec.")

def annulus_average(f: ArrayLike, theta: ArrayLike) -> ArrayLike:
    """
    Compute annulus average of spatial quantity f(r,theta) over rotor

    Args:
        f (ArrayLike): Spatial quantity
        theta (ArrayLike): Azimuthal locations over which to average
    """

    X_azim = 1 / (2 * np.pi) * np.trapz(f, theta, axis=-1)

    return X_azim

def rotor_average(f: ArrayLike, r: ArrayLike, theta: ArrayLike) -> ArrayLike:
    """
    Compute rotor average of spatial quantity f(r,theta) over rotor

    Args:
        f (ArrayLike): Spatial quantity
        r (ArrayLike): Radial points over rotor
        theta (ArrayLike): Azimuthal points over rotor
    """

    X_azim = 1 / (2 * np.pi) * np.trapz(f, theta, axis=-1)

    X_rotor = 2 * np.trapz(X_azim * r, r)

    return X_rotor

def per_error(A: float, E: float) -> float:
    """
    Returns the error between from reference value A and another value E

    Args:
        A (float): Comparison value
        E (float): Reference value
    """

    error = ((A - E) / E) * 100

    return error

def extract_sounding(file_list: list[str]) -> Dict[float, Any]:
    """
    Extracts u and v velocity components from wrf sounding file

    Args:
        file_list (list): List of file paths to sounding files
    """

    data_dict = {}

    i = 0
    
    for file in file_list:
        with open(file, 'r') as f:
            lines = f.readlines()
            
            # Skip the header line and extract columns 1, 4, and 5
            data = np.loadtxt(lines[1:])[:, [0, 3, 4]]

            key = Path(file).parent.name
            data_dict[key] = data
    
    return data_dict