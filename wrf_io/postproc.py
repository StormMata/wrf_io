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

    remove_data = params['exclude_time']
    save_period = params['save_interval']

    for case in casenames:

        print(f'Working on {case}...')

        case_path = f"{params['base_dir']}/{model}/{dir_name}"

        file2read = netCDF4.Dataset(f'/anvil/scratch/x-smata/wrf_les_sweep/iea15MW_validation/gad_sweep/{case}/wrfout_d02_0001-01-01_00_00_00','r',mmap=False) # type: ignore # Read Netcdf-type WRF output file
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

        plt.savefig(f"/scratch/09909/smata/wrf_les_sweep/runs/grid_study/convergence_{case}.png", bbox_inches="tight", dpi=600) 

# def parallel_process_function(file):

#     for i in range(1, 101):
#         time.sleep(0.01)

#     return True

def parallel_process_function(file):

    save_period = 10.0 # in seconds
    remove_data = 15.0 # in minutes;  discard first xxx minutes (e.g., ~2 flow-through times)

    diameter   = 199.00
    dhub       = 4.80
    hub_height = 378.00
    Ntrb       = 1
    Nsct       = 158
    Nelm       = 26
    uinf       = 7.0
    rho        = 1.225

    # smearing distance
    smearingDist = 3

    ## tower and rotor apex locations:
    tower_xloc = 695.0
    tower_yloc = 500.0

    #============================================================================================================
    # Main logic [generally no edits beyond this point]
    #============================================================================================================

    file2read = netCDF4.Dataset(file,'r',mmap=False) # type: ignore # Read Netcdf-type WRF output file
    file2read.variables.keys()

    timeidx = wrf.extract_times(file2read, timeidx=wrf.ALL_TIMES, meta=False)
    times={}
    for i in range(0,len(timeidx)):
        times[i] = pd.to_datetime(str(timeidx[i])).strftime('%Y-%m-%d %H:%M:%S')

    # Field variables
    dx = file2read.getncattr('DX')
    dy = file2read.getncattr('DY')
    dt = file2read.getncattr('DT')
    Nx = file2read.getncattr('WEST-EAST_PATCH_END_UNSTAG')
    Ny = file2read.getncattr('SOUTH-NORTH_PATCH_END_UNSTAG')
    Nz = file2read.getncattr('BOTTOM-TOP_PATCH_END_UNSTAG')
    Nt = file2read.variables['Times'].shape[0]

    if(remove_data == 0.0):
        save_period = 0.0
    else:
        save_period = (remove_data * 60 / save_period) + 1 # first xxx timesteps are not included in analysis
    process_period  = Nt - int(save_period) # consider only xxx timesteps in analysis

    Ts = Nt - int(process_period)
    Te = Nt
    Nt = Te - Ts

    downstreamDist = int(np.floor(1 * diameter / dx))

    h   = file2read.variables['HGT'][Ts:Te,:,:]
    ph  = file2read.variables['PH' ][Ts:Te,:,:,:]
    phb = file2read.variables['PHB'][Ts:Te,:,:,:]
    p   = file2read.variables['P'  ][Ts:Te,:,:,:]
    pb  = file2read.variables['PB' ][Ts:Te,:,:,:]
    th  = file2read.variables['T'  ][Ts:Te,:,:,:]
    u   = file2read.variables['U'  ][Ts:Te,:,:,:]
    v   = file2read.variables['V'  ][Ts:Te,:,:,:]
    w   = file2read.variables['W'  ][Ts:Te,:,:,:]

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
    rotorApex_x = file2read.variables['WTP_ROTORAPEX_X' ][Ts:Te,:]
    rotorApex_y = file2read.variables['WTP_ROTORAPEX_Y' ][Ts:Te,:]
    rotorApex_z = file2read.variables['WTP_ROTORAPEX_Z' ][Ts:Te,:]

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

    Lx = Nx * dx # Lx of the computational domain
    Ly = Ny * dy # Ly of the computational domain

    nrow_vect = np.arange(0,Nx)
    ncol_vect = np.arange(0,Ny)
    neta_vect = np.arange(0,Nz)
    ntme_vect = np.arange(0,Nt)

    h_reshape = h.reshape(Nt,1,Ny,Nx)

    u4d = 0.5*(  u[:,:,:,nrow_vect] +   u[:,:,:,nrow_vect+1] ) # x-component of wind speed in 4d
    del u
    gc.collect()
    v4d = 0.5*(  v[:,:,ncol_vect,:] +   v[:,:,ncol_vect+1,:] ) # y-component of wind speed in 4d
    del v
    gc.collect()
    w4d = 0.5*(  w[:,neta_vect,:,:] +   w[:,neta_vect+1,:,:] ) # z-component of wind speed in 4d
    del w
    gc.collect()
    phm  = 0.5*( ph[:,neta_vect,:,:] +  ph[:,neta_vect+1,:,:] )
    del ph
    gc.collect()
    phbm = 0.5*(phb[:,neta_vect,:,:] + phb[:,neta_vect+1,:,:] )
    del phb
    gc.collect()
    z4d = (phm[:,neta_vect,:,:] + phbm[:,neta_vect,:,:])/9.81 - h_reshape # altitude in 4d
    del phm,phbm,h,h_reshape
    gc.collect()
    p4d = p + pb # total pressure in Pa in 4d (perturbation pressure + base state pressure)
    del p,pb
    gc.collect()

    file2read.close()
    ###########################################################################
    # tower
    ix_tower = int(np.floor((tower_xloc + 0.5*dx)/dx))     # i-level of the tower in input file
    jy_tower = int(np.floor((tower_yloc + 0.5*dy)/dy))     # j-level of the tower in input file

    # rotor
    ix_rotor = int(np.floor((np.mean(rotorApex_x) + 0.5*dx)/dx)) # i-level of the rotor in wrfout file         
    jy_rotor = int(np.floor((np.mean(rotorApex_y) + 0.5*dy)/dy)) # j-level of the tower in wrfout file

    jyl_rotor = jy_rotor - int(np.floor(((diameter/2) + 0.5*dy)/dy)) - smearingDist
    jyr_rotor = jy_rotor + int(np.floor(((diameter/2) + 0.5*dy)/dy)) + smearingDist

    rotor_xloc = np.mean(rotorApex_x)                 # Rotor x-position in meters
    rotor_yloc = np.mean(rotorApex_y)                 # Rotor y-position in meters
    rotor_zloc = np.mean(rotorApex_z)                 # Rotor z-position in meters
    ###########################################################################
    ## Vertical positions:
    # Time-averaged vertical levels:
    z_m  = np.mean(np.mean(z4d,axis=3),axis=2)    # Horizontal averaging in 4d 
    z_av = np.mean(z_m,axis=0)

    # Turbine hub upper and lower boundaries based on k-level:
    for k in range(0,Nz-1):
        if z_av[k] > hub_height:
            ktop = k; kbot = k-1
            break

    # k-level of the turbine hub center:
    k_loc = min(range(len(z_av)),key=lambda m: abs(z_av[m]-hub_height))
    z_height = z_av[k_loc]

    kzd_rotor = k_loc - int(np.floor(((diameter/2) + 0.5*dx)/dx)) - smearingDist
    kzu_rotor = k_loc + int(np.floor(((diameter/2) + 0.5*dx)/dx)) + smearingDist
    ###########################################################################
    # mean and fluctuations of velocity components and temperature:

    # u-velocity component:
    um = np.mean(np.mean(u4d,axis=3),axis=2)    # U_mean
    up = u4d - um.reshape(Nt,Nz,1,1)         # U_prime = U - U_mean
        
    # v-velocity component:
    vm = np.mean(np.mean(v4d,axis=3),axis=2)    # V_mean
    vp = v4d - vm.reshape(Nt,Nz,1,1)        # V_prime = V - V_mean

    # w-velocity component:
    wm = np.mean(np.mean(w4d,axis=3),axis=2)    # W_mean
    wp = w4d - wm.reshape(Nt,Nz,1,1)        # W_prime = W - W_mean

    # Total horizontal wind speed:
    uh = np.sqrt(u4d**2 + v4d**2)              # Total horizontal wind speed
    ###########################################################################
    # calculation of the horizontal and vertical spacings:
    xx = np.linspace(dx/2,Lx-(dx/2),Nx) # Longitudinal spacing in meters
    yy = np.linspace(dy/2,Ly-(dy/2),Ny) # Lateral spacing in meters

    X2,Y2 = np.meshgrid(xx,yy)            # Meshgrid for plotting contours in x-y plane
    X3,Z2 = np.meshgrid(xx,z_av)          # Meshgrid for plotting contours in x-z plane
    Y3,Z3 = np.meshgrid(yy,z_av)          # Meshgrid for plotting contours in y-z plane
    ###########################################################################
    ## downstream positions,time averages,and interpolations

    # Velocity components,and total horizontal wind speed at rotor's lateral position:
    uy = u4d[:,:,jy_rotor,:] + ( u4d[:,:,jy_rotor+1,:] - u4d[:,:,jy_rotor,:] )*(rotor_yloc - jy_rotor*dy)/dy
    vy = v4d[:,:,jy_rotor,:] + ( v4d[:,:,jy_rotor+1,:] - v4d[:,:,jy_rotor,:] )*(rotor_yloc - jy_rotor*dy)/dy
    wy = w4d[:,:,jy_rotor,:] + ( w4d[:,:,jy_rotor+1,:] - w4d[:,:,jy_rotor,:] )*(rotor_yloc - jy_rotor*dy)/dy
    uhy = uh[:,:,jy_rotor,:] + ( uh[:,:,jy_rotor+1,:] - uh[:,:,jy_rotor,:] )*(rotor_yloc - jy_rotor*dy)/dy

    idownstreamrotor = ix_rotor+downstreamDist
    ux_1D = u4d[:,:,:,idownstreamrotor]; vx_1D = v4d[:,:,:,idownstreamrotor]; wx_1D = w4d[:,:,:,idownstreamrotor]

    # Velocity components at hub-height:
    uz = u4d[:,kbot,:,:] + ( u4d[:,ktop,:,:] -  u4d[:,kbot,:,:] )*( ( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] ) )
    vz = v4d[:,kbot,:,:] + ( v4d[:,ktop,:,:] -  v4d[:,kbot,:,:] )*( ( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] ) )
    wz = w4d[:,kbot,:,:] + ( w4d[:,ktop,:,:] -  w4d[:,kbot,:,:] )*( ( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] ) )
    pz = p4d[:,kbot,:,:] + ( p4d[:,ktop,:,:] -  p4d[:,kbot,:,:] )*( ( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] ) )

    # Variance of velocity components at hub-height:
    uz_var = np.var(uz,axis=0); vz_var = np.var(vz,axis=0); wz_var = np.var(wz,axis=0)

    # Variance of velocity components at rotor's lateral position:
    uy_var = np.var(uy,axis=0); vy_var = np.var(vy,axis=0); wy_var = np.var(wy,axis=0)

    # Total horizontal wind speed at hub-height:
    uh_z = uh[:,kbot,:,:] + ( uh[:,ktop,:,:] -  uh[:,kbot,:,:] )*( ( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] ) )

    # Time averages of velocity components and total horizontal wind speed:
    ut = np.mean(u4d,axis=0); vt = np.mean(v4d,axis=0); wt = np.mean(w4d,axis=0); uht = np.mean(uh,axis=0)

    # Time-averaged total pressure
    pt = np.mean(p4d,axis=0)

    # Time averages of velocity components at hub-height:
    utz = ut[kbot,:,:] + ( ut[ktop,:,:] - ut[kbot,:,:] )*( ( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] ) )
    vtz = vt[kbot,:,:] + ( vt[ktop,:,:] - vt[kbot,:,:] )*( ( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] ) )
    wtz = wt[kbot,:,:] + ( wt[ktop,:,:] - wt[kbot,:,:] )*( ( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] ) )

    # Time-averaged total pressure at hub-height:
    # ptz = pt[kbot,:,:] + ( pt[ktop,:,:] - pt[kbot,:,:] )*( ( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] ) )

    # Time averages of total horizontal wind speed at hub-height:
    uhtz = uht[kbot,:,:] + ( uht[ktop,:,:] - uht[kbot,:,:] )*( ( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] ) )

    # Time averages of velocity components at rotor's longitudinal position:
    uxt_1D = np.mean(ux_1D,axis=0); vxt_1D = np.mean(vx_1D,axis=0); wxt_1D = np.mean(wx_1D,axis=0)

    # Variance of velocity components at rotor's longitudinal position:
    ux_1D_var = np.var(ux_1D,axis=0); vx_1D_var = np.var(vx_1D,axis=0); wx_1D_var = np.var(wx_1D,axis=0)

    # Time-averaged total pressure at rotor's longitudinal position at hub-height:
    # pytz = ptz[jy_rotor,:] + ( ptz[jy_rotor+1,:] - ptz[jy_rotor,:] )*(rotor_yloc - jy_rotor*dy)/dy

    # Time averages of velocity components at rotor's longitudinal position at hub-height:
    # uytz = utz[jy_rotor,:] + ( utz[jy_rotor+1,:] - utz[jy_rotor,:] )*(rotor_yloc - jy_rotor*dy)/dy
    # vytz = vtz[jy_rotor,:] + ( vtz[jy_rotor+1,:] - vtz[jy_rotor,:] )*(rotor_yloc - jy_rotor*dy)/dy
    # wytz = wtz[jy_rotor,:] + ( wtz[jy_rotor+1,:] - wtz[jy_rotor,:] )*(rotor_yloc - jy_rotor*dy)/dy

    # Time-and-disk-averaged pressure and velocity components:
    pytz = np.mean(pt[kzd_rotor:kzu_rotor,jyl_rotor:jyr_rotor,:],axis=(0,1))
    uytz = np.mean(ut[kzd_rotor:kzu_rotor,jyl_rotor:jyr_rotor,:],axis=(0,1))
    vytz = np.mean(vt[kzd_rotor:kzu_rotor,jyl_rotor:jyr_rotor,:],axis=(0,1))
    wytz = np.mean(wt[kzd_rotor:kzu_rotor,jyl_rotor:jyr_rotor,:],axis=(0,1))
    ###########################################################################
    uinf = np.mean(uh_z[:,int(jy_rotor),0],axis=0)
    # uinf = 12.0
    # Donwstream positions in meters:
    dist_n3D = rotor_xloc - (3.4*diameter)
    dist_0D  = rotor_xloc + (0*diameter)
    dist_2D  = rotor_xloc + (2*diameter)
    dist_4D  = rotor_xloc + (4*diameter)
    dist_6D  = rotor_xloc + (6*diameter)
    dist_8D  = rotor_xloc + (8*diameter)
    dist_10D = rotor_xloc + (10*diameter)
    # dist_15D = rotor_xloc + (15*diameter)

    # Downstream positions in i-level:
    lat_dist_n3D = int(np.floor((dist_n3D+(0.5*dx))/dx))
    lat_dist_0D  = int(np.floor((dist_0D+(0.5*dx))/dx))
    lat_dist_2D  = int(np.floor((dist_2D+(0.5*dx))/dx))
    lat_dist_4D  = int(np.floor((dist_4D+(0.5*dx))/dx))
    lat_dist_6D  = int(np.floor((dist_6D+(0.5*dx))/dx))
    lat_dist_8D  = int(np.floor((dist_8D+(0.5*dx))/dx))
    lat_dist_10D = int(np.floor((dist_10D+(0.5*dx))/dx))
    # lat_dist_15D = int(np.floor((dist_15D+(0.5*dx))/dx))

    # Donwstream positions in meters:
    dist_lon_0D = rotor_yloc + (0*diameter)

    # Downstream positions in j-level:
    lon_dist_0D = int(np.floor((dist_lon_0D+(0.5*dy))/dy))

# ================================================================================================================================

    # u-velocity component at different downstream locations, y-z plots:
    ux_n3D = u4d[:,:,:,lat_dist_n3D] + ( u4d[:,:,:,lat_dist_n3D+1] - u4d[:,:,:,lat_dist_n3D] )*(dist_n3D - lat_dist_n3D*dx)/dx
    ux_0D  = u4d[:,:,:,lat_dist_0D]  + ( u4d[:,:,:,lat_dist_0D+1]  - u4d[:,:,:,lat_dist_0D]  )*(dist_0D  - lat_dist_0D*dx)/dx
    ux_2D  = u4d[:,:,:,lat_dist_2D]  + ( u4d[:,:,:,lat_dist_2D+1]  - u4d[:,:,:,lat_dist_2D]  )*(dist_2D  - lat_dist_2D*dx)/dx
    ux_4D  = u4d[:,:,:,lat_dist_4D]  + ( u4d[:,:,:,lat_dist_4D+1]  - u4d[:,:,:,lat_dist_4D]  )*(dist_4D  - lat_dist_4D*dx)/dx
    ux_6D  = u4d[:,:,:,lat_dist_6D]  + ( u4d[:,:,:,lat_dist_6D+1]  - u4d[:,:,:,lat_dist_6D]  )*(dist_6D  - lat_dist_6D*dx)/dx
    ux_8D  = u4d[:,:,:,lat_dist_8D]  + ( u4d[:,:,:,lat_dist_8D+1]  - u4d[:,:,:,lat_dist_8D]  )*(dist_8D  - lat_dist_8D*dx)/dx
    ux_10D = u4d[:,:,:,lat_dist_10D] + ( u4d[:,:,:,lat_dist_10D+1] - u4d[:,:,:,lat_dist_10D] )*(dist_10D - lat_dist_10D*dx)/dx

    uxy_n3D = ux_n3D[:,:,lon_dist_0D] + ( ux_n3D[:,:,lon_dist_0D+1] - ux_n3D[:,:,lon_dist_0D] )*(dist_lon_0D - lon_dist_0D*dy)/dy
    uxy_0D  = ux_0D[:,:,lon_dist_0D]  + ( ux_0D[:,:,lon_dist_0D+1]  - ux_0D[:,:,lon_dist_0D]  )*(dist_lon_0D - lon_dist_0D*dy)/dy
    uxy_2D  = ux_2D[:,:,lon_dist_0D]  + ( ux_2D[:,:,lon_dist_0D+1]  - ux_2D[:,:,lon_dist_0D]  )*(dist_lon_0D - lon_dist_0D*dy)/dy
    uxy_4D  = ux_4D[:,:,lon_dist_0D]  + ( ux_4D[:,:,lon_dist_0D+1]  - ux_4D[:,:,lon_dist_0D]  )*(dist_lon_0D - lon_dist_0D*dy)/dy
    uxy_6D  = ux_6D[:,:,lon_dist_0D]  + ( ux_6D[:,:,lon_dist_0D+1]  - ux_6D[:,:,lon_dist_0D]  )*(dist_lon_0D - lon_dist_0D*dy)/dy
    uxy_8D  = ux_8D[:,:,lon_dist_0D]  + ( ux_8D[:,:,lon_dist_0D+1]  - ux_8D[:,:,lon_dist_0D]  )*(dist_lon_0D - lon_dist_0D*dy)/dy
    uxy_10D = ux_10D[:,:,lon_dist_0D] + ( ux_10D[:,:,lon_dist_0D+1] - ux_10D[:,:,lon_dist_0D] )*(dist_lon_0D - lon_dist_0D*dy)/dy

    uxyt_0D  = np.mean(uxy_0D,axis=0)/uinf
    uxyt_2D  = np.mean(uxy_2D,axis=0)/uinf
    uxyt_4D  = np.mean(uxy_4D,axis=0)/uinf
    uxyt_6D  = np.mean(uxy_6D,axis=0)/uinf
    uxyt_8D  = np.mean(uxy_8D,axis=0)/uinf
    uxyt_10D = np.mean(uxy_10D,axis=0)/uinf

    # u-velocity component at different downstream locations, x-y plots:
    uxz_0D  = uz[:,:,lat_dist_0D]  + ( uz[:,:,lat_dist_0D+1]  - uz[:,:,lat_dist_0D]  )*( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] )
    uxz_2D  = uz[:,:,lat_dist_2D]  + ( uz[:,:,lat_dist_2D+1]  - uz[:,:,lat_dist_2D]  )*( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] )
    uxz_4D  = uz[:,:,lat_dist_4D]  + ( uz[:,:,lat_dist_4D+1]  - uz[:,:,lat_dist_4D]  )*( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] )
    uxz_6D  = uz[:,:,lat_dist_6D]  + ( uz[:,:,lat_dist_6D+1]  - uz[:,:,lat_dist_6D]  )*( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] )
    uxz_8D  = uz[:,:,lat_dist_8D]  + ( uz[:,:,lat_dist_8D+1]  - uz[:,:,lat_dist_8D]  )*( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] )
    uxz_10D = uz[:,:,lat_dist_10D] + ( uz[:,:,lat_dist_10D+1] - uz[:,:,lat_dist_10D] )*( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] )

    uxzt_0D  = np.mean(uxz_0D,axis=0)/uinf
    uxzt_2D  = np.mean(uxz_2D,axis=0)/uinf
    uxzt_4D  = np.mean(uxz_4D,axis=0)/uinf
    uxzt_6D  = np.mean(uxz_6D,axis=0)/uinf
    uxzt_8D  = np.mean(uxz_8D,axis=0)/uinf
    uxzt_10D = np.mean(uxz_10D,axis=0)/uinf

# ================================================================================================================================

    # v-velocity component at different downstream locations, y-z plots:
    vx_0D  = v4d[:,:,:,lat_dist_0D]  + ( v4d[:,:,:,lat_dist_0D+1]  - v4d[:,:,:,lat_dist_0D]  )*(dist_0D  - lat_dist_0D*dx)/dx
    vx_2D  = v4d[:,:,:,lat_dist_2D]  + ( v4d[:,:,:,lat_dist_2D+1]  - v4d[:,:,:,lat_dist_2D]  )*(dist_2D  - lat_dist_2D*dx)/dx
    vx_4D  = v4d[:,:,:,lat_dist_4D]  + ( v4d[:,:,:,lat_dist_4D+1]  - v4d[:,:,:,lat_dist_4D]  )*(dist_4D  - lat_dist_4D*dx)/dx
    vx_6D  = v4d[:,:,:,lat_dist_6D]  + ( v4d[:,:,:,lat_dist_6D+1]  - v4d[:,:,:,lat_dist_6D]  )*(dist_6D  - lat_dist_6D*dx)/dx
    vx_8D  = v4d[:,:,:,lat_dist_8D]  + ( v4d[:,:,:,lat_dist_8D+1]  - v4d[:,:,:,lat_dist_8D]  )*(dist_8D  - lat_dist_8D*dx)/dx
    vx_10D = v4d[:,:,:,lat_dist_10D] + ( v4d[:,:,:,lat_dist_10D+1] - v4d[:,:,:,lat_dist_10D] )*(dist_10D - lat_dist_10D*dx)/dx

    vxy_0D  = vx_0D[:,:,lon_dist_0D]  + ( vx_0D[:,:,lon_dist_0D+1]  - vx_0D[:,:,lon_dist_0D]  )*(dist_lon_0D - lon_dist_0D*dy)/dy
    vxy_2D  = vx_2D[:,:,lon_dist_0D]  + ( vx_2D[:,:,lon_dist_0D+1]  - vx_2D[:,:,lon_dist_0D]  )*(dist_lon_0D - lon_dist_0D*dy)/dy
    vxy_4D  = vx_4D[:,:,lon_dist_0D]  + ( vx_4D[:,:,lon_dist_0D+1]  - vx_4D[:,:,lon_dist_0D]  )*(dist_lon_0D - lon_dist_0D*dy)/dy
    vxy_6D  = vx_6D[:,:,lon_dist_0D]  + ( vx_6D[:,:,lon_dist_0D+1]  - vx_6D[:,:,lon_dist_0D]  )*(dist_lon_0D - lon_dist_0D*dy)/dy
    vxy_8D  = vx_8D[:,:,lon_dist_0D]  + ( vx_8D[:,:,lon_dist_0D+1]  - vx_8D[:,:,lon_dist_0D]  )*(dist_lon_0D - lon_dist_0D*dy)/dy
    vxy_10D = vx_10D[:,:,lon_dist_0D] + ( vx_10D[:,:,lon_dist_0D+1] - vx_10D[:,:,lon_dist_0D] )*(dist_lon_0D - lon_dist_0D*dy)/dy

    vxyt_0D  = np.mean(vxy_0D,axis=0)/uinf
    vxyt_2D  = np.mean(vxy_2D,axis=0)/uinf
    vxyt_4D  = np.mean(vxy_4D,axis=0)/uinf
    vxyt_6D  = np.mean(vxy_6D,axis=0)/uinf
    vxyt_8D  = np.mean(vxy_8D,axis=0)/uinf
    vxyt_10D = np.mean(vxy_10D,axis=0)/uinf

    vxz_0D  = vz[:,:,lat_dist_0D]  + ( vz[:,:,lat_dist_0D+1]  - vz[:,:,lat_dist_0D]  )*( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] )
    vxz_2D  = vz[:,:,lat_dist_2D]  + ( vz[:,:,lat_dist_2D+1]  - vz[:,:,lat_dist_2D]  )*( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] )
    vxz_4D  = vz[:,:,lat_dist_4D]  + ( vz[:,:,lat_dist_4D+1]  - vz[:,:,lat_dist_4D]  )*( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] )
    vxz_6D  = vz[:,:,lat_dist_6D]  + ( vz[:,:,lat_dist_6D+1]  - vz[:,:,lat_dist_6D]  )*( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] )
    vxz_8D  = vz[:,:,lat_dist_8D]  + ( vz[:,:,lat_dist_8D+1]  - vz[:,:,lat_dist_8D]  )*( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] )
    vxz_10D = vz[:,:,lat_dist_10D] + ( vz[:,:,lat_dist_10D+1] - vz[:,:,lat_dist_10D] )*( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] )

    vxzt_0D  = np.mean(vxz_0D,axis=0)/uinf
    vxzt_2D  = np.mean(vxz_2D,axis=0)/uinf
    vxzt_4D  = np.mean(vxz_4D,axis=0)/uinf
    vxzt_6D  = np.mean(vxz_6D,axis=0)/uinf
    vxzt_8D  = np.mean(vxz_8D,axis=0)/uinf
    vxzt_10D = np.mean(vxz_10D,axis=0)/uinf

# ================================================================================================================================

    # pressure at different downstream locations, y-z plots:
    p_0D  = p4d[:,:,:,lat_dist_0D]  + ( p4d[:,:,:,lat_dist_0D+1]  - p4d[:,:,:,lat_dist_0D]  )*(dist_0D  - lat_dist_0D*dx)/dx
    p_2D  = p4d[:,:,:,lat_dist_2D]  + ( p4d[:,:,:,lat_dist_2D+1]  - p4d[:,:,:,lat_dist_2D]  )*(dist_2D  - lat_dist_2D*dx)/dx
    p_4D  = p4d[:,:,:,lat_dist_4D]  + ( p4d[:,:,:,lat_dist_4D+1]  - p4d[:,:,:,lat_dist_4D]  )*(dist_4D  - lat_dist_4D*dx)/dx
    p_6D  = p4d[:,:,:,lat_dist_6D]  + ( p4d[:,:,:,lat_dist_6D+1]  - p4d[:,:,:,lat_dist_6D]  )*(dist_6D  - lat_dist_6D*dx)/dx
    p_8D  = p4d[:,:,:,lat_dist_8D]  + ( p4d[:,:,:,lat_dist_8D+1]  - p4d[:,:,:,lat_dist_8D]  )*(dist_8D  - lat_dist_8D*dx)/dx
    p_10D = p4d[:,:,:,lat_dist_10D] + ( p4d[:,:,:,lat_dist_10D+1] - p4d[:,:,:,lat_dist_10D] )*(dist_10D - lat_dist_10D*dx)/dx

    pxy_0D  = p_0D[:,:,lon_dist_0D]  + ( p_0D[:,:,lon_dist_0D+1]  - p_0D[:,:,lon_dist_0D]  )*(dist_lon_0D - lon_dist_0D*dy)/dy
    pxy_2D  = p_2D[:,:,lon_dist_0D]  + ( p_2D[:,:,lon_dist_0D+1]  - p_2D[:,:,lon_dist_0D]  )*(dist_lon_0D - lon_dist_0D*dy)/dy
    pxy_4D  = p_4D[:,:,lon_dist_0D]  + ( p_4D[:,:,lon_dist_0D+1]  - p_4D[:,:,lon_dist_0D]  )*(dist_lon_0D - lon_dist_0D*dy)/dy
    pxy_6D  = p_6D[:,:,lon_dist_0D]  + ( p_6D[:,:,lon_dist_0D+1]  - p_6D[:,:,lon_dist_0D]  )*(dist_lon_0D - lon_dist_0D*dy)/dy
    pxy_8D  = p_8D[:,:,lon_dist_0D]  + ( p_8D[:,:,lon_dist_0D+1]  - p_8D[:,:,lon_dist_0D]  )*(dist_lon_0D - lon_dist_0D*dy)/dy
    pxy_10D = p_10D[:,:,lon_dist_0D] + ( p_10D[:,:,lon_dist_0D+1] - p_10D[:,:,lon_dist_0D] )*(dist_lon_0D - lon_dist_0D*dy)/dy

    pxyt_0D  = np.mean(pxy_0D,axis=0)
    pxyt_2D  = np.mean(pxy_2D,axis=0)
    pxyt_4D  = np.mean(pxy_4D,axis=0)
    pxyt_6D  = np.mean(pxy_6D,axis=0)
    pxyt_8D  = np.mean(pxy_8D,axis=0)
    pxyt_10D = np.mean(pxy_10D,axis=0)

    pxz_0D  = pz[:,:,lat_dist_0D]  + ( pz[:,:,lat_dist_0D+1]  - pz[:,:,lat_dist_0D]  )*( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] )
    pxz_2D  = pz[:,:,lat_dist_2D]  + ( pz[:,:,lat_dist_2D+1]  - pz[:,:,lat_dist_2D]  )*( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] )
    pxz_4D  = pz[:,:,lat_dist_4D]  + ( pz[:,:,lat_dist_4D+1]  - pz[:,:,lat_dist_4D]  )*( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] )
    pxz_6D  = pz[:,:,lat_dist_6D]  + ( pz[:,:,lat_dist_6D+1]  - pz[:,:,lat_dist_6D]  )*( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] )
    pxz_8D  = pz[:,:,lat_dist_8D]  + ( pz[:,:,lat_dist_8D+1]  - pz[:,:,lat_dist_8D]  )*( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] )
    pxz_10D = pz[:,:,lat_dist_10D] + ( pz[:,:,lat_dist_10D+1] - pz[:,:,lat_dist_10D] )*( hub_height-z_av[kbot] )/( z_av[ktop] - z_av[kbot] )

    pxzt_0D  = np.mean(pxz_0D,axis=0)
    pxzt_2D  = np.mean(pxz_2D,axis=0)
    pxzt_4D  = np.mean(pxz_4D,axis=0)
    pxzt_6D  = np.mean(pxz_6D,axis=0)
    pxzt_8D  = np.mean(pxz_8D,axis=0)
    pxzt_10D = np.mean(pxz_10D,axis=0)

    ###########################################################################
    # turbulence statistics
    # uup = up**2  # uu_prime in 4d
    # uupx_0D  = uup[:,:,:,lat_dist_0D]  + ( uup[:,:,:,lat_dist_0D+1]  - uup[:,:,:,lat_dist_0D]  )*(dist_0D  - lat_dist_0D*dx)/dx
    # uupx_1D  = uup[:,:,:,lat_dist_1D]  + ( uup[:,:,:,lat_dist_1D+1]  - uup[:,:,:,lat_dist_1D]  )*(dist_1D  - lat_dist_1D*dx)/dx
    # uupx_3D  = uup[:,:,:,lat_dist_3D]  + ( uup[:,:,:,lat_dist_3D+1]  - uup[:,:,:,lat_dist_3D]  )*(dist_3D  - lat_dist_3D*dx)/dx
    # uupx_6D  = uup[:,:,:,lat_dist_6D]  + ( uup[:,:,:,lat_dist_6D+1]  - uup[:,:,:,lat_dist_6D]  )*(dist_6D  - lat_dist_6D*dx)/dx
    # uupx_9D  = uup[:,:,:,lat_dist_9D]  + ( uup[:,:,:,lat_dist_9D+1]  - uup[:,:,:,lat_dist_9D]  )*(dist_9D  - lat_dist_9D*dx)/dx
    # uupx_12D = uup[:,:,:,lat_dist_12D] + ( uup[:,:,:,lat_dist_12D+1] - uup[:,:,:,lat_dist_12D] )*(dist_12D - lat_dist_12D*dx)/dx
    # uupx_15D = uup[:,:,:,lat_dist_15D] + ( uup[:,:,:,lat_dist_15D+1] - uup[:,:,:,lat_dist_15D] )*(dist_15D - lat_dist_15D*dx)/dx

    # uupxy_0D  = uupx_0D[:,:,lon_dist_0D]  + ( uupx_0D[:,:,lon_dist_0D+1]  - uupx_0D[:,:,lon_dist_0D]  )*(dist_lon_0D - lon_dist_0D*dy)/dy
    # uupxy_1D  = uupx_1D[:,:,lon_dist_0D]  + ( uupx_1D[:,:,lon_dist_0D+1]  - uupx_1D[:,:,lon_dist_0D]  )*(dist_lon_0D - lon_dist_0D*dy)/dy
    # uupxy_3D  = uupx_3D[:,:,lon_dist_0D]  + ( uupx_3D[:,:,lon_dist_0D+1]  - uupx_3D[:,:,lon_dist_0D]  )*(dist_lon_0D - lon_dist_0D*dy)/dy
    # uupxy_6D  = uupx_6D[:,:,lon_dist_0D]  + ( uupx_6D[:,:,lon_dist_0D+1]  - uupx_6D[:,:,lon_dist_0D]  )*(dist_lon_0D - lon_dist_0D*dy)/dy
    # uupxy_9D  = uupx_9D[:,:,lon_dist_0D]  + ( uupx_9D[:,:,lon_dist_0D+1]  - uupx_9D[:,:,lon_dist_0D]  )*(dist_lon_0D - lon_dist_0D*dy)/dy
    # uupxy_12D = uupx_12D[:,:,lon_dist_0D] + ( uupx_12D[:,:,lon_dist_0D+1] - uupx_12D[:,:,lon_dist_0D] )*(dist_lon_0D - lon_dist_0D*dy)/dy
    # uupxy_15D = uupx_15D[:,:,lon_dist_0D] + ( uupx_15D[:,:,lon_dist_0D+1] - uupx_15D[:,:,lon_dist_0D] )*(dist_lon_0D - lon_dist_0D*dy)/dy

    # uupxyt_0D  = np.mean(uupxy_0D,axis=0)/(uinf**2)
    # uupxyt_1D  = np.mean(uupxy_1D,axis=0)/(uinf**2)
    # uupxyt_3D  = np.mean(uupxy_3D,axis=0)/(uinf**2)
    # uupxyt_6D  = np.mean(uupxy_6D,axis=0)/(uinf**2)
    # uupxyt_9D  = np.mean(uupxy_9D,axis=0)/(uinf**2)
    # uupxyt_12D = np.mean(uupxy_12D,axis=0)/(uinf**2)
    # uupxyt_15D = np.mean(uupxy_15D,axis=0)/(uinf**2)

    # del uup # Free-up some memory

    # uupxy_0D  = uy_var[:,lat_dist_0D]  + ( uy_var[:,lat_dist_0D+1]  - uy_var[:,lat_dist_0D]  )*(dist_0D  - lat_dist_0D*dx)/dx
    # uupxy_2D  = uy_var[:,lat_dist_2D]  + ( uy_var[:,lat_dist_2D+1]  - uy_var[:,lat_dist_2D]  )*(dist_2D  - lat_dist_2D*dx)/dx
    # uupxy_4D  = uy_var[:,lat_dist_4D]  + ( uy_var[:,lat_dist_4D+1]  - uy_var[:,lat_dist_4D]  )*(dist_4D  - lat_dist_4D*dx)/dx
    # uupxy_6D  = uy_var[:,lat_dist_6D]  + ( uy_var[:,lat_dist_6D+1]  - uy_var[:,lat_dist_6D]  )*(dist_6D  - lat_dist_6D*dx)/dx
    # uupxy_8D  = uy_var[:,lat_dist_8D]  + ( uy_var[:,lat_dist_8D+1]  - uy_var[:,lat_dist_8D]  )*(dist_8D  - lat_dist_8D*dx)/dx
    # uupxy_10D = uy_var[:,lat_dist_10D] + ( uy_var[:,lat_dist_10D+1] - uy_var[:,lat_dist_10D] )*(dist_10D - lat_dist_10D*dx)/dx
    # # uupxy_15D = uy_var[:,lat_dist_15D] + ( uy_var[:,lat_dist_15D+1] - uy_var[:,lat_dist_15D] )*(dist_15D - lat_dist_15D*dx)/dx

    # uupxyt_0D  = uupxy_0D/(uinf**2)
    # uupxyt_2D  = uupxy_2D/(uinf**2)
    # uupxyt_4D  = uupxy_4D/(uinf**2)
    # uupxyt_6D  = uupxy_6D/(uinf**2)
    # uupxyt_8D  = uupxy_8D/(uinf**2)
    # uupxyt_10D = uupxy_10D/(uinf**2)
    # uupxyt_15D = uupxy_15D/(uinf**2)
    ###########################################################################
    # wind turbine data
    aoam = np.mean(aoa,axis=(0,2))
    v1m = np.mean(v1,axis=(0,2))
    lm = np.mean(l,axis=(0,2))
    dm = np.mean(d,axis=(0,2))

    rhub = dhub/2
    dist = 0.0
    dr = np.zeros(Nelm)
    for i in range(0,Nelm):
        dist = dist + 0.5*((diameter/2 - rhub)/Nelm)
        dr[i] = rhub + dist
        dist = dist + 0.5*((diameter/2 - rhub)/Nelm)

    rOverR = dr/(diameter/2)

    fl_star = lm/(diameter*rho*uinf**2)
    fd_star = dm/(diameter*rho*uinf**2)
    v1m_star = v1m/uinf
    ###########################################################################
    # store data
    var_holder = {}

    var_holder['diameter'] = diameter
    var_holder['hub_height'] = hub_height
    var_holder['dx'] = dx
    var_holder['dy'] = dy
    var_holder['dt'] = dt
    var_holder['rho'] = rho
    var_holder['Nx'] = Nx
    var_holder['Ny'] = Ny
    var_holder['Nz'] = Nz
    var_holder['tower_xloc'] = tower_xloc
    var_holder['tower_yloc'] = tower_yloc
    var_holder['rotor_xloc'] = rotor_xloc
    var_holder['rotor_yloc'] = rotor_yloc
    var_holder['ix_tower'] = ix_tower
    var_holder['jy_tower'] = jy_tower
    var_holder['ix_rotor'] = ix_rotor
    var_holder['jy_rotor'] = jy_rotor
    var_holder['k_loc'] = k_loc
    var_holder['z_av'] = z_av
    var_holder['ktop'] = ktop
    var_holder['kbot'] = kbot
    var_holder['X2'] = X2
    var_holder['Y2'] = Y2
    var_holder['Z2'] = Z2
    var_holder['X3'] = X3
    var_holder['Y3'] = Y3
    var_holder['Z3'] = Z3
    var_holder['rOverR'] = rOverR
    var_holder['uinf'] = uinf
    var_holder['omega'] = rotspeed
    var_holder['thrust'] = thrust
    var_holder['power_aero'] = power_aero
    var_holder['power_mech'] = power_mech
    var_holder['power_gen'] = power_gen
    var_holder['torque_aero'] = torque_aero
    var_holder['ct'] = ct
    var_holder['cp'] = cp
    var_holder['v0'] = v0
    var_holder['f'] = f
    var_holder['fn'] = fn
    var_holder['ft'] = ft
    var_holder['l'] = l
    var_holder['d'] = d
    var_holder['cl'] = cl
    var_holder['cd'] = cd
    var_holder['aoa'] = aoa
    var_holder['v1'] = v1
    var_holder['um'] = um
    var_holder['vm'] = vm
    var_holder['wm'] = wm
    var_holder['uh'] = uh
    var_holder['uhy'] = uhy
    var_holder['uytz'] = uytz
    var_holder['vytz'] = vytz
    var_holder['wytz'] = wytz
    var_holder['pytz'] = pytz
    var_holder['uz_var'] = uz_var
    var_holder['vz_var'] = vz_var
    var_holder['wz_var'] = wz_var
    var_holder['uy_var'] = uy_var
    var_holder['vy_var'] = vy_var
    var_holder['wy_var'] = wy_var
    var_holder['ux_var'] = ux_1D_var
    var_holder['vx_var'] = vx_1D_var
    var_holder['wx_var'] = wx_1D_var
    var_holder['uhtz'] = uhtz

    var_holder['ux_n3D']   = ux_n3D
    var_holder['ux_0D']    = ux_0D
    var_holder['ux_2D']    = ux_2D
    var_holder['ux_4D']    = ux_4D
    var_holder['ux_6D']    = ux_6D
    var_holder['ux_8D']    = ux_8D
    var_holder['ux_10D']   = ux_10D

    var_holder['vx_0D']    = vx_0D
    var_holder['vx_2D']    = vx_2D
    var_holder['vx_4D']    = vx_4D
    var_holder['vx_6D']    = vx_6D
    var_holder['vx_8D']    = vx_8D
    var_holder['vx_10D']   = vx_10D

    var_holder['uxyt_0D']  = uxyt_0D
    var_holder['uxyt_2D']  = uxyt_2D
    var_holder['uxyt_4D']  = uxyt_4D
    var_holder['uxyt_6D']  = uxyt_6D
    var_holder['uxyt_8D']  = uxyt_8D
    var_holder['uxyt_10D'] = uxyt_10D
    
    var_holder['vxyt_0D']  = vxyt_0D
    var_holder['vxyt_2D']  = vxyt_2D
    var_holder['vxyt_4D']  = vxyt_4D
    var_holder['vxyt_6D']  = vxyt_6D
    var_holder['vxyt_8D']  = vxyt_8D
    var_holder['vxyt_10D'] = vxyt_10D

    var_holder['pxyt_0D']  = pxyt_0D
    var_holder['pxyt_2D']  = pxyt_2D
    var_holder['pxyt_4D']  = pxyt_4D
    var_holder['pxyt_6D']  = pxyt_6D
    var_holder['pxyt_8D']  = pxyt_8D
    var_holder['pxyt_10D'] = pxyt_10D

    var_holder['uxzt_0D']  = uxzt_0D
    var_holder['uxzt_2D']  = uxzt_2D
    var_holder['uxzt_4D']  = uxzt_4D
    var_holder['uxzt_6D']  = uxzt_6D
    var_holder['uxzt_8D']  = uxzt_8D
    var_holder['uxzt_10D'] = uxzt_10D

    var_holder['vxzt_0D']  = vxzt_0D
    var_holder['vxzt_2D']  = vxzt_2D
    var_holder['vxzt_4D']  = vxzt_4D
    var_holder['vxzt_6D']  = vxzt_6D
    var_holder['vxzt_8D']  = vxzt_8D
    var_holder['vxzt_10D'] = vxzt_10D

    var_holder['pxzt_0D']  = pxzt_0D
    var_holder['pxzt_2D']  = pxzt_2D
    var_holder['pxzt_4D']  = pxzt_4D
    var_holder['pxzt_6D']  = pxzt_6D
    var_holder['pxzt_8D']  = pxzt_8D
    var_holder['pxzt_10D'] = pxzt_10D

    # var_holder['uupxyt_0D'] = uupxyt_0D
    # var_holder['uupxyt_2D'] = uupxyt_2D
    # var_holder['uupxyt_4D'] = uupxyt_4D
    # var_holder['uupxyt_6D'] = uupxyt_6D
    # var_holder['uupxyt_8D'] = uupxyt_8D
    # var_holder['uupxyt_10D'] = uupxyt_10D

    var_holder['aoam']     = aoam
    var_holder['v1m_star'] = v1m_star
    var_holder['fl_star']  = fl_star
    var_holder['fd_star']  = fd_star
    var_holder['bpx']      = bpx
    var_holder['bpy']      = bpy
    var_holder['bpz']      = bpz
    var_holder['trb_x']    = rotor_xloc
    var_holder['trb_y']    = rotor_yloc
    var_holder['trb_z']    = rotor_zloc

    # ###############################################################
    # # Create a console object
    # console = Console()

    # # Create a table
    # table = Table(title="Variable Holder Values")

    # # Add columns (you can adjust the widths if needed)
    # table.add_column("Variable", justify="left", style="cyan", no_wrap=True)
    # table.add_column("Value", justify="right", style="magenta")

    # # Add rows to the table from the var_holder dictionary
    # for key, value in var_holder.items():
    #     table.add_row(key, str(value))

    # # Print the table to the console
    # console.print(table)
    # ###############################################################

    # Save low-frequency output variables into .npz file
        # print('##### saving data #####')
    np.savez( os.path.join(f'/anvil/scratch/x-smata/wrf_les_sweep/runs/clockwise/gad_sweep/{case}.npz'),**var_holder)
    del var_holder


def parproc(processes: int, params: Dict[str, Any]) -> None:
    """
    Processes the WRF output files in parallel

    Args:
        processes (int): The number of parallel processes to use
        opt_params (Dict): A dictionary of settings including sample locations if desired
    """

    filelist = sorted(glob.glob(params['output_top_dir'] + '/*/wrfout_d02_0001-01-01_00_00_00'))

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

    # Process the files
    start_time = time.time()
    
    # Parallel pool
    with Pool(processes) as pool:
        pool.map(parallel_process_function, filelist)

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