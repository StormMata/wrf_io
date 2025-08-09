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
from wrf_io import postproc
from scipy.io import loadmat
from itertools import product
from multiprocessing import Pool
from rich.console import Console
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d
from matplotlib.gridspec import GridSpec
from MITRotor import IEA10MW, IEA15MW, IEA22MW
from typing import Dict, Any, Optional, List, Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def print_stats(variables: Dict) -> None:
    """
    Print descriptive statistics (mean, std, min, max) for each array in the dictionary.

    Parameters:
        variables (dict): A dictionary where keys are variable names and values are numpy arrays.
    """
    print("Variable Statistics:")
    print("-" * 75)
    print(f"{'Name':<20} | {'Mean':>10} | {'Std Dev':>10} | {'Min':>10} | {'Max':>10}")
    print("-" * 75)

    for name, arr in variables.items():
        arr_flat = np.array(arr).flatten()
        mean = np.mean(arr_flat)
        std = np.std(arr_flat)
        min_val = np.min(arr_flat)
        max_val = np.max(arr_flat)
        print(f"{name:<20} | {mean:>10.5f} | {std:>10.5f} | {min_val:>10.5f} | {max_val:>10.5f}")


def merge_data_dicts(d1, d2):
    """
    Combine two dictionaries of arrays by concatenating values with matching keys.
    The axis of concatenation is inferred from array shape:
    - If arrays are 1D: concatenate along axis 0
    - If arrays are 2D: concatenate along axis 1
    - If arrays are 3D: concatenate along axis 2
    - For column_stack cases (e.g. r_ann), we handle them specifically by key name.
    """
    merged = {}
    for key in d1:
        a1 = np.array(d1[key])
        a2 = np.array(d2[key])

        if key == "r_ann":
            merged[key] = np.column_stack([a1, a2])
        elif a1.ndim == 1:
            merged[key] = np.concatenate([a1, a2], axis=0)
        elif a1.ndim == 2:
            merged[key] = np.concatenate([a1, a2], axis=1)
        elif a1.ndim == 3:
            merged[key] = np.concatenate([a1, a2], axis=2)
        else:
            raise ValueError(f"Cannot determine concat axis for key: {key} with ndim={a1.ndim}")
    return merged


def generate_train_data(casenames, params_path, fields=False, local=False):

    params, inflow_data, wrfles = load_data(casenames, params_path, fields, local)

    data = compute_les_data(casenames, params, inflow_data, wrfles, fields)

    return data


def load_data(casenames, params_path, fields, local):

    params      = postproc.load_params(params_path)
    inflow_data = postproc.extract_sounding(params=params, local=local)
    wrfles      = postproc.load_data(params=params, casenames=casenames, local=local)

    return params, inflow_data, wrfles


def compute_les_data(casenames, params, inflow_data, les_data, fields):

    exclude = params['excluded_pairs']
    cases = [
        pair for pair in product(params['shear'], params['veer'])
        if pair not in exclude
    ]

    shears = [pair[0] for pair in cases]
    veers  = [pair[1] for pair in cases]

    shears = np.array(shears)  
    veers  = np.array(veers)

    Nelm = les_data[0]['Nelm']
    Nsct = les_data[0]['Nsct']

    t = np.linspace(0,2*np.pi,Nsct)
    r = np.linspace(0,1,Nelm)

    rho = 1.225

    z_hh = les_data[0]['hub_height']

    R, T = np.meshgrid(r, t)

    X = R * np.sin(T)
    Y = (R * np.cos(T)) * les_data[0]['radius'] + z_hh

    wrf_vax      = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_vtn      = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')

    wrf_vax_real = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_vtn_real = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')

    wrf_vtn_NR   = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_vtn_NR_real= np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')

    wrf_U_inf    = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_wdir_inf = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')

    wrf_U        = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_wdir     = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')

    wrf_W        = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_W_real   = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')

    wrf_CL       = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_CD       = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')

    wrf_CL_real  = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_CD_real  = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')

    wrf_L        = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_D        = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')

    wrf_L_real   = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_D_real   = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')

    wrf_FN       = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_FT       = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')

    wrf_FN_real  = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_FT_real  = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')

    wrf_phi      = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_phi_real = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')

    wrf_aoa      = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_aoa_real = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')

    wrf_cax      = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')

    wrf_pow      = np.zeros(len(casenames),dtype='longdouble')
    wrf_thr      = np.zeros(len(casenames),dtype='longdouble')

    wrf_pow_real = np.zeros(len(casenames),dtype='longdouble')
    wrf_thr_real = np.zeros(len(casenames),dtype='longdouble')

    Uhub         = np.zeros(len(casenames), dtype=float)

    wrf_tsr      = np.zeros(len(casenames), dtype=float)
    wrf_omg      = np.zeros(len(casenames), dtype=float)

    wrf_cot_rot  = np.zeros(len(casenames), dtype=float)
    wrf_ind_rot  = np.zeros(len(casenames), dtype=float)

    r_ann        = np.zeros((Nelm, len(casenames)),dtype='longdouble')
    shears_ann   = np.zeros((Nelm, len(casenames)),dtype='longdouble')
    veers_ann    = np.zeros((Nelm, len(casenames)),dtype='longdouble')

    wrf_cot_ann  = np.zeros((Nelm, len(casenames)),dtype='longdouble')
    wrf_ind_ann  = np.zeros((Nelm, len(casenames)),dtype='longdouble')

    wrf_cot_loc  = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_ind_loc  = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')

    if params['turb_model'] == 'iea10MW':
        rotor = IEA10MW()
    elif params['turb_model'] == 'iea15MW':
        rotor = IEA15MW()
    elif params['turb_model'] == 'iea22MW':
        rotor = IEA22MW()
    else:
        print(f'Unrecognized turbine {params["turb_model"]}.')
        return

    for count,case in enumerate(casenames):

        # Rotor radius
        R = les_data[count]['radius']

        # Extract background flow
        u_func = interp1d(inflow_data[case][:,0], inflow_data[case][:,1], kind='linear')
        v_func = interp1d(inflow_data[case][:,0], inflow_data[case][:,2], kind='linear')

        # u and v velocity functions
        u_inf = u_func(Y)
        v_inf = v_func(Y)

        U_inf = np.sqrt(u_inf**2 + v_inf**2).T

        wdir_inf = np.atan2(v_inf,u_inf).T

        # Extract u and v velocity components at the rotor disk
        # u_rotor = np.mean(les_data[count]['shapiroM'][0] * les_data[count]['u'],axis=0)
        u_rotor = np.mean(les_data[count]['u'],axis=0)
        v_rotor = np.mean(les_data[count]['v'],axis=0)
        w_rotor = np.mean(les_data[count]['w'],axis=0)

        # Magnitude of wind speed at rotor disk
        U = np.sqrt(u_rotor**2 + v_rotor**2)

        # Local wind misalignment angle at the rotor disk
        wdir = np.atan2(v_rotor,u_rotor)

        # Nondimensional radial positions
        r = les_data[count]['rOverR']
        r_mat =  (np.ones_like(wdir.T) * r).T

        # Azimuthal coordinates
        theta = np.linspace(0, 2*np.pi, les_data[count]['Nsct'])

        # Freestream wind speed at hub height
        U_hub = np.sqrt(u_func(z_hh)**2 + v_func(z_hh)**2)

        # Radial chord lengths
        chord = rotor.chord_func(r)
        chord =  (np.ones_like(wdir.T) * chord).T

        # Radial twist
        twist = rotor.twist_func(r)
        
        # Local solidity
        sigma_r = 3 * chord / (2 * np.pi * r_mat * R)

        # Local solidity
        sigma = 3/Nsct

        # Tip speed ratio
        omega = np.mean(les_data[count]['omega'], axis=0) * 2 * np.pi / 60
        tsr = np.mean(omega*R/(U_hub), axis=0) 

        trbYaw = np.mean(les_data[count]['yaw'], axis=0) 

        Vax, Vtn_NR, _ = postproc.rotGlobalToLocal(Nelm,Nsct,u_rotor,v_rotor,np.zeros_like(u_rotor))

        Vtn = omega * r_mat * R - Vtn_NR

#-------------------------------------------
        # Vax = u_rotor
        # Vtn = (
        #     (1 + 0) * omega * r_mat * R
        #     - v_rotor
        #     * np.cos(t)
        #     * np.sin(0)
        # )
#-------------------------------------------

        # phi = np.arctan2(Vax, Vtn)
        # aoa = phi - rotor.twist(r_mat)
        # aoa = np.clip(aoa, -np.pi / 2, np.pi / 2)

        # Relative velocity
        W = np.sqrt(Vax**2 + Vtn**2)

        # Inflow angle
        phi = np.atan2(Vax,Vtn)

        aoa  = phi - twist[:, np.newaxis]
        Cl, Cd = rotor.clcd(r_mat, aoa)

        # Axial coefficient
        Cax = Cl * np.cos(phi) + Cd * np.sin(phi)

        # Local CT
        # ct = sigma_r * (W/U_hub)**2 * Cax
        ct = sigma_r * (W/(np.sqrt(u_func(Y)**2 + v_func(Y)**2)).T)**2 * Cax

        L = 1/2 * rho * chord * (Cl * W**2)
        D = 1/2 * rho * chord * (Cd * W**2)

        FN = L * np.cos(phi) + D * np.sin(phi)
        FT = L * np.sin(phi) - D * np.cos(phi)

        dr = (R - rotor.hub_radius)/Nelm

        T = np.sum(FN * dr * sigma)
        P = np.sum(FT * r_mat * R * dr * sigma * omega)

        wrf_vax[:,:,count]      = Vax
        wrf_vtn[:,:,count]      = Vtn

        wrf_vax_real[:,:,count] = np.mean(les_data[count]['v1'], axis=0)
        wrf_vtn_real[:,:,count] = np.mean(les_data[count]['v_tan'], axis=0)
    
        # wrf_vtn_NR[:,:,count]   = Vtn_NR
        wrf_vtn_NR_real[:,:,count] = np.mean(les_data[count]['v_tan_no_rot'], axis=0)

        wrf_phi[:,:,count]      = phi
        wrf_phi_real[:,:,count] = np.deg2rad(np.mean(les_data[count]['phi'], axis=0))

        wrf_aoa[:,:,count]      = aoa
        wrf_aoa_real[:,:,count] = np.deg2rad(np.mean(les_data[count]['aoa'], axis=0))

        wrf_W[:,:,count]        = W
        wrf_W_real[:,:,count]   = np.mean(les_data[count]['vrel'], axis=0)

        wrf_cax[:,:,count]      = Cax

        # wrf_cot_rot[count]      = postproc.rotor_average(ct,r,theta)
        # wrf_ind_rot[count]      = postproc.rotor_average(1 - u_rotor/ U_hub,r,theta)

        wrf_cot_rot[count]      = postproc.rotor_average(ct,np.linspace(0,0.99999,26),theta)
        wrf_ind_rot[count]      = postproc.rotor_average(1 - u_rotor/ u_func(Y),theta)

        wrf_cot_ann[:,count]    = postproc.annulus_average(ct,theta)
        wrf_ind_ann[:,count]    = postproc.annulus_average(1 - u_rotor/ U_hub,theta)

        r_ann[:,count]          = r
        shears_ann[:,count]     = shears[count] * np.ones_like(r)
        veers_ann[:,count]      = veers[count] * np.ones_like(r)

        wrf_cot_loc[:,:,count]  = ct
        wrf_ind_loc[:,:,count]  = 1 - u_rotor/ U_hub

        wrf_CL[:,:,count]       = Cl
        wrf_CD[:,:,count]       = Cd

        wrf_CL_real[:,:,count]  = np.mean(les_data[count]['cl'], axis=0)
        wrf_CD_real[:,:,count]  = np.mean(les_data[count]['cd'], axis=0)

        wrf_L[:,:,count]        = L
        wrf_D[:,:,count]        = D

        wrf_L_real[:,:,count]   = np.mean(les_data[count]['l'], axis=0)
        wrf_D_real[:,:,count]   = np.mean(les_data[count]['d'], axis=0)

        wrf_FN[:,:,count]       = FN
        wrf_FT[:,:,count]       = FT

        wrf_FN_real[:,:,count]  = np.mean(les_data[count]['fn'], axis=0)
        wrf_FT_real[:,:,count]  = np.mean(les_data[count]['ft'], axis=0)

        wrf_thr[count]          = T
        wrf_pow[count]          = P

        wrf_thr_real[count]     = np.mean(les_data[count]['thrust'], axis=0)[0]
        wrf_pow_real[count]     = np.mean(les_data[count]['power_aero'], axis=0)[0]

        Uhub[count]             = U_hub
        wrf_tsr[count]          = tsr
        wrf_omg[count]          = omega[0]

        wrf_U_inf[:,:,count]    = U_inf
        wrf_wdir_inf[:,:,count] = wdir_inf
        
        wrf_U[:,:,count]        = U
        wrf_wdir[:,:,count]     = wdir

    if fields:
        pass
    else:
        return {
            'r_annulus'     : r_ann,
            'cot_local'     : wrf_cot_loc,
            'ind_local'     : wrf_ind_loc,
            'cot_annulus'   : wrf_cot_ann,
            'ind_annulus'   : wrf_ind_ann,
            'cot_rotor'     : wrf_cot_rot,
            'ind_rotor'     : wrf_ind_rot,
            'shears_rotor'  : shears,
            'veers_rotor'   : veers,
            'shears_annulus': shears_ann,
            'veers_annulus' : veers_ann,
        }


def scale_and_encode(input_dict, training: bool, scalars: Optional[Dict] = None):
    scaled_dict = {}
    scalers_dict = {} if training else scalars

    # Passthrough
    passthrough_keys = ['r_annulus']

    # Keys to standard scale
    standard_keys = [
        'cot_local', 'ind_local',
        'cot_annulus', 'ind_annulus',
        'cot_rotor', 'ind_rotor',
        'shears_rotor', 'veers_rotor',
        'shears_annulus', 'veers_annulus',
    ]

    # Compute regimes from shear and veer
    def compute_regimes(shear, veer, suffix):
        regimes = (
            (shear.flatten() != 0).astype(int) * 2 + (veer.flatten() != 0).astype(int)
        ).reshape(-1, 1)

        if training:
            encoder = OneHotEncoder(sparse_output=False)
            onehot = encoder.fit_transform(regimes)
            scalers_dict[f'encoder_{suffix}'] = encoder
        else:
            encoder = scalars[f'encoder_{suffix}']
            onehot = encoder.transform(regimes)

        scaled_dict[f'shear_regime_{suffix}'] = onehot

    compute_regimes(input_dict['shears_rotor'], input_dict['veers_rotor'], 'rotor')
    compute_regimes(input_dict['shears_annulus'], input_dict['veers_annulus'], 'annulus')

    # Handle passthrough keys
    for key in passthrough_keys:
        scaled_dict[key] = input_dict[key]

    # Scale remaining keys
    for key in standard_keys:
        value = input_dict[key]

        if training:
            scaler = StandardScaler()
        else:
            scaler = scalars[f'scaler_{key}']

        # Reshape based on dimensions — always treat as (num_points, 1) for 1D/2D
        if value.ndim == 1:
            data = value[:, None]
        elif value.ndim == 2:
            data = value.flatten()[:, None]  # flatten grid -> (num_points, 1)
        elif value.ndim == 3:
            # data = value.reshape(-1, value.shape[-1])  # e.g., (N, features)
            data = value.flatten()[:, None]  # flatten grid -> (num_points, 1)
        else:
            raise ValueError(f"Unsupported shape for key '{key}': {value.shape}")

        # Fit or transform
        if training:
            scaled_data = scaler.fit_transform(data)
            scalers_dict[f'scaler_{key}'] = scaler
            print(f"{key}: scaler trained with {scaler.n_features_in_} features")
        else:
            print(f"[{key}] training={training}, scaler expects {scaler.n_features_in_}, got {data.shape[1]}")

            scaled_data = scaler.transform(data)

        # Reshape back to original
        if value.ndim == 1:
            scaled_value = scaled_data.ravel()
        elif value.ndim == 2:
            scaled_value = scaled_data.ravel().reshape(value.shape)
        elif value.ndim == 3:
            scaled_value = scaled_data.reshape(value.shape)

        scaled_dict[key] = scaled_value

    return scaled_dict, scalers_dict if training else None


# def save_dataset():
    