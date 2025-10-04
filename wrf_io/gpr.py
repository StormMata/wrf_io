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
from scipy.io import savemat
from scipy.io import loadmat
from itertools import product
from multiprocessing import Pool
from rich.console import Console
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d
from matplotlib.gridspec import GridSpec
from MITRotor import IEA10MW, IEA15MW, IEA22MW
from typing import Dict, Any, Optional, List, Tuple, Union
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


# def generate_train_data(casenames, params_path, fields=False, local=False):
def generate_train_data(params: Dict[str, Any], D = float, field_data=False, rotor_data=False, local=False):

    # params, inflow_data, wrfles = load_data(casenames, params_path, fields, local)
    # data = compute_les_data(casenames, params, inflow_data, wrfles, fields)

    combinations = sweep.get_combinations(params=params, D=D)
    formats      = sweep.determine_format(combinations)

    casenames    = sweep.return_case_strings(combinations,formats)

    model_str = params['rotor_model'].lower() + f'_sweep'

    params_path = os.path.join(params['base_dir'],model_str,'opt_params.pkl')

    params, inflow_data, wrfles = load_data(casenames, params_path, local)

    data = compute_les_data(casenames, params, inflow_data, wrfles, field_data, rotor_data)

    return data


def load_data(casenames, params_path, local):

    params      = postproc.load_params(params_path)
    inflow_data = postproc.extract_sounding(params=params, local=local)
    wrfles      = postproc.load_data(params=params, casenames=casenames, local=local)

    return params, inflow_data, wrfles


def compute_les_data(casenames, params, inflow_data, les_data, field_data, rotor_data):

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

    wrf_U_inf     = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_wdir_inf  = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
 
    wrf_U_disk    = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_wdir_disk = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
 
    wrf_CL        = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_CD        = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
 
    wrf_CL_real   = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_CD_real   = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
 
    wrf_L         = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_D         = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
 
    wrf_L_real    = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_D_real    = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
 
    wrf_FN        = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_FT        = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
 
    wrf_FN_real   = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_FT_real   = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
 
    wrf_pow       = np.zeros(len(casenames),dtype='longdouble')
    wrf_thr       = np.zeros(len(casenames),dtype='longdouble')
 
    wrf_pow_real  = np.zeros(len(casenames),dtype='longdouble')
    wrf_thr_real  = np.zeros(len(casenames),dtype='longdouble')
 
    Uhub          = np.zeros(len(casenames), dtype=float)
    wrf_pitch     = np.zeros(len(casenames), dtype=float)
    wrf_omg       = np.zeros(len(casenames), dtype=float)
 
    wrf_cot_rot   = np.zeros(len(casenames), dtype=float)
    wrf_ind_rot   = np.zeros(len(casenames), dtype=float)
 
    r_ann         = np.zeros((Nelm, len(casenames)),dtype='longdouble')
    shears_ann    = np.zeros((Nelm, len(casenames)),dtype='longdouble')
    veers_ann     = np.zeros((Nelm, len(casenames)),dtype='longdouble')
 
    wrf_cot_ann   = np.zeros((Nelm, len(casenames)),dtype='longdouble')
    wrf_ind_ann   = np.zeros((Nelm, len(casenames)),dtype='longdouble')
 
    wrf_cot_loc   = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')
    wrf_ind_loc   = np.zeros((Nelm, Nsct, len(casenames)),dtype='longdouble')

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

        print(case)

        # Rotor radius
        R     = les_data[count]['radius']
        R_hub = les_data[count]['hub_radius']

        hub_height = les_data[count]['hub_height']

        # Extract background flow
        u_func = interp1d(inflow_data[case][:,0], inflow_data[case][:,1], kind='linear')
        v_func = interp1d(inflow_data[case][:,0], inflow_data[case][:,2], kind='linear')

        # u and v velocity functions
        u_inf = u_func(Y)
        v_inf = v_func(Y)

        # Compute hub height velocity for normalization
        U_hub = np.sqrt(u_func(hub_height)**2 + v_func(hub_height)**2)

        U_inf = np.sqrt(u_inf**2 + v_inf**2).T

        wdir_inf = np.atan2(v_inf,u_inf).T

        r  = les_data[count]['rOverR'] * R
        mu = les_data[count]['rOverR']

        # Azimuthal coordinates
        theta = np.linspace(0, 2*np.pi, Nsct)

        # Extract velocity components at the rotor disk
        u_rotor = np.mean(les_data[count]['v1'],axis=0)
        # v_rotor = np.mean(les_data[count]['v'],axis=0)
        # w_rotor = np.mean(les_data[count]['w'],axis=0)

        # Compute induction
        # ind         = 1 - u_rotor / U_inf 
        ind         = 1 - u_rotor / u_inf.T
        ind_annulus = postproc.annulus_average(theta, ind)
        ind_rotor   = postproc.rotor_average(mu, ind_annulus, R_hub, R)

        # Compute new rotor disk velocities based on background flow and rotor-averaged axial induction
        # u_rot_avg = ((((1-ind_rotor) * u_inf)**2 + v_inf**2)**(1/2) * np.cos(np.atan2(v_inf, u_inf))).T
        # v_rot_avg = ((((1-ind_rotor) * u_inf)**2 + v_inf**2)**(1/2) * np.sin(np.atan2(v_inf, u_inf))).T

        u_rot_avg = U_inf * (1 - ind_rotor) * np.cos(wdir_inf)
        v_rot_avg = U_inf * (1 - ind_rotor) * np.sin(wdir_inf)

        # u_ann_avg = U_inf * (1 - ind_annulus) * np.cos(wdir_inf)
        # v_ann_avg = U_inf * (1 - ind_annulus) * np.sin(wdir_inf)

        U_disk    = np.sqrt(u_rot_avg**2 + v_rot_avg**2)
        wdir_disk = np.atan2(v_rot_avg, u_rot_avg)

        r_mat  = (np.ones_like(u_inf) * r).T
        mu_mat = (np.ones_like(u_inf) * mu).T

        chord = rotor.chord_func(mu)
        chord =  (np.ones_like(u_inf) * chord).T

        # Radial twist
        twist = rotor.twist_func(mu)

        # Solidity
        sigma_bem = 3 * chord / (2 * np.pi * r_mat)

        # Local solidity
        sigma_les = 3/Nsct

        # Tip speed ratio
        omega = np.mean(les_data[count]['omega'], axis=0) * 2 * np.pi / 60

        Vax, Vtn_NR, _ = postproc.rotGlobalToLocal(Nelm,Nsct,u_rot_avg,v_rot_avg,np.zeros_like(u_rotor))

        Vtn = omega * r_mat - Vtn_NR

        # Relative velocity
        W = np.sqrt(Vax**2 + Vtn**2)

        # Inflow angleCompu
        phi = np.atan2(Vax,Vtn)

        pitch = np.deg2rad(np.mean(les_data[count]['pitch'], axis=0))

        aoa  = phi - twist[:, np.newaxis] - pitch
        Cl, Cd = rotor.clcd(mu_mat, aoa)

        # Axial coefficient
        Cax = Cl * np.cos(phi) + Cd * np.sin(phi)

        # Local CT
        ct = sigma_bem * (W/U_hub)**2 * Cax
        # ct = sigma_bem * (W/U_inf)**2 * Cax

        rho = 1.225

        L = 1/2 * rho * chord * (Cl * W**2)
        D = 1/2 * rho * chord * (Cd * W**2)

        FN = L * np.cos(phi) + D * np.sin(phi)
        FT = L * np.sin(phi) - D * np.cos(phi)

        dr = (R - rotor.hub_radius)/Nelm

        T = np.sum(FN * dr * sigma_les)
        P = np.sum(FT * r_mat * dr * sigma_les * omega)

        wrf_ind_loc[:,:,count]   = ind
        wrf_cot_loc[:,:,count]   = ct
 
        wrf_cot_ann[:,count]     = postproc.annulus_average(theta, ct)
        wrf_ind_ann[:,count]     = ind_annulus
 
        wrf_cot_rot[count]       = postproc.rotor_average(mu, wrf_cot_ann[:,count], R_hub, R)
        wrf_ind_rot[count]       = ind_rotor
 
        r_ann[:,count]           = r
        shears_ann[:,count]      = shears[count] * np.ones_like(r)
        veers_ann[:,count]       = veers[count] * np.ones_like(r)
 
        wrf_CL[:,:,count]        = Cl
        wrf_CD[:,:,count]        = Cd
 
        wrf_CL_real[:,:,count]   = np.mean(les_data[count]['cl'], axis=0)
        wrf_CD_real[:,:,count]   = np.mean(les_data[count]['cd'], axis=0)
 
        wrf_L[:,:,count]         = L
        wrf_D[:,:,count]         = D
 
        wrf_L_real[:,:,count]    = np.mean(les_data[count]['l'], axis=0)
        wrf_D_real[:,:,count]    = np.mean(les_data[count]['d'], axis=0)
 
        wrf_FN[:,:,count]        = FN
        wrf_FT[:,:,count]        = FT
 
        wrf_FN_real[:,:,count]   = np.mean(les_data[count]['fn'], axis=0)
        wrf_FT_real[:,:,count]   = np.mean(les_data[count]['ft'], axis=0)
 
        wrf_thr[count]           = T
        wrf_pow[count]           = P
 
        wrf_thr_real[count]      = np.mean(les_data[count]['thrust'], axis=0)[0]
        wrf_pow_real[count]      = np.mean(les_data[count]['power_aero'], axis=0)[0]
 
        Uhub[count]              = U_hub
        wrf_omg[count]           = np.mean(les_data[count]['omega'], axis=0)
        wrf_pitch[count]         = pitch
        wrf_U_inf[:,:,count]     = U_inf
        wrf_wdir_inf[:,:,count]  = wdir_inf

        wrf_U_disk[:,:,count]    = U_disk
        wrf_wdir_disk[:,:,count] = wdir_disk

    if field_data:
        np.save(params['field_data_path'] + 'sweep_names.npy', casenames)
        np.save(params['field_data_path'] + 'U_inf.npy', wrf_U_inf)
        np.save(params['field_data_path'] + 'dir_inf.npy', wrf_wdir_inf)
        np.save(params['field_data_path'] + 'U_disk.npy', wrf_U_disk)
        np.save(params['field_data_path'] + 'dir_disk.npy', wrf_wdir_disk)
        np.save(params['field_data_path'] + 'Uhub.npy', Uhub)
        np.save(params['field_data_path'] + 'pitch.npy', wrf_pitch)
        np.save(params['field_data_path'] + 'wrf_omg.npy', wrf_omg)


        print(f'Field data saved at {params["field_data_path"]}')

    if rotor_data:
        return {
            'lift'          : wrf_L,
            'drag'          : wrf_D,
            'normal'        : wrf_FN,
            'tangential'    : wrf_FT,
            'thrust'        : wrf_thr,
            'power'         : wrf_pow,
        }
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
            'cotp_rotor'    : wrf_cot_rot / (1 - wrf_ind_rot)**2,
            'cotp_annulus'  : wrf_cot_ann / (1 - wrf_ind_ann)**2,
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
        'cotp_rotor',
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


def load_scalars(opt_params: Dict) -> Dict[str, object]:
    """
    Load scalers/encoders for inference, combining gp_base_path with per-item paths.

    Required:
      opt_params['gp_base_path'] = base directory
      And for each of these keys, a path RELATIVE to base (strings may accidentally be single-item tuples):
        'scaler_cot_annulus', 'scaler_ind_annulus',
        'scaler_cot_rotor', 'scaler_ind_rotor',
        'scaler_shears_rotor', 'scaler_veers_rotor',
        'scaler_shears_annulus', 'scaler_veers_annulus',
        'encoder_rotor', 'encoder_annulus'
      Optionally:
        'scaler_cot_local', 'scaler_ind_local' (if omitted, identity scalers will be created)

    Returns:
      Dict[str, object] for use as `scalars` in scale_and_encode(..., training=False, scalars=...)
    """
    if 'gp_base_path' not in opt_params:
        raise ValueError("Missing required opt_params['gp_base_path']")

    base = os.path.expanduser(os.path.expandvars(opt_params['gp_base_path']))

    # Required items saved during training (except local, which we can identity-pass if absent)
    required_keys = [
        'scaler_cot_annulus', 'scaler_ind_annulus',
        'scaler_cot_rotor', 'scaler_ind_rotor',
        'scaler_shears_rotor', 'scaler_veers_rotor',
        'scaler_shears_annulus', 'scaler_veers_annulus',
        'encoder_rotor', 'encoder_annulus', 'scaler_cotp_rotor',
    ]
    optional_local_keys = ['scaler_cot_local', 'scaler_ind_local']

    def _unwrap_path(p: Union[str, list, tuple], key: str) -> str:
        # Handle accidental trailing commas making single-item tuples
        if isinstance(p, (list, tuple)):
            if len(p) == 1 and isinstance(p[0], str):
                return p[0]
            raise ValueError(f"Expected a single path string for {key}, got {p}")
        if not isinstance(p, str):
            raise ValueError(f"Expected a string path for {key}, got {type(p)}")
        return p

    def _resolve(base_dir: str, rel_path: str) -> str:
        # Treat provided path as relative to base even if it starts with '/'
        rel_path = os.path.expanduser(os.path.expandvars(rel_path))
        rel_path = rel_path.lstrip(os.sep)
        return os.path.normpath(os.path.join(base_dir, rel_path))

    scalars: Dict[str, Any] = {}
    missing_files = []

    # Load required non-local items
    for key in required_keys:
        if key not in opt_params:
            continue
        rel = _unwrap_path(opt_params[key], key)
        fpath = _resolve(base, rel)
        if not os.path.isfile(fpath):
            missing_files.append((key, fpath))
            continue
        with open(fpath, "rb") as f:
            scalars[key] = pickle.load(f)

    # If only rotor encoder provided, reuse for annulus
    if 'encoder_annulus' not in scalars and 'encoder_rotor' in scalars:
        scalars['encoder_annulus'] = scalars['encoder_rotor']

    # Local scalers: load if provided; otherwise create identity scalers
    def _identity_scaler():
        s = StandardScaler()
        s.fit(np.zeros((1, 1), dtype=float))  # n_features_in_ = 1
        return s

    for key in optional_local_keys:
        if key in opt_params:
            rel = _unwrap_path(opt_params[key], key)
            fpath = _resolve(base, rel)
            if not os.path.isfile(fpath):
                raise FileNotFoundError(f"Missing file for {key}: {fpath}")
            with open(fpath, "rb") as f:
                scalars[key] = pickle.load(f)
        else:
            scalars.setdefault(key, _identity_scaler())

    # Final validation: everything scale_and_encode needs when training=False
    must_have = [
        'scaler_cot_local', 'scaler_ind_local',
        'scaler_cot_annulus', 'scaler_ind_annulus',
        'scaler_cot_rotor', 'scaler_ind_rotor',
        'scaler_shears_rotor', 'scaler_veers_rotor',
        'scaler_shears_annulus', 'scaler_veers_annulus',
        'encoder_rotor', 'encoder_annulus','scaler_cotp_rotor',
    ]
    missing = [k for k in must_have if k not in scalars]
    if missing:
        msg = "Missing required scalers/encoders: " + ", ".join(missing)
        if missing_files:
            msg += ". Some expected files were not found: " + ", ".join(f"{k}:{p}" for k, p in missing_files)
        raise FileNotFoundError(msg)

    return scalars


def save_dataset(params: Dict[str, Any], data):
    # Generate MATLAB tables of standardized inputs

    X_rot = np.column_stack([data['cot_rotor'], data['shears_rotor'], data['veers_rotor']])
    X_ann = np.column_stack([data['r_annulus'].flatten(), data['cot_annulus'].flatten(), data['shears_annulus'].flatten(), data['veers_annulus'].flatten()])

    y_rot = data['ind_rotor']
    y_ann = data['ind_annulus'].flatten()

    savemat(params['gp_dir']+ 'wrf_10MW_ann.mat', {'X': X_ann, 'y': y_ann.reshape(-1, 1)})
    savemat(params['gp_dir']+ 'wrf_10MW_rot.mat', {'X': X_rot, 'y': y_rot.reshape(-1, 1)})

    X_rot = np.hstack([X_rot, data['shear_regime_rotor']])
    X_ann = np.hstack([X_ann, data['shear_regime_annulus']])

    savemat(params['gp_dir']+ 'wrf_10MW_ann_OH.mat', {'X': X_ann, 'y': y_ann.reshape(-1, 1)})
    savemat(params['gp_dir']+ 'wrf_10MW_rot_OH.mat', {'X': X_rot, 'y': y_rot.reshape(-1, 1)})

    print(f'Data saved to {params["gp_dir"]}')

def save_scalars(params: Dict[str, Any], scalars):
    with open(os.path.join(params['gp_dir'], 'scaler_wrf_cot_ann.pkl'), 'wb') as f:
        pickle.dump(scalars['scaler_cot_annulus'], f)

    with open(os.path.join(params['gp_dir'], 'scaler_wrf_ind_ann.pkl'), 'wb') as f:
        pickle.dump(scalars['scaler_ind_annulus'], f)

    with open(os.path.join(params['gp_dir'], 'scaler_wrf_cot_rot.pkl'), 'wb') as f:
        pickle.dump(scalars['scaler_cot_rotor'], f)

    with open(os.path.join(params['gp_dir'], 'scaler_wrf_ind_rot.pkl'), 'wb') as f:
        pickle.dump(scalars['scaler_ind_rotor'], f)

    with open(os.path.join(params['gp_dir'], 'scaler_shears_rot.pkl'), 'wb') as f:
        pickle.dump(scalars['scaler_shears_rotor'], f)

    with open(os.path.join(params['gp_dir'], 'scaler_veers_rot.pkl'), 'wb') as f:
        pickle.dump(scalars['scaler_veers_rotor'], f)

    with open(os.path.join(params['gp_dir'], 'scaler_shears_ann.pkl'), 'wb') as f:
        pickle.dump(scalars['scaler_shears_annulus'], f)

    with open(os.path.join(params['gp_dir'], 'scaler_veers_ann.pkl'), 'wb') as f:
        pickle.dump(scalars['scaler_veers_annulus'], f)

    with open(os.path.join(params['gp_dir'], 'encoder_rot.pkl'), 'wb') as f:
        pickle.dump(scalars['encoder_rotor'], f)

    with open(os.path.join(params['gp_dir'], 'encoder_ann.pkl'), 'wb') as f:
        pickle.dump(scalars['encoder_annulus'], f)

    with open(os.path.join(params['gp_dir'], 'scaler_wrf_cotp_rot.pkl'), 'wb') as f:
        pickle.dump(scalars['scaler_cotp_rotor'], f)

    print(f'Scalars saved to {params["gp_dir"]}')