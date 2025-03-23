import os
import shutil
import platform
import subprocess
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from wrf_io import preproc
from rich.table import Table
from itertools import product
from rich.console import Console
from scipy.interpolate import interp1d
from typing import Dict, Any, List, Tuple, Optional


def summary_table(params: dict) -> None:
    """
    Generate summary table and print to screen.

    Args:
        params (Dict): A dictionary of settings
    """

    # Create a Rich table summary
    console = Console()
    combinations = [pair for pair in product(params['shear'], params['veer']) if pair not in params['excluded_pairs']]

    # Determine the number of digits for formatting
    int_digits, frac_digits = determine_format(params['shear'], params['veer'])

    # Display combinations
    table = Table(title = "Combinations")
    table.add_column("Case",  justify = "right")
    table.add_column("Shear", justify = "right", style = "green")
    table.add_column("Veer",  justify = "right", style = "red")

    # Format each value for table display with decimal points
    for idx, (s, v) in enumerate(combinations, start=1):
        shear_str = f"{s: {int_digits + frac_digits + 1}.{frac_digits}f}"
        veer_str  = f"{v: {int_digits + frac_digits + 1}.{frac_digits}f}"
        table.add_row(str(idx), shear_str, veer_str)

    console.print(table)


def format_value(val: float, int_digits: int, frac_digits: int) -> str:
    """
    Format directory name values with the following rules:
    1. The number of digits before the decimal point (int_digits) is consistent across all values.
    2. The number of digits after the decimal point (frac_digits) is consistent across all values.
    3. Decimal points are removed in the output (e.g., 3.0 becomes 300).
    4. A minus sign is replaced with 'n' (e.g., -3.0 becomes n300).

    Args:
        val (float): The value to format.
        int_digits (int): Number of digits before the decimal point.
        frac_digits (int): Number of digits after the decimal point.

    Returns:
        str: The formatted string.
    """

    if int_digits == 0:
        # Format with only fractional part
        formatted = f"{abs(val):.{frac_digits}f}".split('.')[-1]  # Keep only decimal part
    else:
        # Standard formatting with integer and fractional part
        formatted = f"{abs(val):0{int_digits}.{frac_digits}f}".replace('.', '')

    # Add 'n' for negative values or return the positive string
    return f"n{formatted}" if val < 0 else formatted


def get_combinations(params: Dict[str, Any], D: Optional[float] = None, force: Optional[bool] = False, precision: Optional[int] = 6) -> list:
    """
    Get the full list of shear/veer combinations as a list for iterating over
    
    Args:
        params (Dict): A dictionary of discrete shear and veer values and excluded pairs if necessary
        D (float): Turbine diameter. If params['shear_type'] == 'Total', this is used to calculate the equivalent shear rate
    
    Returns:
        list: a list of combinations
    """

    if params['shear_type'] == 'Rate':

        combinations = [pair for pair in product(params['shear'], params['veer']) if pair not in params['excluded_pairs']]

    elif params['shear_type'] == 'Total':

        if force:

            combinations = [pair for pair in product(params['shear'], params['veer']) if pair not in params['excluded_pairs']]

        else:
            if D is None:
                raise ValueError("ERROR: Turbine diameter 'D' must be provided when 'shear_type' is 'Total'.")

            equivalent_shear_rate = np.round(np.array(params['shear']) / D, precision)
            equivalent_veer_rate  = np.round(np.array(params['veer']) / D, precision)

            combinations = [pair for pair in product(equivalent_shear_rate, equivalent_veer_rate) if pair not in params['excluded_pairs']]

    else:

        raise ValueError(f'ERROR. Unrecognized shear type "{params['shear_type']}". Cannot compute equivalent rate.')

    return combinations


def determine_format(pairs: List[Tuple[float, float]]) -> Tuple[int, int]:
    """
    Determine the maximum number of integer and fractional digits from a list of (shear, veer) pairs.

    Args:
        pairs (list[tuple[float, float]]): List of (shear, veer) value pairs.

    Returns:
        tuple: (int_digits, frac_digits)
    """

    all_values = [val for pair in pairs for val in pair]  # Flatten the list of tuples
    
    # Determine max fractional digits
    max_frac_digits = max(len(str(val).split('.')[-1]) if '.' in str(val) else 0 for val in all_values)

    # Check if all integer parts are zero
    if all(int(abs(val)) == 0 for val in all_values):
        max_int_digits = 0
    else:
        max_int_digits = max(len(str(int(abs(val)))) for val in all_values)
    
    return max_int_digits, max_frac_digits


def return_case_strings(pairs: List[Tuple[float, float]], format_specs: Tuple[int, int]) -> List[str]:
    """
    Return a formatted list of strings with the casenames to iterate over

    Args:
        pairs (list[float]): List of shear/veer pairs.
        int_digits (int): number of digits for precision on left of decimal.
        frac_digits (int): number of digits for precision on right of decimal.

    Returns:
        list: ['s0_v0','s0_v1',...]
    """
    # If pairs is a single pair (not a list), convert it into a list
    if isinstance(pairs, tuple):
        pairs = [pairs]
    
    int_digits, frac_digits = format_specs
    case_strings = []
    
    for pair in pairs:
        # Ensure pair is treated as a tuple (shear, veer)
        shear_str = format_value(pair[0], int_digits, frac_digits)
        veer_str = format_value(pair[1], int_digits, frac_digits)
        case_strings.append(f"s{shear_str}_v{veer_str}")
    
    return case_strings


def run_subprocess(command: List[str], mac_flag = False) -> None:
    """Run subprocess commands with OS compatibility."""
    if platform.system() == "Darwin" and mac_flag:
        command.insert(2, "")
    subprocess.run(command, check = True)


def create_symlinks(wrf_path: str, target_dir: str) -> None:
    """Create symbolic links in the target directory."""
    links = [
        f"{wrf_path}/main/wrf.exe",
        f"{wrf_path}/main/ideal.exe",
        f"{wrf_path}/run/README.namelist",
        f"{wrf_path}/run/README.physics_files",
        f"{wrf_path}/run/README.tslist",
    ]
    for link in links:
        subprocess.run([f"ln -s {link} {target_dir}"], shell = True)


def copy_files(source: str, destination: str, dirs_exist_ok = True) -> None:
    """Copy files and directories from source to destination."""
    if os.path.isdir(source):
        shutil.copytree(source, destination, dirs_exist_ok = dirs_exist_ok)
    else:
        shutil.copy2(source, destination)


def generate_profile(x: np.ndarray, D: float, shear: float) -> np.ndarray:
    """
    Generate lateral (v) velocity component given a shear RATE

    Args:
        x (ArrayLike): An array of vertical (z) positions
        D (float): Turbine diameter
        shear (float): a shear rate

    Returns:
        ArrayLike: Array of lateral velocities at each x location
    """
    # Define the regions
    x1, x2 = -(D/2) * 1.75, (D/2) * 1.75 # Region boundaries

    # Calculate the constant values based on Region 2
    value_region1 = shear * x1  # f(x_1) for Region 1
    value_region3 = shear * x2  # f(x_2) for Region 3

    # Define the piecewise behavior
    return np.array(np.piecewise(
        x,
        [x < x1, (x >= x1) & (x <= x2), x > x2],
        [value_region1, lambda x: shear * x, value_region3]
    ), dtype=float)

def smooth_piecewise(y: np.ndarray, sigma: int, dy: float) -> np.ndarray:
    """
    Smooth velocity components using a Gaussian convolution approach.

    Args:
        y (ArrayLike): An array of data to smooth
        sigma (int): Kernel width
        dx (float): Uniform spacing of data in y
    Returns:
        ArrayLike: Array of smoothed y data
    """
    # Create a Gaussian kernel with a given standard deviation (sigma)
    kernel_size = int(6 * sigma / dy)  # Ensure the kernel is large enough to cover the influence of the Gaussian
    kernel = np.exp(-np.linspace(-3, 3, kernel_size)**2 / (2 * sigma**2))
    kernel /= np.sum(kernel)  # Normalize the kernel

    # Apply convolution (low-pass filter) to smooth the signal
    y_smoothed = signal.convolve(y, kernel, mode='same')

    # Skip the endpoints by copying them directly from the original signal
    skip_points = kernel_size // 2  # Number of points influenced by the kernel
    y_smoothed[:skip_points] = y[:skip_points]  # Preserve the start
    y_smoothed[-skip_points:] = y[-skip_points:]  # Preserve the end
    
    return y_smoothed


def create_sounding(current_path: str, figure_path: str, figure_name: str, params: Dict[str, Any], namelist: Dict[str, Any], turbine: Dict[str, Any], pair: Tuple[float, float]) -> None:
    """
    Create sounding files based on shear and veer settings.

    Args:
        current_path (str): Path to location to create namelist file
        figure_path (str): Path to location to save profile plots
        figure_name (str): Formatted figure name
        params (Dict): A dictionary of settings
        namelist (Dict): A dictionary of simulation parameters
        turbine (Dict): A dictionary of turbine parameters
        pair (Dict): The shear,veer pair
    """

    z = np.arange(-1000, 1000, 1,dtype=float)

    if params['shear_type'] in ['Rate', 'Total']:

        uinf = generate_profile(z, turbine.turb_diameter, pair[0]/params['Ufst']) + 1.0
        uinf = smooth_piecewise(uinf, 30, z[1]-z[0])

        wdir = generate_profile(z, turbine.turb_diameter, pair[1])
        wdir = -smooth_piecewise(wdir, 30, z[1]-z[0])

    else:

        print(f'ERROR. Unsupported shear type: {params["shear_type"]}.')
        return False

    cor_func = interp1d(wdir, z, kind='linear')
    z_corr   = cor_func(0)
    dir_func = interp1d(z-z_corr,wdir,kind='linear', fill_value='extrapolate')
    wdir     = dir_func(z)
    
    vinf = np.full_like(z,1) * np.sin(np.deg2rad(wdir))

    u = params['Ufst']*uinf
    v = params['Ufst']*vinf

    u[u == -0.00] = 0.0
    v[v == -0.00] = 0.0

    wdir = wdir + 270

    z = z + turbine.hubheight

    ######################################################################
    # Plot profiles

    if(params['plot_profiles']):
        plot_sounding(figure_path, figure_name, namelist, pair, params, turbine, u, v, wdir, z)

    ######################################################################
    # write to sounding file
    theta = 288.0 # temperature in K
    w     = 0.0 # water vapor mixing ratio in g kg-1

    mask = (z >= 0) & (z <= namelist.ztop * 1.1)

    z_clip = z[mask]
    u_clip = u[mask]
    v_clip = v[mask]

    theta = theta * np.ones(len(z_clip))
    w     = w * np.ones(len(z_clip))

    fmt = '%1.2f %1.2f %1.2f %1.2f %1.2f'

    prependline = r"1000.00 288.00 00.00"

    filepath = current_path + '/' + 'input_sounding'
    
    # Open the file in write mode
    with open(filepath, 'w') as f:
        # Write the prepended line first
        f.write(prependline + '\n')
        
        # Append the NumPy array data
        np.savetxt(f, np.stack([z_clip, theta, w, u_clip, v_clip], axis=1), fmt=fmt)


def plot_sounding(figure_path: str, figure_name: str, namelist: Dict[str, Any], pair: Tuple[float, float], params: Dict[str, Any], turbine: Dict[str, Any], u: np.ndarray, v: np.ndarray, wdir: np.ndarray, z: np.ndarray) -> None:
    """
    Create plot of velocity profiles.

    Args:
        figure_path (str): Path to location to save profile plots
        figure_name (str): Formatted figure name
        namelist (Dict): A dictionary of simulation parameters
        pair (Dict): The shear,veer pair
        params (Dict): A dictionary of settings
        turbine (Dict): A dictionary of turbine parameters

    """
    # Create a figure with a custom gridspec layout

    latex_path = shutil.which("latex")

    if latex_path:
        latex_dir = os.path.dirname(latex_path)
        os.environ["PATH"] += os.pathsep + latex_dir
    else:
        raise RuntimeError("LaTeX not found. Ensure it is installed and in the system path.")

    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })
    
    _, axs = plt.subplots(
        nrows = 1,
        ncols = 3,
        figsize = (9, 4),
        sharey=True,
        constrained_layout = True,
    )
    
    # velocity profiles
    axs[0].axhline(turbine.hubheight-(0.5*turbine.turb_diameter), linestyle='dashed', linewidth=1, dashes=(8, 3))
    axs[0].axhline(turbine.hubheight, linestyle='dotted', linewidth=1)
    axs[0].axhline(turbine.hubheight+(0.5*turbine.turb_diameter), linestyle='dashed', linewidth=1, dashes=(8, 3))
    axs[0].axvline(0.0, linestyle='dotted', linewidth=1)
    axs[0].axvline(params['Ufst'], linestyle='dotted', linewidth=1)
    axs[0].plot(u, z, color='#0000FF', linestyle='solid', label=r'$u_{\infty}$')
    axs[0].plot(v, z, color='#E50000', linestyle='solid', label=r'$v_{\infty}$')
    axs[0].set_xlim([-params['Ufst'], params['Ufst']*2])
    axs[0].set_ylim([0, namelist.ztop])
    axs[0].legend()
    axs[0].set_xticks(np.arange(-8, params['Ufst']+10, 4))
    axs[0].set_xlabel(r'$u_{i}~[\textrm{m~s$^{-1}$}]$')
    axs[0].set_ylabel(r'$z~[\textrm{m}]$')

    test_z    = np.array([(-0.5*turbine.turb_diameter),(0.5*turbine.turb_diameter)])
    test_line = pair[0] * test_z

    axs[0].set_title(rf"$\Delta u_{{\infty}}: {abs(test_line[0]) + abs(test_line[1]):.2f}$ [m/s]")
    axs[0].plot(test_line + params['Ufst'], test_z + turbine.hubheight, color='orange', linestyle='solid', label=r'_nolegend_')
    
    tip_u = np.interp(turbine.hubheight-0.5*turbine.turb_diameter,z,u)
    diff = (test_line[0]+ params['Ufst'] - tip_u)

    x_range = (params['Ufst']*2) - -params['Ufst']
    axs[0].text(x_range*0.7 + -params['Ufst'], 75, f"Tip error: {abs(diff):.3f} [m/s]", fontsize=6, color="k", ha='left')


    # wind direction
    axs[1].axhline(turbine.hubheight-(0.5*turbine.turb_diameter), linestyle='dashed', linewidth=1, dashes=(8, 3))
    axs[1].axhline(turbine.hubheight, linestyle='dotted', linewidth=1)
    axs[1].axhline(turbine.hubheight+(0.5*turbine.turb_diameter), linestyle='dashed', linewidth=1, dashes=(8, 3))
    axs[1].axvline(270.0, linestyle='dotted', linewidth=1)
    axs[1].plot(wdir, z, color='#006400', linestyle='solid', label=r'_nolegend_')

    test_z    = np.array([(-0.5*turbine.turb_diameter),(0.5*turbine.turb_diameter)])
    test_line = -pair[1] * test_z
    axs[1].set_title(rf"$\Delta\beta: {abs(test_line[0]) + abs(test_line[1]):.2f}$ [deg]")

    axs[1].plot(test_line + 270, test_z + turbine.hubheight, color='orange', linestyle='solid', label=r'_nolegend_')

    tip_deg = np.interp(turbine.hubheight-0.5*turbine.turb_diameter,z,wdir)
    diff = (((test_line[0] + 270) - tip_deg) + 180) % 360 - 180
    axs[1].text(300, 75, f"Tip error: {abs(diff):.2f} [deg]", fontsize=6, color="k", ha='left')

    if wdir[-1] <= 180:
        axs[1].text(275, 660, f"RESVERSE FLOW", fontsize=6, color='r', ha='left', fontweight='bold')

    axs[1].set_xlim([170.0, 370.0])
    axs[1].set_ylim([0, namelist.ztop])
    axs[1].set_xlabel(r'$\beta~[\textrm{$^{\circ}$}]$')

    # wind speed magnitude
    axs[2].axhline(turbine.hubheight-(0.5*turbine.turb_diameter), linestyle='dashed', linewidth=1, dashes=(8, 3))
    axs[2].axhline(turbine.hubheight, linestyle='dotted', linewidth=1)
    axs[2].axhline(turbine.hubheight+(0.5*turbine.turb_diameter), linestyle='dashed', linewidth=1, dashes=(8, 3))
    axs[2].axvline(params['Ufst'], linestyle='dotted', linewidth=1)
    axs[2].plot((u**2 + v**2)**(0.5), z, color='#730080', linestyle='solid')
    axs[2].set_xlim([0, 15])
    axs[2].set_ylim([0, namelist.ztop])
    axs[2].set_xlabel(r'Wind speed $[\textrm{m~s$^{-1}$}]$')

    # print(figure_path)
    print(figure_name)

    plt.savefig(figure_path + '/' + figure_name + '.png', bbox_inches="tight", dpi=600)

    plt.show()


def setup(params: Dict[str, Any], model: str) -> bool:

    namelist, turbtuple = preproc.validate(opt_params=params)

    combinations = get_combinations(params=params, D=turbtuple.turb_diameter)

    formats = determine_format(combinations)

    model_str = params['rotor_model'].lower() + f'_sweep'

    # Initialize the success flag
    success = True

    # Check if the top directory already exists
    top_dir = f"{params['base_dir']}/{model_str}"
    if os.path.exists(top_dir):
        print(f"Directory {top_dir} already exists.")
        return False 
    
    # Make top directory
    os.makedirs(f"{params['base_dir']}/{model_str}/figs", exist_ok=False)

    # Create batch submit file if requested
    batch_file_path = f"{params['base_dir']}/submit_group_{model_str}.sh"

    if params['batch_submit']:
        with open(batch_file_path, 'w') as batch_file:
            batch_file.write("#!/bin/bash\n\n")

    ntasks = namelist.nproc_x * namelist.nproc_y

    counter = 1

    # Only create pairs requested
    for pair in combinations:
        if pair in params['excluded_pairs']:
            continue

        # Format shear and veer values
        dir_name = return_case_strings(pair,formats)[0]
        current_path = f"{params['base_dir']}/{model_str}/{dir_name}"

        # Create shear + veer directory
        os.makedirs(current_path, exist_ok=False)

        # Create sounding file
        create_sounding(current_path, f"{params['base_dir']}/{model_str}/figs", dir_name, params, namelist, turbtuple, pair)

        # Create symbolic links for the executables
        create_symlinks(params['wrf_path'], current_path)

        # Copy items in case template directory
        case_path = f"{params['template_path']}/case"
        for item in os.listdir(case_path):
            source_path = os.path.join(case_path, item)
            if os.path.isfile(source_path):  # Check if the item is a file
                copy_files(source_path, os.path.join(current_path, item))

        # Copy requested turbine directory
        turb_source_dir = f"{params['template_path']}/case/windTurbines/{params['turb_model']}"
        destination_dir = os.path.join(current_path, 'windTurbines', params['turb_model'])

        if os.path.exists(turb_source_dir) and os.path.isdir(turb_source_dir):
            os.makedirs(os.path.join(current_path, 'windTurbines'), exist_ok=False)
            shutil.copytree(turb_source_dir, destination_dir)
        else:
            print(f'ERROR. Source directory {turb_source_dir} does not exist.')
            return False
        
        # Copy wind turbine database
        shutil.copy2(f"{params['template_path']}/case/windTurbines/windTurbineTypes.dat", os.path.join(current_path, 'windTurbines'))

        # Copy additional files
        file_map = {
            f"{params['template_path']}/namelists/{params['turb_model']}/{model.lower()}_namelist.input": 'namelist.input',
            f"{params['template_path']}/turbines/{params['turb_model']}/{model.lower()}_windturbines-ij.dat": 'windturbines-ij.dat',
            f"{params['template_path']}/shell/export_libs_load_modules_{params['system']}.sh": 'export_libs_load_modules.sh',
            f"{params['template_path']}/shell/submit_template_{params['system']}.sh": 'submit.sh',
        }
        for src, dst in file_map.items():
            copy_files(src, os.path.join(current_path, dst))

        # Update file placeholders with requested values
        turb_jobname = params['turb_model'].replace("iea", "").replace("MW", "").strip()
        replacements = {
            "lib_path": params['wrf_path'].replace("/", "\\/"),
            "{PH_NNODES}": params['num_nodes'],
            "{PH_JOB_NAME}": f"{turb_jobname}_{dir_name}",
            "{PH_ALLOCATION}": f"{params['allocation']}",
            "{PH_NTASKS}": ntasks,
            "{PH_TIME}": f"{params['runtime']}",
            "{PH_PARTITION}": f"{params['partition']}",
        }
        # for key, val in replacements.items():
        #     run_subprocess(['sed', '-i', f"s/{key}/{val}/g", os.path.join(current_path, 'export_libs_load_modules.sh')])
        #     run_subprocess(['sed', '-i', f"s/{key}/{val}/g", os.path.join(current_path, 'submit.sh')])

        if params['batch_submit']:
            with open(batch_file_path, 'a') as batch_file:
                batch_file.write(f"cd {current_path}\nsbatch submit.sh\n\n")

        counter = counter + 1
    
    if params['batch_submit']:
        os.chmod(batch_file_path, 0o777)

    print('Done.')
    return success