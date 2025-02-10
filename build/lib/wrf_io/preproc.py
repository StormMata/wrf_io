import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from io import StringIO
from rich.table import Table
from scipy import interpolate
from rich.console import Console
from collections import namedtuple
from typing import Dict, Any, Tuple
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams['figure.dpi'] = 500

# Define namedtuples for passing parsed information between functions
Namelist = namedtuple('Namelist', [
    'run_days', 'run_hrs', 'run_mins', 'run_secs', 'time_step',
    'outer_dx', 'outer_dy', 'inner_dx', 'inner_dy', 'outer_e_we', 'outer_e_sn',
    'ztop', 'inner_e_we', 'inner_e_sn', 'e_vert', 'i_parent_start',
    'j_parent_start', 'parent_grd_rat', 'time_step_rat', 'nproc_x', 'nproc_y',
    'nSections', 'nElements'
])

Turbine = namedtuple('Turbine', [
    'hubheight', 'turb_diameter','hub_diameter', 'inflow_loc', 'rot_dir', 'irrot_ct', 'turb_x', 'turb_y'
])


def parse_namelist(opt_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse settings in the namelist.input file.

    Args:
        opt_params (Dict): A dictionary of settings with root directory path

    Returns:
        config: A dictionary of parsed values
    """

    # If the namelist file path is specified directly, use that, otherwise use base path
    if 'name_path' in opt_params:
        file_path = opt_params['name_path']
    else:
        file_path = opt_params['read_from'] + '/namelists/' + opt_params['rotor_model'].lower() +'_namelist.input'

    config = {}
    current_section = None

    # Open the file and read its contents
    with open(file_path, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespaces and skip empty lines
            line = line.strip()
            if not line:
                continue

            # Detect section headers (e.g., &time_control)
            section_match = re.match(r'^\s*&(\w+)\s*$', line)
            if section_match:
                current_section = section_match.group(1)
                config[current_section] = {}
                continue

            # Skip lines that do not have '=' (likely comments or empty lines)
            if '=' not in line:
                continue

            # Extract key-value pairs from lines
            key, value = [part.strip() for part in line.split('=', 1)]
            value = value.rstrip(',')  # Remove trailing comma
            value = value.strip()  # Strip leading/trailing spaces

            # Handle comma-separated values (if any)
            if ',' in value:
                value = [v.strip() for v in value.split(',')]
            elif value.lower() in ['.true.', '.false.']:  # Handle Boolean values
                value = value.lower() == '.true.'
            else:  # Otherwise treat it as a single value (string or number)
                try:
                    value = float(value) if '.' in value else int(value)
                except ValueError:
                    pass  # Keep it as string if it's not a number

            # Store the parsed value in the section
            if current_section:
                config[current_section][key] = value

    return config

def parse_turbine_properties(opt_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse settings in the turbineProperties.tbl file.

    Args:
        opt_params (Dict): A dictionary of settings with root directory path

    Returns:
        config: A dictionary of parsed values
    """
    file_path = opt_params['read_from'] + '/case/windTurbines/' + opt_params['turb_model'] + '/turbineProperties.tbl'

    config = {}

    # Open the file and read its contents
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check for key-value pairs where value comes first and key is in quotes
            key_value_match = re.match(r'^\s*([\d\.\-]+)\s+"([^"]+)"\s*$', line)
            if key_value_match:
                value = float(key_value_match.group(1))  # Convert value to float
                key = key_value_match.group(2)  # Extract key as string
                config[key] = value  # Store in dictionary with key-value pair

    return config

def parse_turbine_location(opt_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse settings in the windturbines-ij.dat file.

    Args:
        opt_params (Dict): A dictionary of settings with root directory path

    Returns:
        config: A dictionary of parsed values
    """
    file_path = opt_params['read_from'] + '/turbines/' + opt_params['rotor_model'].lower() + '_windturbines-ij.dat'

    config = []

    # Open the file and read its contents
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Split the line into individual values based on space
            values = line.split()
            
            # Convert the values into appropriate data types (e.g., float, int)
            config = [float(value) if '.' in value else int(value) for value in values]

    return config

def load_variables(parsed_config: Dict[str, Any], parsed_turbine: Dict[str, Any], parsed_location: Tuple[float, float]) -> Tuple[Namelist, Turbine]:
    """
    Load variables from the parsed configuration and return as two namedtuples.
    
    Args:
        parsed_config (Dict[str, Any]): Parsed configuration dictionary.
        parsed_turbine (Dict[str, Any]): Parsed turbine dictionary.
        parsed_location (Tuple[float, float]): Turbine location (x, y).
    
    Returns:
        Tuple[Namelist, Turbine]: Two namedtuples containing formatted variables from parsed dictionaries
    """
    
    # Extract variables for the namelist
    namelist = Namelist(
        run_days       = int(parsed_config['time_control'].get('run_days', None)),
        run_hrs        = int(parsed_config['time_control'].get('run_hours', None)),
        run_mins       = int(parsed_config['time_control'].get('run_minutes', None)),
        run_secs       = int(parsed_config['time_control'].get('run_seconds', None)),

        time_step      = float(parsed_config['domains'].get('time_step_fract_num', None))/float(parsed_config['domains'].get('time_step_fract_den', None)),

        outer_dx       = float(parsed_config['domains'].get('dx', None)),
        outer_dy       = float(parsed_config['domains'].get('dy', None)),

        inner_dx       = float(parsed_config['domains'].get('dx', None)) / float(parsed_config['domains'].get('parent_grid_ratio', None)[1]),
        inner_dy       = float(parsed_config['domains'].get('dy', None)) / float(parsed_config['domains'].get('parent_grid_ratio', None)[1]),
            
        outer_e_we     = int(parsed_config['domains'].get('e_we', None)[0]),
        outer_e_sn     = int(parsed_config['domains'].get('e_sn', None)[0]),
        ztop           = int(parsed_config['domains'].get('ztop', None)[0]),
            
        inner_e_we     = int(parsed_config['domains'].get('e_we', None)[1]),
        inner_e_sn     = int(parsed_config['domains'].get('e_sn', None)[1]),

        e_vert         = int(parsed_config['domains'].get('e_vert', None)[0]),

        i_parent_start = int(parsed_config['domains'].get('i_parent_start', None)[1]),
        j_parent_start = int(parsed_config['domains'].get('j_parent_start', None)[1]),

        parent_grd_rat = int(parsed_config['domains'].get('parent_grid_ratio', None)[1]),
        time_step_rat  = int(parsed_config['domains'].get('parent_time_step_ratio', None)[1]),

        nproc_x        = parsed_config['domains'].get('nproc_x', None),
        nproc_y        = parsed_config['domains'].get('nproc_y', None),

        nSections      = int(parsed_config['physics'].get('wind_wtp_nSections', None)),
        nElements      = int(parsed_config['physics'].get('wind_wtp_nElements', None))
    )

    # Extract variables for the turbine
    turbine = Turbine(
        hubheight      = float(parsed_turbine['Hub height [m]']),

        turb_diameter  = float(parsed_turbine['Rotor diameter [m]']),
        hub_diameter   = float(parsed_turbine['Hub diameter [m]']),

        inflow_loc     = float(parsed_turbine['Inflow location [m]']),

        rot_dir        = int(parsed_turbine['Rotation direction [-], 1: clockwise, -1: counter-clockwise, 0: no rotational effects (GAD/GADrs; only for experimental use)']),
        irrot_ct       = float(parsed_turbine['CT, Constant thrust coefficient for GAD/GADrs. It should be applied when rotational effects are neglected. E.g., CT=0.77 for Uinf=8.0 m s-1. Only for experimental use.']),

        turb_x         = float(parsed_location[0]),
        turb_y         = float(parsed_location[1])
    )

    return namelist, turbine

def distribute_processors(num_processors: int) -> Tuple[int, int]:
    """
    Determine distribution of processors over inner domain in y and x.
    
    Args:
        num_processors (int): ntasks specificed in shell script
    
    Returns:
        Tuple[Namelist, Turbine]: A tuple containing the processors in y and x.
    """

    # Find the closest square grid dimensions that use all processors
    rows = int(math.sqrt(num_processors))
    cols = num_processors // rows
    
    # Adjust to ensure all processors are used
    if rows * cols < num_processors:
        cols += 1  # Increase columns if the product is less than the number of processors
    
    # Calculate how many processors will be in each block
    return rows, cols

def plot_outer_domain(namelist: Namelist, turbine: Turbine, opt_params: Dict[str, Any]) -> plt.Figure:
    """
    Plots the outer domain.

    Args:
        namelist (Namelist): A named tuple containing domain and simulation parameters.
        turbine (Turbine): A named tuple containing turbine-specific parameters.
        opt_params (Dict): A dictionary of settings with bool for printing plot to terminal

    Returns:
        plt.Figure: The created figure object for saving or displaying.
    """

    outer_length   = namelist.outer_e_we * namelist.outer_dx      # Length of the outer domain (x-direction)
    outer_width    = namelist.outer_e_sn * namelist.outer_dx      # Width of the outer domain (y-direction)
    outer_height   = namelist.ztop                                # Height of the outer domain (z-direction)

    inner_length   = namelist.inner_e_we * namelist.inner_dx      # Length of the inner domain (x-direction)
    inner_width    = namelist.inner_e_sn * namelist.inner_dx      # Width of the inner domain (y-direction)
    inner_height   = namelist.ztop                                # Height of the inner domain (z-direction)
    inner_i        = namelist.i_parent_start * namelist.outer_dx  # Inner domain x-offset from the origin of the outer domain
    inner_j        = namelist.j_parent_start * namelist.outer_dx  # Inner domain y-offset from the origin of the outer domain

    turbine_radius = turbine.turb_diameter / 2                    # Radius of the wind turbine circle

    # Define 3D figure
    fig = plt.figure(figsize=(10, 10))  # Width = 10 inches, Height = 8 inches
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    # Draw outer domain as a cuboid
    outer_corners = [
        [0, 0, 0], [outer_length, 0, 0], [outer_length, outer_width, 0], [0, outer_width, 0],
        [0, 0, outer_height], [outer_length, 0, outer_height], [outer_length, outer_width, outer_height], [0, outer_width, outer_height]
    ]

    faces_outer = [[outer_corners[j] for j in [0, 1, 5, 4]],
                   [outer_corners[j] for j in [3, 0, 4, 7]],
                   [outer_corners[j] for j in [4, 5, 6, 7]]]
    ax.add_collection3d(Poly3DCollection(faces_outer, color='skyblue', alpha=0.2, linewidths=1, edgecolors='b', zorder=7))

    faces_outer = [[outer_corners[j] for j in [1, 2, 6, 5]],
                   [outer_corners[j] for j in [2, 3, 7, 6]],
                   [outer_corners[j] for j in [0, 1, 2, 3]]]
    ax.add_collection3d(Poly3DCollection(faces_outer, color='skyblue', alpha=0.2, linewidths=1, linestyle=':', edgecolors='b', zorder=7))

    # Calculate inner domain position
    inner_x_start = inner_i
    inner_y_start = inner_j
    inner_z_start = 0  # Assume inner domain is at the base of the outer domain

    # Draw inner domain as a smaller cuboid within the outer domain
    inner_corners = [
        [inner_x_start, inner_y_start, inner_z_start],
        [inner_x_start + inner_length, inner_y_start, inner_z_start],
        [inner_x_start + inner_length, inner_y_start + inner_width, inner_z_start],
        [inner_x_start, inner_y_start + inner_width, inner_z_start],
        [inner_x_start, inner_y_start, inner_z_start + inner_height],
        [inner_x_start + inner_length, inner_y_start, inner_z_start + inner_height],
        [inner_x_start + inner_length, inner_y_start + inner_width, inner_z_start + inner_height],
        [inner_x_start, inner_y_start + inner_width, inner_z_start + inner_height]
    ]
    faces_inner = [[inner_corners[j] for j in [0, 1, 5, 4]],
                   [inner_corners[j] for j in [3, 0, 4, 7]],
                   [inner_corners[j] for j in [4, 5, 6, 7]]]
    ax.add_collection3d(Poly3DCollection(faces_inner, color='orange', alpha=0.25, linewidths=1, edgecolors='r',zorder=6))

    faces_inner = [[inner_corners[j] for j in [1, 2, 6, 5]],
                [inner_corners[j] for j in [2, 3, 7, 6]],
                [inner_corners[j] for j in [0, 1, 2, 3]]]
    ax.add_collection3d(Poly3DCollection(faces_inner, color='orange', alpha=0.25, linestyle=':', linewidths=1, edgecolors='r',zorder=1))

    # Draw turbine circle within the inner domain
    turbine_center_x = inner_x_start + turbine.turb_x
    turbine_center_y = inner_y_start + turbine.turb_y
    turbine_center_z = turbine.hubheight

    # Generate points for a filled disk
    theta = np.linspace(0, 2 * np.pi, 100)      # Angle for full circle
    radii = np.linspace(0, turbine_radius, 50)  # Radii from center to edge of disk

    # Generate grid of points for the disk
    x_circle = np.full((len(radii), len(theta)), turbine_center_x)  # Constant X-coordinate for the disk
    y_circle = turbine_center_y + np.outer(radii, np.cos(theta))
    z_circle = turbine_center_z + np.outer(radii, np.sin(theta))

    # Plotting
    ax.plot_surface(x_circle, y_circle, z_circle, color='red', alpha=1.0, zorder=5)

    # Plot vertical line from the disk down to z = 0
    ax.plot([turbine_center_x, turbine_center_x], [turbine_center_y, turbine_center_y], [turbine_center_z - turbine.turb_diameter/2, 0], color='black', linewidth=1, zorder=2)
    ax.plot([turbine_center_x, inner_i], [turbine_center_y, turbine_center_y], [0, 0], color='black', linewidth=1, zorder=3)
    ax.plot([turbine_center_x, turbine_center_x], [turbine_center_y, inner_j], [0, 0], color='black', linewidth=1, zorder=4)

    # Set x-ticks at increments of turbine diameter
    x_ticks  = np.arange(0, outer_length + turbine.turb_diameter, turbine.turb_diameter * 5)
    x_labels = [f"{int(turb / turbine.turb_diameter)}" for turb in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=8.5)

    # Set y-ticks at increments of turbine diameter
    y_ticks  = np.arange(0, outer_length + turbine.turb_diameter, turbine.turb_diameter * 5)
    y_labels = [f"{int(turb / turbine.turb_diameter)}" for turb in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=10)

    z_ticks  = [turbine.turb_diameter, turbine.turb_diameter * 3]
    z_labels = ['1','3']  # Format with one decimal point

    ax.set_zticks(z_ticks)
    ax.set_zticklabels(z_labels, fontsize=10)

    # Set plot limits to show only the outer domain
    ax.set_xlim([0, outer_length])
    ax.set_ylim([0, outer_width])
    ax.set_zlim([0, outer_height])

    # Set labels
    ax.set_xlabel(r'$x/D$ [-]', labelpad=25, fontsize=9)
    ax.set_ylabel(r'$y/D$ [-]', labelpad=15, fontsize=9)
    ax.set_zlabel(r'$z/D$ [-]', fontsize=9)

    # Set camera angle
    if 'outer_align' in opt_params and opt_params['outer_align']:
        ax.view_init(elev=90, azim=-90)
        ax.plot([0, outer_length], [outer_width/2, outer_width/2], [0, 0], color='red', linewidth=1, zorder=2)
    else:
        ax.view_init(elev=30, azim=-150)

    # ax.view_init(elev=9, azim=-170)
    # ax.view_init(elev=0, azim=90)
    # ax.view_init(elev=90, azim=-90)

    plt.gca().set_aspect('equal', adjustable='box')

    # Set the alpha for grid lines on each axis in a 3D plot
    ax.xaxis._axinfo["grid"].update({"color": (0.5, 0.5, 0.5, 0.10)})  # 10% opacity for x-axis grid
    ax.yaxis._axinfo["grid"].update({"color": (0.5, 0.5, 0.5, 0.10)})  # 10% opacity for y-axis grid
    ax.zaxis._axinfo["grid"].update({"color": (0.5, 0.5, 0.5, 0.10)})  # 10% opacity for z-axis grid

    # Set transparent background for axis panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    if 'plot_outer' in opt_params and opt_params['plot_outer'] == False:
        plt.close(fig) 

    return fig

def plot_inner_domain(namelist: Namelist, turbine: Turbine, opt_params: Dict[str, Any]) -> plt.Figure:
    """
    Plots the inner domain indicating the turbine location and inflow point within
    the processor patch, along with the processor breakdown in y and x, and any 
    sampling locations downstream of the turbine.

    Args:
        namelist (Namelist): A named tuple containing domain and simulation parameters.
        turbine (Turbine): A named tuple containing turbine-specific parameters.
        opt_params (Dict): A dictionary of settings with bool for printing plot to terminal

    Returns:
        plt.Figure: The created figure object for saving or displaying.
    """
    
    # Calculate rectangle dimensions
    x = namelist.inner_e_we * namelist.inner_dx  # Width of the rectangle
    y = namelist.inner_e_sn * namelist.inner_dx  # Height of the rectangle

    if (namelist.nproc_x == None) & (namelist.nproc_y == None):
        rows, cols = distribute_processors(opt_params['ntasks'])
    else:
        rows = int(namelist.nproc_y)
        cols = int(namelist.nproc_x)

    # Calculate step sizes (rounded down to integers)
    dx = int(x // cols)  # Integer width of each internal column
    dy = int(y // rows)  # Integer height of each internal row

    # Calculate the leftover space
    leftover_x = x - (dx * cols)
    leftover_y = y - (dy * rows)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))

    # Add the outer rectangle to the plot
    rect = Rectangle((0, 0), x, y, edgecolor='blue', facecolor='none', linewidth=2)
    ax.add_patch(rect)

    # Draw the vertical dividing lines
    x_pos_array = []
    for i in range(1, cols):
        x_pos = i * dx
        if i == cols - 1:  # Extend the final column to include leftover_x
            x_pos += leftover_x
        x_pos_array.append(x_pos)
        ax.plot([x_pos, x_pos], [0, y], color='black', linestyle='--', linewidth=1, alpha=0.25)

    # Draw the horizontal dividing lines
    y_pos_array = []
    for j in range(1, rows):
        y_pos = j * dy
        if j == rows - 1:  # Extend the final row to include leftover_y
            y_pos += leftover_y
        y_pos_array.append(y_pos)
        ax.plot([0, x], [y_pos, y_pos], color='black', linestyle='--', linewidth=1, alpha=0.25)

    # Plot a rectangle on the processor patch
    x_pos_array = np.array(x_pos_array)
    y_pos_array = np.array(y_pos_array)

    low_x = np.searchsorted(x_pos_array, turbine.turb_x, side='left') - 1
    low_y = np.searchsorted(y_pos_array, turbine.turb_y, side='right') - 2

    # print(low_x)

    low_x = low_x if low_x > 0 else 0
    low_y = low_y if low_y > 0 else 0

    # print(low_x)

    x_pos_array = x_pos_array - x_pos_array[0]

    # print(low_x)

    # print(x_pos_array[low_x])

    # Create the rectangle patch
    rectangle = Rectangle(
        (x_pos_array[low_x], y_pos_array[low_y]),            # (x, y) starting position of the rectangle
        x_pos_array[low_x+3] - x_pos_array[low_x],            # Width of the rectangle
        y_pos_array[low_y+3] - y_pos_array[low_y],            # Height of the rectangle
        linewidth=2,               # Border width
        edgecolor='none',          # Border color
        facecolor='grey',          # Fill color
        alpha=0.12                 # Transparency of the rectangle
    )

    # Add the rectangle to the plot
    ax.add_patch(rectangle)

    # Plot location of turbine
    ax.scatter(turbine.turb_x, turbine.turb_y, color='none', edgecolors='red', s=50, zorder=2)
    ax.scatter(turbine.turb_x, turbine.turb_y, color='red', s=125, marker='+', zorder=2)

    # Plot location of inflow sample point
    if turbine.turb_x < turbine.inflow_loc:
        ax.scatter(0, turbine.turb_y, color='red', s=50, marker='<', label="Single Point", zorder=2)
    else:
        interp_array_x  = np.array([0, 4962])
        interp_array_dx = np.array([0, 35])
        interp_f = interpolate.interp1d(interp_array_x, interp_array_dx, fill_value='extrapolate')
        ax.scatter(turbine.turb_x - turbine.inflow_loc +  interp_f(x), turbine.turb_y, color='red', s=50, marker='<', label="Single Point", zorder=2)
        ax.vlines(turbine.turb_x - turbine.inflow_loc, turbine.turb_y - turbine.turb_diameter/2, turbine.turb_y + turbine.turb_diameter/2, colors='red', linestyles='-')

    # Plot downstream sample points is specified
    if 'slice_loc' in opt_params:
        slices = turbine.turb_x + np.array(np.arange(1,opt_params['slice_loc'] + 1,1)) * turbine.turb_diameter

        ax.vlines(slices, turbine.turb_y - turbine.turb_diameter*0.10, turbine.turb_y + turbine.turb_diameter*0.10, colors='green', linestyles='-')

    # Set labels and title
    ax.set_xlabel("West-East [m]")
    ax.set_ylabel("South-North [m]")
    ax.set_title("Inner domain")

    ax.set_aspect('equal', adjustable='box')

    if 'plot_inner' in opt_params and opt_params['plot_inner'] == False:
        plt.close(fig) 

    return fig

def summary_table(namelist: Namelist, turbine: Turbine, opt_params: Dict[str, Any]) -> None:
    """
    Generates a summary table of relevant parameters and indicates and obvious issues.

    Args:
        namelist (Namelist): A named tuple containing domain and simulation parameters.
        turbine (Turbine): A named tuple containing turbine-specific parameters.
        opt_params (Dict): A dictionary of settings
    """

    outer_grid = namelist.outer_e_sn * namelist.outer_e_we * namelist.ztop
    inner_grid = namelist.inner_e_sn * namelist.inner_e_we * namelist.ztop

    inner_i    = namelist.i_parent_start * namelist.outer_dx   # Inner domain x-offset from the origin of the outer domain

    turbine_center_x = namelist.i_parent_start * namelist.outer_dx + turbine.turb_x

    # Define the console for rich output
    console = Console(record=True)

    # Create a rich table
    table = Table()

    # Add columns
    table.add_column("Parameter", justify="left", no_wrap=True)
    table.add_column("Value", justify="right", style="#66f5ff")
    table.add_column("Issue", justify="right", style="#fa1c0c")

    # SETUP TIME
    table.add_row("[bold underline]TIME CONTROL AND SETUP[/bold underline]", "", "")
    table.add_row("", "", "")
    table.add_row("Run time", f"{namelist.run_days:02}:{namelist.run_hrs:02}:{namelist.run_mins:02}:{namelist.run_secs:02}", "")
    table.add_row("", "", "")
    table.add_row("Outer dt", f"{namelist.time_step:4.2f} sec", "")
    table.add_row("Inner dt", f"{(namelist.time_step/namelist.time_step_rat):4.2f} sec", "")
    table.add_row("", "", "")
    ntasks_min  = max([namelist.outer_e_sn/100 * namelist.outer_e_we/100, 10])
    ntasks_max = namelist.inner_e_sn/25 * namelist.inner_e_we/25

    if (namelist.nproc_x != None) & (namelist.nproc_y != None):
        ntasks = namelist.nproc_x * namelist.nproc_y
        rows = int(namelist.nproc_y)
        cols = int(namelist.nproc_x)
    else:
        ntasks = opt_params['ntasks']
        rows, cols = distribute_processors(ntasks)

    y  = namelist.inner_e_sn * namelist.inner_dx  # Height of the rectangle
    dy = int(y // rows)  # Integer height of each internal row
    leftover_y = y - (dy * rows)

    y_pos_array = []
    for j in range(1, rows):
        y_pos = j * dy
        if j == rows - 1:  # Extend the final row to include leftover_y
            y_pos += leftover_y
        y_pos_array.append(y_pos)

    if (ntasks < ntasks_min) | (ntasks > ntasks_max):
        table.add_row("Processes", f"{ntasks:.0f}", f"[{ntasks_min:.0f}, {ntasks_max:.0f}]", end_section=True)
    else:
        table.add_row("Processes", f"{ntasks:.0f}", "", end_section=True)

    # TURBINE
    table.add_row("[bold underline]TURBINE[/bold underline]", "", "")
    table.add_row("", "", "")
    table.add_row("Model", f"{opt_params['turb_model']}", "")
    table.add_row("Diameter", f"{turbine.turb_diameter:.2f} m", "")
    table.add_row("Hub diameter", f"{turbine.hub_diameter:.2f} m", "")
    table.add_row("Inner domain x location", f"{(inner_i / turbine.turb_diameter):.2f} x D", "")
    table.add_row("Turbine x location", f"{(turbine_center_x /turbine.turb_diameter):.2f} x D", "")
    if (turbine.hubheight != (namelist.ztop/2)):
        table.add_row("z location", f"{turbine.hubheight:.1f} m", f">{(namelist.ztop/2):.2f}", end_section=True)
    else:
        table.add_row("z location", f"{turbine.hubheight:.1f} m", "", end_section=True)

    # ROTOR
    table.add_row("[bold underline]ROTOR[/bold underline]", "", "")
    table.add_row("", "", "")
    table.add_row("Model", f"{opt_params['rotor_model']}", "")

    if turbine.rot_dir == 1:
        rot_dir = 'clockwise'
    if turbine.rot_dir == -1:
        rot_dir = 'counterclockwise'
    if turbine.rot_dir == 0:
        rot_dir = 'irrotational'
    table.add_row("Rotation", f"{rot_dir}", "")

    if turbine.rot_dir == 0:
        table.add_row("Constant CT", f"{turbine.irrot_ct}", "")

    nSecmin = np.ceil(np.pi / (np.arcsin(namelist.inner_dx / turbine.turb_diameter)))
    if (namelist.nSections < nSecmin):
        table.add_row("Sections", f"{namelist.nSections:.0f}", f">{nSecmin:.0f}")
    else:
        table.add_row("Sections", f"{namelist.nSections:.0f}", "")
    nElmin = np.ceil(turbine.turb_diameter/2 / namelist.inner_dx)
    if (namelist.nElements < nElmin):
        table.add_row("Elements", f"{namelist.nElements:.0f}", f">{nElmin:.0f}")
    else:
        table.add_row("Elements", f"{namelist.nElements:.0f}", "")
    table.add_row("", "", "")
    if (turbine.inflow_loc < (turbine.turb_diameter*3)):
        table.add_row("V0 location", f"{(turbine.inflow_loc/turbine.turb_diameter):.1f} x D", ">3D", end_section=True)
    else:
        table.add_row("V0 location", f"{(turbine.inflow_loc/turbine.turb_diameter):.1f} x D", "", end_section=True)

    # DOMAIN
    table.add_row("[bold underline]DOMAIN[/bold underline]", "", "")
    table.add_row("", "", "")
    table.add_row("Outer mesh size", f"{outer_grid:,.0f}", "")
    table.add_row("Inner mesh Size", f"{inner_grid:,.0f}", "")
    table.add_row("Mesh ratio", f"{(inner_grid / outer_grid):.2f}", "")
    table.add_row("", "", "")
    table.add_row("Outer dx, dy", f"[{(namelist.outer_dx):4.1f}, {(namelist.outer_dy):4.1f}] m", "")
    table.add_row("Inner dx, dy", f"[{(namelist.inner_dx):4.1f}, {(namelist.inner_dy):4.1f}] m", "")
    if (round((namelist.ztop / namelist.e_vert)) != round((namelist.inner_dy))):
        table.add_row("dz", f"{(namelist.ztop / namelist.e_vert):.2f} m", f"~{(namelist.inner_dy):4.1f} m")
    else:
        table.add_row("dz", f"{(namelist.ztop / namelist.e_vert):.2f} m", "")

    table.add_row("", "", "")
    table.add_row("Outer [Lx, Ly, Lz]", f"[{(namelist.outer_e_we * namelist.outer_dx / turbine.turb_diameter):4.1f}, {(namelist.outer_e_sn * namelist.outer_dx / turbine.turb_diameter):4.1f}, {(namelist.ztop / turbine.turb_diameter):4.1f}] x D", "")
    if (namelist.inner_e_sn * namelist.inner_dx / turbine.turb_diameter) < 5:
        table.add_row("Inner [Lx, Ly, Lz]", f"[{(namelist.inner_e_we * namelist.inner_dx / turbine.turb_diameter):4.1f}, {(namelist.inner_e_sn * namelist.inner_dx / turbine.turb_diameter):4.1f}, {(namelist.ztop / turbine.turb_diameter):4.1f}] x D", "Ly > 5 x D")
    else:
        table.add_row("Inner [Lx, Ly, Lz]", f"[{(namelist.inner_e_we * namelist.inner_dx / turbine.turb_diameter):4.1f}, {(namelist.inner_e_sn * namelist.inner_dx / turbine.turb_diameter):4.1f}, {(namelist.ztop / turbine.turb_diameter):4.1f}] x D", "")

    table.add_row("", "", "")
    table.add_row("Outer [Nx, Ny, Nz]", f"[{(namelist.outer_e_we):4.0f}, {(namelist.outer_e_sn):4.0f}, {(namelist.e_vert):4.0f}]", "")
    table.add_row("Inner [Nx, Ny, Nz]", f"[{(namelist.inner_e_we):4.0f}, {(namelist.inner_e_sn):4.0f}, {(namelist.e_vert):4.0f}]", "")
    table.add_row("", "", "")
    if ((turbine.turb_diameter / (namelist.outer_e_sn * namelist.outer_dx)) > 0.05):
        table.add_row("Lateral blockage", f"{(turbine.turb_diameter / (namelist.outer_e_sn * namelist.outer_dx) * 100):.2f}%", f"<5%", end_section=True)
    else:
        table.add_row("Lateral blockage", f"{(turbine.turb_diameter / (namelist.outer_e_sn * namelist.outer_dx) * 100):.2f}%", "", end_section=True)

    # DECOMPOSITION
    table.add_row("[bold underline]DECOMPOSITION[/bold underline]", "", "")
    table.add_row("", "", "")
    if (np.floor(namelist.outer_e_we / cols) < 10):
        table.add_row("Outer cell width in x", f"{np.floor(namelist.outer_e_we / cols):.0f}", f">10")
    else:
        table.add_row("Outer cell width in x", f"{np.floor(namelist.outer_e_we / cols):.0f}", "")
    if (np.floor(namelist.outer_e_sn / rows) < 10):
        table.add_row("Outer cell width in y", f"{np.floor(namelist.outer_e_sn / rows):.0f}", f">10")
    else:
        table.add_row("Outer cell width in y", f"{np.floor(namelist.outer_e_sn / rows):.0f}", "")
    table.add_row("", "", "")
    if (np.floor(namelist.inner_e_we / cols) < 10):
        table.add_row("Inner cell width in x", f"{np.floor(namelist.inner_e_we / cols):.0f}", f">10")
    else:
        table.add_row("Inner cell width in x", f"{np.floor(namelist.inner_e_we / cols):.0f}", "")
    if (np.floor(namelist.inner_e_sn / rows) < 10):
        table.add_row("Inner cell width in y", f"{np.floor(namelist.inner_e_sn / rows):.0f}", f">10")
    else:
        table.add_row("Inner cell width in y", f"{np.floor(namelist.inner_e_sn / rows):.0f}", "")
    table.add_row("", "", "")
    if (np.mod(namelist.inner_e_we - 1, namelist.parent_grd_rat) != 0):
        table.add_row(f"MOD({namelist.inner_e_we} - 1, {namelist.parent_grd_rat})", f"{(np.mod(namelist.inner_e_we - 1, namelist.parent_grd_rat)):.0f}", f"{(np.mod(namelist.inner_e_we - 1, namelist.parent_grd_rat)):.0f} != 0")
    else:
        table.add_row(f"MOD({namelist.inner_e_we} - 1, {namelist.parent_grd_rat})", f"{(np.mod(namelist.inner_e_we - 1, namelist.parent_grd_rat)):.0f}", "")
    if (np.mod(namelist.inner_e_sn - 1, namelist.parent_grd_rat) != 0):
        table.add_row(f"MOD({namelist.inner_e_sn} - 1, {namelist.parent_grd_rat})", f"{(np.mod(namelist.inner_e_sn - 1, namelist.parent_grd_rat)):.0f}", f"{(np.mod(namelist.inner_e_sn - 1, namelist.parent_grd_rat)):.0f} != 0")
    else:
        table.add_row(f"MOD({namelist.inner_e_sn} - 1, {namelist.parent_grd_rat})", f"{(np.mod(namelist.inner_e_sn - 1, namelist.parent_grd_rat)):.0f}", "")
    table.add_row("", "", "")
    y_pos_array = np.array(y_pos_array)
    less_than_inflow = (y_pos_array[y_pos_array < turbine.turb_y])[-2]
    greater_than_inflow = (y_pos_array[y_pos_array > turbine.turb_y])[1]
    if (turbine.turb_y + turbine.turb_diameter/2) >= greater_than_inflow or (turbine.turb_y - turbine.turb_diameter/2) <= less_than_inflow:
        table.add_row("Rotor contained in pool?", "NO", f"YES", end_section=True)
    else:
        table.add_row("Rotor contained in pool?", "Yes", "", end_section=True)

    #INFLOW
    table.add_row("[bold underline]INFLOW[/bold underline]", "", "")
    table.add_row("", "", "", end_section=True)

    # Print the table to the console
    if 'print_table' in opt_params and opt_params['print_table']:
        console.print(table)
        table_text = console.export_text(styles=False)  # Optionally remove styles for plain text
    else:
        buffer = StringIO()            # Create an in-memory string buffer
        console = Console(file=buffer) # Create a Console instance that writes to the buffer
        console.print(table)           # Print the table to the buffer
        table_text = buffer.getvalue() # Retrieve the text from the buffer

    # Define the output filename
    output_filename = opt_params['save_to'] + "/overview.txt"

    # Write the exported table to the file
    with open(output_filename, "w") as file:
        file.write(table_text)

def combine_domain_plots(fig1: plt.Figure, fig2: plt.Figure, opt_params: Dict[str, Any]) -> None:
    """
    Combines the outer domain and inner domain images into a single image for easy of inspection.

    Args:
        fig1 (Figure): The plot object for the outer domain
        fig2 (Figure): The plot object for the inner domain
        opt_params (Dict): A dictionary of settings including sample locations if desired
    """

    fig1.savefig("plot1.png", dpi=500, bbox_inches='tight')  # Save the first plot as a PNG file
    fig2.savefig("plot2.png", dpi=500, bbox_inches='tight')  # Save the second plot as a PNG file
    
    # Open the saved PNG files using PIL
    img1 = Image.open("plot1.png")
    img2 = Image.open("plot2.png")

    # Convert the image to grayscale
    gray_img1 = img1.convert("L")
    bbox = gray_img1.point(lambda x: 0 if x == 255 else 1, "1").getbbox()

    left, upper, right, lower = bbox
    padded_bbox = (
        max(0, left - opt_params['outer_pad']),            # Left
        max(0, upper - opt_params['outer_pad']),           # Upper
        min(img1.width, right + opt_params['outer_pad']),  # Right
        min(img1.height, lower + opt_params['outer_pad'])  # Lower
    )

    # Crop image
    img1 = img1.crop(padded_bbox)
    
    # Get the width and height of each image
    width1, height1 = img1.size
    width2, height2 = img2.size
    
    # Resize the smaller image to match the width of the larger image while keeping aspect ratio
    if width1 > width2:
        new_height = int(height2 * (width1 / width2))
        img2 = img2.resize((width1, new_height), Image.Resampling.LANCZOS)  # Resize img2 to match img1's width
    elif width2 > width1:
        new_height = int(height1 * (width2 / width1))
        img1 = img1.resize((width2, new_height), Image.Resampling.LANCZOS)  # Resize img1 to match img2's width
    
    # After resizing, get the new dimensions of the images
    width1, height1 = img1.size
    width2, height2 = img2.size
    
    # Create a new image with the width equal to the width of the larger image
    total_height = height1 + height2
    total_width = max(width1, width2)
    
    combined_img = Image.new("RGB", (total_width, total_height))
    
    # Paste the first image (img1) at the top of the combined image
    combined_img.paste(img1, (0, 0))
    
    # Paste the second image (img2) at the bottom of the combined image
    combined_img.paste(img2, (0, height1))
    
    # Save the combined image as a PNG
    combined_img.save(opt_params['save_to'] + "/domains.png", dpi=(500, 500))
    
    # Optionally, remove the temporary individual PNG files
    os.remove("plot1.png")
    os.remove("plot2.png")

    # Close the images
    img1.close()
    img2.close()

def validate(opt_params: Dict[str, Any]) -> Tuple[Namelist, Turbine]:
    """
    Convenience function for:
        1. Parsing input files
        2. Generating summary table of relevant simulation parameters and flagging obvious issues
        2. Plotting (and saving) outer domain plot (if desired)
        3. Plotting (and saving) inner domain plot (if desired)
        4. Combining outer and inner domain plots into single image

    Args:
        opt_params (Dict): A dictionary of settings
    """
    # print('Point 3')

    # Parse setting files
    parsed_config   = parse_namelist(opt_params=opt_params)
    parsed_turbine  = parse_turbine_properties(opt_params=opt_params)
    parsed_location = parse_turbine_location(opt_params=opt_params)

    # print('Point 4')

    # Extract and format relevant variables in namedtuples
    namelist, turbine = load_variables(parsed_config=parsed_config, parsed_location=parsed_location, parsed_turbine=parsed_turbine)

    # print('Point 5')

    # Generate domain plots
    out_fig = plot_outer_domain(namelist=namelist, turbine=turbine, opt_params=opt_params)
    in_fig  = plot_inner_domain(namelist=namelist, turbine=turbine, opt_params=opt_params)

    # print('Point 6')

    if 'save_outer' in opt_params and opt_params['save_outer']:
        out_fig.savefig(opt_params['save_to'] + '/outer.png', dpi=500, bbox_inches='tight', pad_inches=0.05)

    if 'save_inner' in opt_params and opt_params['save_inner']:
        in_fig.savefig(opt_params['save_to'] + '/inner.png', dpi=500, bbox_inches='tight')

    if 'save_both' in opt_params and opt_params['save_both']:
        combine_domain_plots(fig1=out_fig, fig2=in_fig, opt_params=opt_params)

    # print('Point 7')

    # Generate simulation summary table
    summary_table(namelist=namelist, turbine=turbine, opt_params=opt_params)

    return namelist, turbine