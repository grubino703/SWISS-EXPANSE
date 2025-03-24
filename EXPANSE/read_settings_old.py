"""
Copyright 2022 Jan-Philipp Sasse (UNIGE), Evelina Trutnevyte (UNIGE)
~~~~~~~~~~~~~
Read model settings, and technology techno-economic parameters

Inputs: setttings.csv, costs.csv
Outputs: dictionary with all settings -> 'opts'
"""

# Import required packages
import pandas as pd

def read_settings():
    
    # Read model settings
    exp_settings = pd.read_csv('../settings/settings.csv', index_col=0)
    
    # Extract settings
    solver_name = str(exp_settings.loc['solver_name','value'])
    threads = int(exp_settings.loc['threads','value'])
    presolve = int(exp_settings.loc['presolve','value'])
    write_log_files = int(exp_settings.loc['write_log_files','value'])
    write_lp_files = int(exp_settings.loc['write_lp_files','value'])
    barrier_tol = float(exp_settings.loc['barrier_tol','value'])
    feas_tol = float(exp_settings.loc['feas_tol','value'])
    hours = int(exp_settings.loc['hours','value'])
    resolution = int(exp_settings.loc['resolution','value'])
    input_path = str(exp_settings.loc['input_path','value'])
    output_path = str(exp_settings.loc['output_path','value'])
    assumptions_path = str(exp_settings.loc['assumptions_path','value'])
    slack = float(exp_settings.loc['slack','value'])
    r_cap = float(exp_settings.loc['r_cap','value'])
    
    # Setup solver options depending on the solver type
    if (solver_name == 'cplex_direct') or (solver_name == 'cplex'):
        solv_opts = {'threads': threads, 'lpmethod': 4, 'solutiontype':2,
        'barrier_convergetol': barrier_tol, 'feasopt_tolerance': feas_tol, 
        'preprocessing_presolve': presolve}

    elif solver_name == 'gurobi':
        solv_opts = {'threads': threads, 'method': 2,'crossover': 0,
        'BarConvTol': barrier_tol, 'FeasibilityTol': feas_tol,
        'AggFill': 0, 'PreDual': 0, 'GURO_PAR_BARDENSETHRESH': 200, 
        'Presolve': presolve}

    elif solver_name == 'cbc':
        solv_opts = {'threads': threads, 'startalg': 'barrier', 'crossover': 0, 'tol_primal': feas_tol}

    elif solver_name == 'mosek':
        solv_opts = {'MSK_IPAR_NUM_THREADS': threads}

    elif solver_name == 'ipopt':
        solv_opts = {'tol': barrier_tol}

    else:
        solv_opts = {}

    # Read the costs file
    fname = assumptions_path + 'costs.csv'
    costs = pd.read_csv(fname, index_col=[0,2])

    # Store all the settings into a dictionary
    opts = dict()
    opts['hours'] = hours
    opts['resolution'] = resolution
    opts['slack'] = slack
    opts['r_cap'] = r_cap
    opts['costs'] = costs
    opts['input_path'] = input_path
    opts['output_path'] = output_path
    opts['assumptions_path'] = assumptions_path
    opts['solver_name'] = solver_name
    opts['solv_opts'] = solv_opts
    opts['write_log_files'] = write_log_files
    opts['write_lp_files'] = write_lp_files
        
    return opts
