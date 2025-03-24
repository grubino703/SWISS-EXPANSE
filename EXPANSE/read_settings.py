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
    solver_options = str(exp_settings.loc['solver_options','value'])
    write_log_files = int(exp_settings.loc['write_log_files','value'])
    write_lp_files = int(exp_settings.loc['write_lp_files','value'])
    hours = int(exp_settings.loc['hours','value'])
    resolution = int(exp_settings.loc['resolution','value'])
    input_path = str(exp_settings.loc['input_path','value'])
    output_path = str(exp_settings.loc['output_path','value'])
    assumptions_path = str(exp_settings.loc['assumptions_path','value'])
    slack = float(exp_settings.loc['slack','value'])
    r_cap = float(exp_settings.loc['r_cap','value'])

    # H2 Addition
    # Extract H2 settings
    h2_settings = pd.read_csv('../settings/settings_h2.csv', sep=";", index_col = 0)
    
    # Setup solver options depending on the solver type
    if solver_options == 'cplex-default':
        solv_opts = {'threads': 4, 'lpmethod': 4, 'solutiontype': 2,
        'barrier_convergetol': 1.e-5, 'feasopt_tolerance': 1.e-6}
        solver_name = 'cplex_direct'

    elif solver_options == 'gurobi-default':
        solv_opts = {'threads': 4, 'method': 2, 'crossover': 0,
        'BarConvTol': 1.e-6, 'Seed': 123, 'AggFill': 0, 
        'PreDual': 0, 'GURO_PAR_BARDENSETHRESH': 200}
        solver_name = 'gurobi'

    elif solver_options == 'gurobi-stability':
        solv_opts = {'NumericFocus': 3, 'method': 2, 'crossover': 0, 
        'BarHomogeneous': 1, 'BarConvTol': 1.e-5, 
        'FeasibilityTol': 1.e-4, 'OptimalityTol': 1.e-4, 
        'ObjScale': -0.5, 'threads': 8, 'Seed': 123}
        solver_name = 'gurobi'

    elif solver_options == 'gurobi-factory-settings':
        solv_opts = {'crossover': 0, 'method': 2, 'BarHomogeneous': 1,
        'BarConvTol': 1.e-5, 'FeasibilityTol': 1.e-5, 
        'OptimalityTol': 1.e-5, 'Seed': 123, 'threads': 8}
        solver_name = 'gurobi'

    elif solver_options == 'gurobi-speed':
        solv_opts = {'threads': 4, 'method': 2, 'crossover': 0,
        'BarConvTol': 1.e-4, 'FeasibilityTol': 1.e-3, 'OptimalityTol': 1.e-3,
        'Seed': 123, 'AggFill': 0, 
        'PreDual': 0, 'GURO_PAR_BARDENSETHRESH': 200}
        solver_name = 'gurobi'

    elif solver_options == 'gurobi-highspeed':
        solv_opts = {'threads': 4, 'method': 2, 'crossover': 0,
        'BarConvTol': 1.e-4, 'FeasibilityTol': 1.e-4,
        'AggFill': 0, 'PreDual': 0, 'GURO_PAR_BARDENSETHRESH': 200}
        solver_name = 'gurobi'

    elif solver_options == 'cbc-default':
        solv_opts = {}
        solver_name = 'cbc'

    elif solver_options == 'glpk-default':
        solv_opts = {}
        solver_name = 'glpk'

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

    # H2 Addition
    # load the additional variables from the storyline defined in the settings
    opts['h2_scenario'] = str(h2_settings.loc['Scenario', 'Value'])
    opts['hydrogen_assumptions_path'] = '../settings/variables_scenarios_2050.csv'
    opts['hydrogen_assumptions'] = pd.read_csv(opts['hydrogen_assumptions_path'], sep=";", index_col=0, encoding="latin1")


    return opts
