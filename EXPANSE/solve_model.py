"""
Copyright 2021 Jan-Philipp Sasse (UNIGE), Evelina Trutnevyte (UNIGE)
~~~~~~~~~~~~~
Solve the Pyomo model instance

"""

# Import packages
import pandas as pd
import time
from pyomo.environ import *
import glob
import numpy as np
from EXPANSE.read_settings import *
from EXPANSE.read_infrastructure import *

# Read settings
opts = read_settings()
network = read_infrastructure(opts['input_path'],opts['resolution'], opts['h2_scenario'])

def print_out(model,results,start,end):
    
    '''
    Prints out overview of the results
    '''
    
    print('# ----------------------------------------------------------')
    print('#   Overall Results Information')
    print('# ----------------------------------------------------------')
    print('Time to solve model: {} seconds'.format(round((end-start))))
    print('Total system cost:',round(model.cost()/1e9,3),'BEUR')
    print("Total jobs:", round(model.jobs()/1e3,3), "thousand")
    print("Total GHG emissions:", round(model.ghg()/1e6,3), "MtCO2")
    print("Total PM10 emissions:", round(model.pm10()/1e6,3), "ktPM10")
    print("Total land use:", round(model.landuse()/1e6,3), "km2")
    print("Termination:", results.solver.termination_condition.value)
    print("Status:", results.solver.status.value)

def solve_model_mincost(model):
    """
    Solves the cost-minimization model
    """

    start = time.time()

    model.cost.activate()

    if bool(opts['write_lp_files']):
        fname = opts['output_path'] + 'model.lp'
        model.write(fname, format="lp", io_options={"symbolic_solver_labels": True})

    results = SolverFactory(opts['solver_name']).solve(model,
                                                       options=opts['solv_opts'], tee=bool(opts['write_log_files']))
    results.write()

    model.cost.deactivate()

    end = time.time()

    if results.solver.termination_condition.value == 'optimal':
        print_out(model, results, start, end)

    return model, results


def get_mincost(cur_configuration):
    
    """
    Gets the cost of the minimum cost scenario
    """

    fname = opts['output_path'] + cur_configuration +  '/case.csv'
    df = pd.read_csv(fname, index_col=0)
    min_cost = df.loc[cur_configuration,'Cost (BEUR)'] * 1e9

    return min_cost

def create_mga_scenario():
    
    """
    Creates the MGA scenario objectives dataframe
    Returns a dataframe with Min and Max values for each generator and storage unit
    """

    MGA_items = list(network['generators'].index)
    MGA_items += list(network['storage_units'].index)
    MGA_Objective = pd.DataFrame(index=MGA_items)
    for ID in MGA_Objective.index:
        MGA_Objective.loc[ID,'MinMax'] = ['Min', '', 'Max'][np.random.randint(0, 3)]
    
    return MGA_Objective

def mga_obj_expr(model, MGA_Objective):
    """
    Creates the MGA objective expression for Pyomo
    """

    expr = 0

    for c in MGA_Objective.index:
        if MGA_Objective.loc[c,'MinMax'] == 'Max':
            try:
                # if generator
                expr += model.p_nom[c]
            except:
                # if storage unit
                expr += model.p_s_nom[c]
        if MGA_Objective.loc[c,'MinMax'] == 'Min':
            try:
                # if generator
                expr -= model.p_nom[c]
            except:
                # if storage unit
                expr -= model.p_s_nom[c]

    return expr

def solve_model_mga(model, scen_nr, cur_configuration):
    
    """
    Solves the MGA scenarios
    """

    start = time.time()

    # Get cost of MinCost scenario
    min_cost = get_mincost(cur_configuration)

    # Create random MGA objective and cost constraint
    MGA_Objective = create_mga_scenario()
    MGA_CostSlack = np.random.randint(1, 101) / 100 * opts['slack']

    # Set-up model objective and constraint
    model.MGA = Objective(expr = mga_obj_expr(model, MGA_Objective), sense = maximize)
    model.cost_constraint = Constraint(expr = model.cost.expr <= min_cost * (1 + MGA_CostSlack / 100))

    # Print
    print('MGA run #{} with Cost Slack {}%.'.format(scen_nr, MGA_CostSlack))

    # Solve the model
    results = SolverFactory(opts['solver_name']).solve(model,
                                                       options=opts['solv_opts'], tee=bool(opts['write_log_files']))
    results.write()

    # Delete the constraints
    model.del_component(model.MGA)
    model.del_component(model.cost_constraint)

    end = time.time()
    print_out(model, results, start, end)

    return model, results, MGA_Objective, MGA_CostSlack


def solve_model_current(model):
    
    """
    Solves the frozen scenario with current generation, storage, and transmission capacities
    """

    start = time.time()

    # Freeze the generation capacities
    for c in list(network['generators'].index):
        model.p_nom[c].fix(network['generators'].loc[c, 'p_nom_cur'])

    # Freeze the storage capacities
    for c in list(network['storage_units'].index):
        model.p_s_nom[c].fix(network['storage_units'].loc[c, 'p_nom_cur'])

    # Freeze the transmission capacities
    for c in list(network['links'].index):
       model.links_p_nom[c].fix(network['links'].loc[c, 'p_nom_cur'])

    for c in list(network['lines'].index):
       model.lines_p_nom[c].fix(network['lines'].loc[c, 'p_nom_cur'])

    # Solve the model
    model.cost.activate()
    results = SolverFactory(opts['solver_name']).solve(model, options = opts['solv_opts'],
                                                       tee = bool(opts['write_log_files']))
    results.write()
    model.cost.deactivate()

    end = time.time()
    print_out(model, results, start, end)

    return model, results
