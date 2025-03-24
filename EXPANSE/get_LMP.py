"""
Copyright 2022 Jan-Philipp Sasse (UNIGE), Evelina Trutnevyte (UNIGE)
~~~~~~~~~~~~~
Initialize a new Pyomo model to get the locational marginal prices.
This model runs only in operation mode.

Inputs: pre-solved electricity system infrastructure from capacity expansion mode
Outputs: locational marginal prices
"""

# Import packages
import time
import networkx as nx
from pyomo.environ import *
import pandas as pd
import numpy as np
from EXPANSE.read_settings import *

opts_LMP = read_settings()

def read_case_LMP(scenario_path):
    
    """
    Read pre-solved electricity system infrastructure of the scenario

    Input: scenario path 
    Output: pre-solved network files
    """

    # Read hourly demand
    f_loads_p_set = scenario_path + 'loads_p_set.csv'
    loads_p_set = pd.read_csv(f_loads_p_set, index_col=[0])

    # Read buses
    f_buses = scenario_path + 'buses.csv'
    buses = pd.read_csv(f_buses, index_col=[0])

    # Read hourly generator CF
    f_generators_p_max_pu = scenario_path + 'generators_p_max_pu.csv'
    generators_p_max_pu = pd.read_csv(f_generators_p_max_pu, index_col=[0])

    # Read generators
    f_generators = scenario_path + 'generators.csv'
    generators = pd.read_csv(f_generators, index_col=[0])

    # Read storage units
    f_storage_units = scenario_path + 'storage_units.csv'
    storage_units = pd.read_csv(f_storage_units, index_col=[0])

    # Read storage inflow
    f_storage_inflow = scenario_path + 'storage_inflow.csv'
    storage_inflow = pd.read_csv(f_storage_inflow, index_col=[0])

    # Read HVAC transmission lines
    f_lines = scenario_path + 'lines.csv'
    lines = pd.read_csv(f_lines, index_col=[0])

    # Read HVDC transmission links
    f_links = scenario_path + 'links.csv'
    links = pd.read_csv(f_links, index_col=[0])

    # Store all dataframes into a dictionary
    network = dict()
    network['loads_p_set'] = loads_p_set
    network['buses'] = buses
    network['generators_p_max_pu'] = generators_p_max_pu
    network['generators'] = generators
    network['storage_units'] = storage_units
    network['storage_inflow'] = storage_inflow
    network['lines'] = lines
    network['links'] = links

    return network


def init_model_LMP(network):
    
    """
    Returns the initialized LMP model

    Input: network
    Output: pyomo model
    """

    ### Initialize Pyomo model
    model = ConcreteModel()

    ### Create variable indices
    # Generation technologies, importers, exporters
    Gen_IDs = list(network['generators'].index)
    
    # Storage technologies
    Sto_IDs = list(network['storage_units'].index)
    
    # HVAC lines
    Line_IDs = list(network['lines'].index)

    # HVDC lines
    Link_IDs = list(network['links'].index)

    # Snapshots to optimize
    Snapshots = list(network['loads_p_set'].index)

    # Storage units with inflow
    Inflow_IDs = list(network['storage_inflow'].columns)

    # Fixed storage units
    Fixed_storage_IDs = list(network['storage_units'][network['storage_units'].p_nom_extendable==False].index)

    # Nodes of system
    nodes = list(network['buses'].index)

    # Variable generation technologies
    variable_techs = []
    for col in network['generators_p_max_pu'].columns:
        variable_techs.append(col.split()[-1])
    variable_techs = list(set(variable_techs))

    # Flexible technologies (for reserve capacity constraint)
    flex_techs = list(np.setdiff1d(network['generators'].carrier.unique(), variable_techs))
    flex_techs += list(network['storage_units'].carrier.unique())

    # Inflexible GenIDs (where ramping matters)
    inflex_GenIDs = []
    for c in network['generators'].index:
        if network['generators'].loc[c,'ramp_limit'] * opts_LMP['resolution'] < 1:
            inflex_GenIDs.append(c)

    ### Create all the required dictionaries to speed up model
    generators_dict = network['generators'].to_dict()
    storage_units_dict = network['storage_units'].to_dict()
    lines_dict = network['lines'].to_dict()
    links_dict = network['links'].to_dict()
    g_p_max_pu_dict = network['generators_p_max_pu'].to_dict()
    inflow_dict = network['storage_inflow'].to_dict()
    loads_dict = network['loads_p_set'].to_dict()

    ### Create variables
    # Generator power output
    model.p = Var(Gen_IDs, Snapshots, domain = Reals, initialize = 0.0)

    # Initialize and fix generator installed capacity
    model.p_nom = Var(Gen_IDs, domain = NonNegativeReals)
    for c in Gen_IDs:
        model.p_nom[c].fix(generators_dict['p_nom_opt'][c])

    # Storage dispatch
    model.dispatch = Var(Sto_IDs, Snapshots, domain = NonNegativeReals, initialize = 0.0)

    # Storage store
    model.store = Var(Sto_IDs, Snapshots, domain = NonNegativeReals, initialize = 0.0)

    # Storage spillage
    model.spillage = Var(Inflow_IDs, Snapshots, domain = NonNegativeReals, initialize = 0.0)

    # Initialize and fix storage installed capacity
    model.p_s_nom = Var(Sto_IDs, domain = NonNegativeReals)
    for c in Sto_IDs:
        model.p_s_nom[c].fix(storage_units_dict['p_nom_opt'][c])

    # Storage state of charge
    model.soc = Var(Sto_IDs, Snapshots, domain = NonNegativeReals, initialize = 0.0)

    # Initialize and fix line installed capacity
    model.lines_p_nom = Var(Line_IDs, domain = NonNegativeReals)
    for l in Line_IDs:
        model.lines_p_nom[l].fix(lines_dict['p_nom_opt'][l])

    # Line power flow from bus0 to bus1
    model.lines_p_p0 = Var(Line_IDs, Snapshots, domain = NonNegativeReals, initialize = 0.0)

    # Line power flow from bus1 to bus0
    model.lines_p_p1 = Var(Line_IDs, Snapshots, domain = NonNegativeReals, initialize = 0.0)

    # Initialize and fix link installed capacity
    model.links_p_nom = Var(Link_IDs, domain = NonNegativeReals)
    for l in Link_IDs:
        model.links_p_nom[l].fix(links_dict['p_nom_opt'][l])

    # Link power flow from bus0 to bus1
    model.links_p_p0 = Var(Link_IDs, Snapshots, domain = NonNegativeReals, initialize = 0.0)

    # Link power flow from bus1 to bus0
    model.links_p_p1 = Var(Link_IDs, Snapshots, domain = NonNegativeReals, initialize = 0.0)

    ### Line/link power flow constraint: power flow < capacity
    def lines_p_p0_max_rule(model,l,t):
        return(model.lines_p_p0[l,t] <= lines_dict['p_max_pu'][l] * model.lines_p_nom[l])

    def lines_p_p1_max_rule(model,l,t):
        return(model.lines_p_p1[l,t] <= lines_dict['p_max_pu'][l] * model.lines_p_nom[l])

    def links_p_p0_max_rule(model,l,t):
        return(model.links_p_p0[l,t] <= links_dict['p_max_pu'][l] * model.links_p_nom[l])

    def links_p_p1_max_rule(model,l,t):
        return(model.links_p_p1[l,t] <= links_dict['p_max_pu'][l] * model.links_p_nom[l])

    model.lines_p_p0_max = Constraint(Line_IDs, Snapshots, rule = lines_p_p0_max_rule)
    model.lines_p_p1_max = Constraint(Line_IDs, Snapshots, rule = lines_p_p1_max_rule)
    model.links_p_p0_max = Constraint(Link_IDs, Snapshots, rule = links_p_p0_max_rule)
    model.links_p_p1_max = Constraint(Link_IDs, Snapshots, rule = links_p_p1_max_rule)

    ### Generator maximum power output constraint
    def p_max_rule(model,c,t):
        
        # Variable generator: p(t) < p_nom * CF_max(t)
        if c.split()[-1] in variable_techs:
            expr = model.p[c,t] <= g_p_max_pu_dict[c][t] * model.p_nom[c]
        
        # Flexible generator: p(t) < p_nom
        else:
            expr = model.p[c,t] <= generators_dict['p_max_pu'][c] * model.p_nom[c]
        
        return(expr)

    model.p_max = Constraint(Gen_IDs, Snapshots, rule = p_max_rule)

    ### Generator minimum power output constraint
    def p_min_rule(model,c,t):
        
        return(model.p[c,t] >= generators_dict['p_min_pu'][c] * model.p_nom[c])

    model.p_min = Constraint(Gen_IDs, Snapshots, rule = p_min_rule)

    ### Add constraint for reserve capacity margin
    # Count the total flexible generation capacity
    def flex_gen_cap(model):
        
        # Initialize expression
        expr = 0
        
        # Iterate through all flexible generators and add capacity
        for c in Gen_IDs:
            if c.split()[-1] in flex_techs:
                expr += model.p_nom[c]    
        
        return expr

    # Count the total flexible storage capacity
    def flex_sto_cap(model):
        
        # Initialize expression
        expr = 0
        
        # Iterate through all flexible storage units and add capacity
        for c in Sto_IDs:
            if c.split()[-1] in flex_techs:
                expr += model.p_s_nom[c]
                
        return expr

    # Count the total utilized power output of flexible generators
    def flex_gen_pout(model,t):
        
        # Initialize expression
        expr = 0
        
        # Iterate through all flexible generators and add power output
        for c in Gen_IDs:
            if c.split()[-1] in flex_techs:
                expr += model.p[c,t]
        
        return expr

    # Count the total utilized power output of flexible storage units
    def flex_sto_pout(model,t):
        
        # Initialize expression
        expr = 0
        
        # Iterate through all flexible storage units and add power output
        for c in Sto_IDs:
            if c.split()[-1] in flex_techs:
                expr += model.dispatch[c,t]
        
        return expr

    # Create reserve capacity constraint
    def reserve_capacity_rule(model,t):
        
        Flex_Cap = flex_gen_cap(model) + flex_sto_cap(model)
        Flex_Gen = flex_gen_pout(model,t) + flex_sto_pout(model,t)
        Flex_Cap_Unused = Flex_Cap - Flex_Gen
        Reserve_Cap_Requirement = opts_LMP['r_cap'] * network['loads_p_set'].sum(axis=1).max()
        
        return(Flex_Cap_Unused >= Reserve_Cap_Requirement)

    model.reserve_capacity = Constraint(Snapshots, rule = reserve_capacity_rule)

    ### Storage dispatch constraint
    # Constraint for storage dispatch: h(c,t) < h_nom(c)
    def h_max_rule(model,c,t):
        return(model.dispatch[c,t] <= storage_units_dict['p_max_pu'][c] * model.p_s_nom[c])

    model.h_max = Constraint(Sto_IDs, Snapshots, rule = h_max_rule)

    ### Storage charging constraint
    # Constraint for storage charging: f(c,t) < h_nom(c)
    def f_max_rule(model,c,t):
        return(model.store[c,t] <= - storage_units_dict['p_min_pu'][c] * model.p_s_nom[c])

    model.f_max = Constraint(Sto_IDs, Snapshots, rule = f_max_rule)

    ### Maximum state of charge constraint
    # Constraint for max. storage state of charge: E < P * h (discharge hours)
    def soc_max_rule(model,c,t):
        return(model.soc[c,t] <= storage_units_dict['max_hours'][c] * model.p_s_nom[c])

    model.soc_max = Constraint(Sto_IDs, Snapshots, rule = soc_max_rule)
    
    ### Hourly state of charge constraint
    # Relates the SOC of previous hours to current hour
    # soc(c,t) = soc(c,t-1) + n(store,c) * f(c,t) - 1/n(dispatch,c) * h(c,t) + inflow(c,t) - spillage(c,t) 
    def soc_cons_rule(model,c,t):
        
        try:
            t_1 = Snapshots[Snapshots.index(t)-1]
        except:
            t_1 = Snapshots[-1]
        
        expr = 0
        expr += model.soc[c,t_1]
        expr += storage_units_dict['efficiency_store'][c] * opts_LMP['resolution'] * model.store[c,t]
        expr -= 1 / storage_units_dict['efficiency_dispatch'][c] * opts_LMP['resolution'] * model.dispatch[c,t]
        
        if c in Inflow_IDs:
            expr += model.p_s_nom[c] * inflow_dict[c][t] * opts_LMP['resolution']
            expr -= model.spillage[c,t] * opts_LMP['resolution']
        
        return(expr == model.soc[c,t])

    model.soc_cons = Constraint(Sto_IDs, Snapshots, rule = soc_cons_rule)

    ### Nodal power balance constraint
    # Sum of generator output of selected node and snapshot
    def p_sum(model,n,t):
        p_sum_expr = 0
        for c in Gen_IDs:
            if generators_dict['bus'][c] == n:
                p_sum_expr += model.p[c,t]
        return p_sum_expr

    # Sum of storage dispatch of selected node and snapshot
    def h_sum(model,n,t):
        h_sum_expr = 0
        for c in Sto_IDs:
            if storage_units_dict['bus'][c] == n:
                h_sum_expr += model.dispatch[c,t]
        return h_sum_expr

    # Sum of storage charging of selected node and snapshot
    def f_sum(model,n,t):
        f_sum_expr = 0
        for c in Sto_IDs:
            if storage_units_dict['bus'][c] == n:
                f_sum_expr += model.store[c,t]
        return f_sum_expr

    # Sum power flow that are connected to each node
    def flow_sum(model,n,t):
        
        flow_sum_expr = 0
        
        for l in Line_IDs:
            
            # if line starts at the node
            if lines_dict['bus0'][l] == n:
                flow_sum_expr += lines_dict['efficiency'][l] * model.lines_p_p1[l,t]
                flow_sum_expr -= model.lines_p_p0[l,t]
            
            # if line ends at the node
            if lines_dict['bus1'][l] == n:
                flow_sum_expr += lines_dict['efficiency'][l] * model.lines_p_p0[l,t]
                flow_sum_expr -= model.lines_p_p1[l,t]
                
        for l in Link_IDs:
            
            # if link starts at the node
            if links_dict['bus0'][l] == n:
                flow_sum_expr += links_dict['efficiency'][l] * model.links_p_p1[l,t]
                flow_sum_expr -= model.links_p_p0[l,t]
            
            # if link ends at the node
            if links_dict['bus1'][l] == n:
                flow_sum_expr += links_dict['efficiency'][l] * model.links_p_p0[l,t]
                flow_sum_expr -= model.links_p_p1[l,t]
            
        return flow_sum_expr

    # Create nodal power balances
    def p_balance_rule(model,n,t):

        LHS = p_sum(model,n,t) + h_sum(model,n,t) - f_sum(model,n,t) + flow_sum(model,n,t)
        RHS = loads_dict[n][t]

        return(LHS == RHS)

    model.p_balance = Constraint(nodes, Snapshots, rule = p_balance_rule)

    ### Kirchhoff Voltage Law constraint
    # Get nodes that are connected to lines
    line_nodes = []
    for b in network['lines'].bus0:
        line_nodes.append(b)
    for b in network['lines'].bus1:
        line_nodes.append(b)
    line_nodes = list(set(line_nodes))

    # Create list and dictionary of lines and directions
    edgelist = []
    mapping = dict()
    sign = dict()
    for line in network['lines'].index:
        mapping[(network['lines'].loc[line,'bus0'], network['lines'].loc[line,'bus1'])] = line
        sign[(network['lines'].loc[line,'bus0'], network['lines'].loc[line,'bus1'])] = 1
        mapping[(network['lines'].loc[line,'bus1'], network['lines'].loc[line,'bus0'])] = line
        sign[(network['lines'].loc[line,'bus1'], network['lines'].loc[line,'bus0'])] = -1
        edgelist.append([network['lines'].loc[line,'bus0'],network['lines'].loc[line,'bus1']])

    # Create NetworkX graph and add the nodes and edges of lines
    G = nx.Graph()
    G.add_nodes_from(line_nodes)
    G.add_edges_from(edgelist)

    # Builds list of tuples of nodes representing cycle edges
    def build_cycle(nodes):
        
        cycle = []
        
        for i in range(len(nodes)-1):
            cycle.append((nodes[i],nodes[i+1]))
        
        cycle.append((nodes[-1],nodes[0]))
            
        return cycle

    # Find all cycles in graph
    cycle_basis = nx.cycle_basis(G)

    # Create empty lists of ordered cycles (buses, and lines)
    buses_cycles = []
    lines_cycles = []

    # Iterate through each cycle basis
    for cycle in cycle_basis:
        
        # Create tuple of buses from list of buses in cycle
        buses_cycle = build_cycle(cycle)
        
        # Translate bus tuples to tuples of lines and directions
        lines_cycle = []
        for bus_tuple in buses_cycle:
            lines_cycle.append((mapping[bus_tuple],sign[bus_tuple]))
        
        # Append the line tuples to list of cycles
        lines_cycles.append(lines_cycle)
        
        # Append the bus tuples to list of cycles
        buses_cycles.append(buses_cycle)

    # Builds Kirchhoff Voltage Law constraints
    model.cyc_cons = ConstraintList()
    for lines_cycle in lines_cycles:
        for t in Snapshots:
            cycle_expr = 0
            for i in range(len(lines_cycle)):
                cur_sign = lines_cycle[i][1]
                cur_name = lines_cycle[i][0]
                cycle_expr += cur_sign * model.lines_p_p0[cur_name,t] * lines_dict['x'][cur_name]
                cycle_expr -= cur_sign * model.lines_p_p1[cur_name,t] * lines_dict['x'][cur_name]
            model.cyc_cons.add(expr = cycle_expr == 0.0)

    ### Add ramping constraints
    # Ramping up constraint: p(c,t) - p(c,t-1) <= p_nom(c) * ramp_limit(c) * resolution
    # for t in 1 -> end
    def ramp_up_rule(model,c,t):
        
        t_1 = Snapshots[Snapshots.index(t)-1]
        
        ramp_up_MW = model.p[c,t] - model.p[c,t_1]
        ramp_up_limit_MW = model.p_nom[c] * generators_dict['ramp_limit'][c] * opts_LMP['resolution']
        
        return(ramp_up_MW <= ramp_up_limit_MW)

    model.ramp_up = Constraint(inflex_GenIDs, Snapshots[1:], rule = ramp_up_rule)

    # Ramping down constraint: p(c,t) - p(c,t-1) >= (-1) * p_nom(c) * ramp_limit(c) * resolution
    # for t in 1 -> end
    def ramp_down_rule(model,c,t):
        
        t_1 = Snapshots[Snapshots.index(t)-1]
        
        ramp_down_MW = model.p[c,t] - model.p[c,t_1]
        ramp_down_limit_MW = (-1) * model.p_nom[c] * generators_dict['ramp_limit'][c] * opts_LMP['resolution']
        
        return(ramp_down_MW >= ramp_down_limit_MW)

    model.ramp_down = Constraint(inflex_GenIDs, Snapshots[1:], rule = ramp_down_rule)


    ### Objective function
    def gen_cap_cost(model):
        return sum(model.p_nom[c] * generators_dict['capital_cost'][c]
                   for c in Gen_IDs)

    def gen_op_cost(model):
        return sum(model.p[c, t] * generators_dict['marginal_cost'][c]
                   for c in Gen_IDs for t in Snapshots) * 8760 / len(Snapshots)

    def sto_cap_cost(model):
        return sum(model.p_s_nom[c] * storage_units_dict['capital_cost'][c]
                   for c in Sto_IDs)

    def sto_op_cost(model):
        return sum(model.dispatch[c, t] * storage_units_dict['marginal_cost'][c]
                   for c in Sto_IDs for t in Snapshots) * 8760 / len(Snapshots)

    def line_cap_cost(model):
        return sum(model.lines_p_nom[l] * lines_dict['capital_cost'][l]
                   for l in Line_IDs)

    def line_op_cost(model):
        return sum(model.lines_p_p0[l,t] * lines_dict['marginal_cost'][l] + model.lines_p_p1[l,t] * lines_dict['marginal_cost'][l]
                   for l in Line_IDs for t in Snapshots) * 8760 / len(Snapshots)

    def link_cap_cost(model):
        return sum(model.links_p_nom[l] * links_dict['capital_cost'][l]
                   for l in Link_IDs)

    def link_op_cost(model):
        return sum(model.links_p_p0[l,t] * links_dict['marginal_cost'][l] + model.links_p_p1[l,t] * links_dict['marginal_cost'][l]
                   for l in Link_IDs for t in Snapshots) * 8760 / len(Snapshots)

    def tot_cost(model):
        
        expr = 0
        expr += gen_cap_cost(model) + gen_op_cost(model)
        expr += sto_cap_cost(model) + sto_op_cost(model)
        expr += line_cap_cost(model) + line_op_cost(model)
        expr += link_cap_cost(model) + link_op_cost(model)
        
        return expr

    # Assign model objectives
    model.cost = Objective(expr = tot_cost(model), sense = minimize)

    # Create dual and write indexer (constraint index <-> node, snapshot)
    model.dual = Suffix(direction=Suffix.IMPORT)

    return model


def print_out_LMP(model, results, start, end):
    
    """
    Prints out results
    """

    print('Time to solve model: {} seconds'.format(round((end - start))))
    print('Total system cost:', round(model.cost()/1e9, 3), 'BEUR')
    print("Termination:", results.solver.termination_condition.value)
    print("Status:", results.solver.status.value)


def solve_model_LMP(model, solver_name, solv_opts):
    
    """
    Solves the model to minimize cost and get LMPs
    """

    start = time.time()
    results = SolverFactory(solver_name).solve(model, options = solv_opts, tee = bool(opts_LMP['write_log_files']))
    end = time.time()
    print_out_LMP(model, results, start, end)

    return model, results


def read_LMP_bus(model, network):
    
    """
    Reads the LMPs at bus-level
    """

    duals = pd.Series(list(model.dual.values()), index=pd.Index(list(model.dual.keys())))
    marg_prices = pd.Series(list(model.p_balance.values()),
        index=list(model.p_balance.keys())).map(duals).divide(8760/len(network['loads_p_set']))

    LMP_bus = pd.DataFrame(index=list(model.p_balance_index_1),
        columns=list(model.p_balance_index_0))

    for n in LMP_bus.columns:
        for t in LMP_bus.index:
            LMP_bus.loc[t,n] = marg_prices[(n,t)]

    LMP_bus.index.name = 'name'

    return LMP_bus


def read_LMP_nuts(input_path, LMP_bus, network):
    
    """
    Reads the LMPs at municipality-level
    """

    fname = input_path + 'geometry.csv'
    NUTS = pd.read_csv(fname, index_col=[0])

    LMP_nuts = pd.DataFrame(columns=LMP_bus.index, index=NUTS.index)
    for r in NUTS.index:
        for t in network['loads_p_set'].index:
            LMP_nuts.loc[r, t] = LMP_bus.loc[t, NUTS.loc[r, 'bus']]
    for t in network['loads_p_set'].index:
        LMP_nuts[t] = pd.to_numeric(LMP_nuts[t])
    LMP_nuts = LMP_nuts.T
    LMP_nuts.index.name = 'name'

    return LMP_nuts


def save_LMP(scenario_path, LMP_bus, LMP_nuts):
    
    """
    Save locational marginal prices
    """

    # Export bus-level locational marginal prices
    fname = scenario_path + '/LMP_bus.csv'
    LMP_bus.to_csv(fname)
    
    # Export nuts-level locational marginal prices
    fname = scenario_path + '/LMP_nuts.csv'
    LMP_nuts.to_csv(fname)


def save_case_LMP(scenario, scenario_path, model, results):
    
    """
    Add LMP run info to case.csv file
    """

    case_path = scenario_path + 'case.csv'
    case = pd.read_csv(case_path, index_col=0)
    case.loc[scenario, 'Cost LMP run (BEUR)'] = round(model.cost(), 3)
    case.loc[scenario, 'termination LMP run'] = results.solver.termination_condition.value
    case.loc[scenario, 'status LMP run'] = results.solver.status.value
    case.to_csv(case_path)

def run_LMP(scenario):

    """
    Run the LMP model
    """

    # Define path
    scenario_path = opts_LMP['output_path'] + scenario + '/'

    # Load pre-solved results
    network = read_case_LMP(scenario_path)

    # Initialize LMP model instance
    start = time.time()
    
    model = init_model_LMP(network)
    
    end = time.time()
    print('Time to initialize model: {} seconds'.format(round((end - start))))

    # Solve LMP model instance
    model,results = solve_model_LMP(model, opts_LMP['solver_name'], opts_LMP['solv_opts'])

    # Get LMPs at bus-level
    LMP_bus = read_LMP_bus(model, network)

    # Get LMPs at regional-level
    LMP_nuts = read_LMP_nuts(opts_LMP['input_path'], LMP_bus, network)

    # Save LMPs to scenario path
    save_LMP(scenario_path, LMP_bus, LMP_nuts)

    # Save LMP run costs and termination condition to case file
    save_case_LMP(scenario, scenario_path, model, results)

