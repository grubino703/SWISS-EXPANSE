"""
Copyright 2022 Jan-Philipp Sasse (UNIGE), Evelina Trutnevyte (UNIGE)
~~~~~~~~~~~~~
Initialize Pyomo model

Inputs: network
Outputs: model instance
"""

# Import required packages
import pandas as pd
import numpy as np
import networkx as nx
import time
from pyomo.environ import *
from EXPANSE.read_settings import *
from EXPANSE.read_infrastructure import *

opts = read_settings()

# H2 Addition
# select additional variables depending on storyline
H2_assumptions = opts['hydrogen_assumptions']
H2_assumptions = H2_assumptions[opts['h2_scenario']]
# adapted to hand over storyline name to load correct files
network = read_infrastructure(opts['input_path'], opts['resolution'], opts['h2_scenario'])

def create_impact_matrix():
    
    '''
    Creates the impact matrix from the costs.csv file

    Input: 
    Output: impacts matrix
    '''

    lst = list(network['generators'].carrier.unique())+list(network['storage_units'].carrier.unique())
    impacts = pd.DataFrame(index = lst, 
                           columns=['job/MW','tCO2/MWhel','kgPM10/MWhel','m2/MW'])
    
    for tech in impacts.index:
        for impact in impacts.columns:
            
            try:
                cur_val = opts['costs'].loc[(opts['costs'].unit==impact),'value'][tech].values[0]
                impacts.loc[tech,impact] = cur_val
            
            except:
                impacts.loc[tech,impact] = 0.0

    impacts.loc['transmission','job/TWkm'] = opts['costs'].loc[(opts['costs'].unit=='job/TWkm'),'value'].values[0]
    impacts.index.name='technology'

    return impacts

def init_model():
    
    '''
    Builds and returns the Pyomo model instance

    Input: 
    Output: Pyomo model instance
    '''
    
    # Set timer
    init_var_start = time.time()

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
    Snapshots = list(network['loads_p_set_LowRes'].index[:int(opts['hours']/opts['resolution'])])
    
    
    # Create low resolution version of generators_p_max_pu
    network['generators_p_max_pu_LowRes'] = network['generators_p_max_pu'].iloc[::opts['resolution'], :].copy()
    network['generators_p_max_pu_LowRes'] = network['generators_p_max_pu_LowRes'].iloc[:int(opts['hours']/opts['resolution'])]

    # Create low resolution version of storage inflow
    network['storage_inflow_LowRes'] = network['storage_inflow'].iloc[::opts['resolution'], :].copy()
    network['storage_inflow_LowRes'] = network['storage_inflow_LowRes'].iloc[:int(opts['hours']/opts['resolution'])]

    # Storage units with inflow
    Inflow_IDs = list(network['storage_inflow'].columns)

    # Fixed storage units
    Fixed_storage_IDs = list(network['storage_units'][network['storage_units'].p_nom_extendable==False].index)

    # Nodes of system
    nodes = list(network['buses'].index)


    # H2 Addition
    # H2 Electrolysers, storage, fuel cells at nodes, transport paths
    Electrolysers_IDs = list(network['electrolysers'].index)
    H2_fuel_cell_IDs = list(network['fuel_cells'].index)
    H2_storage_IDs = list(network['H2_storage_units'].index)
    H2_transport_pipelines_IDs  = list(network['H2_transport_pipelines'].index)
    H2_transport_trucks_IDs  = list(network['H2_transport_trucks'].index)
    H2_buffer_IDs = list(network['H2_buffer'].index)
    H2_compressor_IDs = list(network['H2_compressor'].index)


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
        if network['generators'].loc[c,'ramp_limit'] * opts['resolution'] < 1:
            inflex_GenIDs.append(c)

    ### Create all the required dictionaries to speed up model
    generators_dict = network['generators'].to_dict()
    electrolysers_dict = network['electrolysers'].to_dict()
    fuel_cells_dict = network['fuel_cells'].to_dict()
    storage_units_dict = network['storage_units'].to_dict()
    H2_storage_units_dict = network['H2_storage_units'].to_dict()
    lines_dict = network['lines'].to_dict()
    links_dict = network['links'].to_dict()
    g_p_max_pu_dict = network['generators_p_max_pu_LowRes'].to_dict()
    inflow_dict = network['storage_inflow_LowRes'].to_dict()
    loads_dict = network['loads_p_set_LowRes'].iloc[:int(opts['hours']/opts['resolution'])].to_dict()

    # H2 Addition
    H2_demand_dict = network['loads_H2_set_LowRes']#.iloc[:int(opts['hours']/opts['resolution'])].to_dict()
    H2_byproduct_dict = network['byproduct_H2_LowRes']#.iloc[:int(opts['hours']/opts['resolution'])].to_dict()
    H2_liquefiers_loads_dict = network['loads_H2_liquefier_set_LowRes']
    H2_transport_pipelines_dict = network['H2_transport_pipelines'].to_dict()
    H2_transport_trucks_dict = network['H2_transport_trucks'].to_dict()
    H2_buffer_dict = network['H2_buffer'].to_dict()
    H2_compressor_dict = network["H2_compressor"].to_dict()

    

    
    ### Create functions for installed capacity bounds
    # Generator capacity bounds
    def p_nom_bounds(model, ID):
        return (generators_dict['p_nom_min'][ID],
                generators_dict['p_nom_max'][ID])
    
    # H2 Addition
    # Electrolyser capacity bounds (MW H2)
    def electrolyser_nom_bounds(model, ID):
        return (electrolysers_dict['p_nom_min'][ID],
                electrolysers_dict['p_nom_max'][ID])
    
    # H2 Addition
    # Fuel cell capacity bounds (MW el)
    def fuel_cell_nom_bounds(model, ID):
        return (fuel_cells_dict['p_nom_min'][ID],
                fuel_cells_dict['p_nom_max'][ID])

    # Storage capacity bounds
    def p_s_nom_bounds(model, ID):
        return (storage_units_dict['p_nom_min'][ID],
                storage_units_dict['p_nom_max'][ID])
    
    # H2 Addition
    # Hydrogen Storage capacity bounds (MW H2)
    def H2_p_s_nom_bounds(model, ID):
        return (H2_storage_units_dict['p_nom_min'][ID],
                H2_storage_units_dict['p_nom_max'][ID])

    # Line capacity bounds
    def lines_p_nom_bounds(model, ID):
        return (lines_dict['p_nom_min'][ID],
                lines_dict['p_nom_max'][ID])

    # Link capacity bounds
    def links_p_nom_bounds(model, ID):
        return (links_dict['p_nom_min'][ID],
                links_dict['p_nom_max'][ID])
    # H2 Addition
    # H2 transport via pipelines capacity bounds (MW H2)
    def H2_transport_pipelines_p_nom_bounds(model, ID):
        return (H2_transport_pipelines_dict['p_nom_min'][ID], 
                H2_transport_pipelines_dict['p_nom_max'][ID])
    
    # H2 Addition
    # H2 buffer bounds (MWh H2)
    def H2_buffer_bounds(model, ID, t):
        return (0, H2_buffer_dict["p_max"][ID])

    # H2 Addition
    # formerly needed currently not required because solved in constraint
    def H2_trucks_bounds(model, ID, t):
        return (0, H2_transport_trucks_dict["p_max"][ID])

    # H2 Addition 
    # currently set to very high (no limit) because indirectly defined by other technologies such as H2 production and transport
    # (MW el)
    def H2_compressor_bounds(model, ID):
        return (0, H2_compressor_dict["p_nom_max"][ID])

    ### Create variables
    # Generator power output
    model.p = Var(Gen_IDs, Snapshots, domain = Reals, initialize = 0.0)

    # Generator installed capacity
    model.p_nom = Var(Gen_IDs, domain = NonNegativeReals, 
                      initialize = dict(network['generators']['p_nom_min']),
                      bounds = p_nom_bounds)

    # H2 Addition
    # Electrolyser H2 output (MWh H2)
    model.p_H2_electrolyser = Var(Electrolysers_IDs, Snapshots, domain=NonNegativeReals, initialize=0.0)

    # Electrolyser installed capacity (MW H2)
    model.p_H2_electrolyser_nom = Var(Electrolysers_IDs, domain=NonNegativeReals, initialize = dict(network['electrolysers']['p_nom_min']),
                      bounds = electrolyser_nom_bounds)

    # Fuel cells electricity output (MWh el)
    model.p_H2_fuel_cell = Var(H2_fuel_cell_IDs, Snapshots, domain=NonNegativeReals, initialize=0.0)

    # Fuel cells installed capacity (MW el)
    model.p_H2_fuel_cell_nom = Var(H2_fuel_cell_IDs, domain=NonNegativeReals, initialize = dict(network['fuel_cells']['p_nom_min']),
                      bounds = fuel_cell_nom_bounds)


    # Storage dispatch
    model.dispatch = Var(Sto_IDs, Snapshots, domain = NonNegativeReals, initialize = 0.0)

    # Storage store
    model.store = Var(Sto_IDs, Snapshots, domain = NonNegativeReals, initialize = 0.0)

    # Storage spillage
    model.spillage = Var(Inflow_IDs, Snapshots, domain = NonNegativeReals, initialize = 0.0)

    # Storage installed capacity
    model.p_s_nom = Var(Sto_IDs, domain = NonNegativeReals, 
                        initialize = dict(network['storage_units']['p_nom_min']),
                        bounds = p_s_nom_bounds)

    # Storage state of charge
    model.soc = Var(Sto_IDs, Snapshots, domain = NonNegativeReals, initialize = 0.0)

    # H2 Addition
    # Hydrogen Storage dispatch (MWh H2)
    model.H2_dispatch = Var(H2_storage_IDs, Snapshots, domain = NonNegativeReals, initialize = 0.0)

    # Hydrogen Storage store (MWh H2)
    model.H2_store = Var(H2_storage_IDs, Snapshots, domain = NonNegativeReals, initialize = 0.0)

    # Hydrogen Storage installed capacity (MW H2)
    model.H2_p_s_nom = Var(H2_storage_IDs, domain = NonNegativeReals, 
                           initialize = dict(network['H2_storage_units']['p_nom_min']),
                           bounds = H2_p_s_nom_bounds)

    # Hydrogen Storage state of charge (MWh H2)
    model.H2_soc = Var(H2_storage_IDs, Snapshots, domain = NonNegativeReals, initialize = 0.0)


    # Line installed capacity
    model.lines_p_nom = Var(Line_IDs, domain = NonNegativeReals, 
                        initialize = dict(network['lines']['p_nom_min']), 
                        bounds = lines_p_nom_bounds
                        ) 

    # Line power flow from bus0 to bus1
    model.lines_p_p0 = Var(Line_IDs, Snapshots, domain = NonNegativeReals, initialize = 0.0)

    # Line power flow from bus1 to bus0
    model.lines_p_p1 = Var(Line_IDs, Snapshots, domain = NonNegativeReals, initialize = 0.0)

    # Link installed capacity
    model.links_p_nom = Var(Link_IDs, domain = NonNegativeReals, 
                        initialize = dict(network['links']['p_nom_min']), 
                        bounds = links_p_nom_bounds)

    # Link power flow from bus0 to bus1
    model.links_p_p0 = Var(Link_IDs, Snapshots, domain = NonNegativeReals, initialize = 0.0)

    # Link power flow from bus1 to bus0
    model.links_p_p1 = Var(Link_IDs, Snapshots, domain = NonNegativeReals, initialize = 0.0)

    # H2 transport via pipelines installed capacity (MW H2)
    model.H2_transport_pipelines_p_nom = Var(H2_transport_pipelines_IDs, domain = NonNegativeReals, 
                        initialize = 0, 
                        bounds = H2_transport_pipelines_p_nom_bounds)
    

    # H2 pipeline power flow from bus0 to bus1 (only one direction as compared to electricity modelling as we represent both pathways separately as they can have different sizes)
    # MWh H2
    model.H2_transport_pipelines_p_p0 = Var(H2_transport_pipelines_IDs, Snapshots, domain = NonNegativeReals, initialize = 0.0)

    # H2 buffer imports from outside of our five countries (only into FR, IT, AT, DE) (MWh H2)
    model.H2_buffer_p = Var(H2_buffer_IDs, Snapshots, domain = NonNegativeReals, initialize=0.0, bounds = H2_buffer_bounds)


    # Truck model based on He et al, 2021, Hydrogen Supply Chain Planning With Flexible Transmission and Storage Scheduling and adapted
    # totally invested trucks (unit = number of trucks) limited to a maximum number, could be increased in future
    model.invested_trucks = Var(within = NonNegativeReals, initialize = 0.0, bounds=(0, H2_assumptions["transport_capacity_overall_max"]))

    # currently (at timestep) empty trucks total (number of trucks)
    model.trucks_empty = Var(Snapshots, domain = NonNegativeReals, initialize = 0.0)
    # currently (at timestep) full trucks total (number of trucks)
    model.trucks_full = Var(Snapshots, domain = NonNegativeReals, initialize = 0.0)
    # trucks driving between two nodes that are full per timestep and route (number of trucks)
    model.trucks_full_in_transit = Var(H2_transport_trucks_IDs, Snapshots, domain = NonNegativeReals, initialize=0.0)
    # trucks staying at one node that are full per timestep and node (number of trucks), an upper limit could be defined next to avoid having more trucks parking than realistically possible
    model.trucks_full_staying = Var(nodes, Snapshots,  domain = NonNegativeReals, initialize=0.0)
    # trucks driving between two nodes that are empty per timestep and route (number of trucks)
    model.trucks_empty_in_transit = Var(H2_transport_trucks_IDs, Snapshots, domain = NonNegativeReals, initialize=0.0)
    # trucks staying at one node that are empty per timestep and node (number of trucks), an upper limit could be defined next to avoid having more trucks parking than realistically possible
    model.trucks_empty_staying = Var(nodes, Snapshots,  domain = NonNegativeReals, initialize=0.0)
    # helper variable indicating trucks that are charged (number of trucks)
    model.trucks_charged = Var(nodes, Snapshots,  domain = NonNegativeReals, initialize=0.0)
    # helper variable indicating trucks that are discharged (number of trucks)
    model.trucks_discharged = Var(nodes, Snapshots,  domain = NonNegativeReals, initialize=0.0)
    # helper variable indicating full trucks leaving a node (number of trucks)
    model.trucks_departing_full = Var(H2_transport_trucks_IDs, Snapshots, domain = NonNegativeReals, initialize=0.0)
    # number arriving trucks in our model are equal to the number of trucks that departed on that route X hours earlier (assumption: as they cannot stop somewhere in between),
    # therefore we do not need the additional variable
    # model.trucks_arriving_full = Var(H2_transport_trucks_IDs, Snapshots, domain = NonNegativeReals, initialize=0.0)
    
    # helper variable indicating empty trucks leaving a node (number of trucks)
    model.trucks_departing_empty = Var(H2_transport_trucks_IDs, Snapshots, domain = NonNegativeReals, initialize=0.0)
    # number arriving trucks in our model are equal to the number of trucks that departed on that route X hours earlier (assumption: as they cannot stop somewhere in between),
    # therefore we do not need the additional variable
    # #model.trucks_arriving_empty = Var(H2_transport_trucks_IDs, Snapshots, domain = NonNegativeReals, initialize=0.0)

    
    
    # H2 compressor power defined by electricity therefore in MW el
    model.H2_compressor_p_nom = Var(H2_compressor_IDs, domain=NonNegativeReals, initialize = 0, bounds = H2_compressor_bounds)
    # H2 compressor amount of compressed and decompressed hydrogen in MWh H2
    model.H2_compressed_p= Var(H2_compressor_IDs, Snapshots, domain=NonNegativeReals, initialize = 0)
    model.H2_decompressed_p= Var(H2_compressor_IDs, Snapshots, domain=NonNegativeReals, initialize = 0)


    init_var_end = time.time()
    print('---- Time needed to initalized following items ----')
    print('All variables: {} seconds'.format(round((init_var_end-init_var_start))))

    ### Line/link power flow constraint: power flow < capacity
    def lines_p_p0_max_rule(model,l,t):
        return(model.lines_p_p0[l,t] <= lines_dict['p_max_pu'][l] * model.lines_p_nom[l])

    def lines_p_p1_max_rule(model,l,t):
        return(model.lines_p_p1[l,t] <= lines_dict['p_max_pu'][l] * model.lines_p_nom[l])

    def links_p_p0_max_rule(model,l,t):
        return(model.links_p_p0[l,t] <= links_dict['p_max_pu'][l] * model.links_p_nom[l])

    def links_p_p1_max_rule(model,l,t):
        return(model.links_p_p1[l,t] <= links_dict['p_max_pu'][l] * model.links_p_nom[l])

    lines_p_max_start = time.time()
    
    model.lines_p_p0_max = Constraint(Line_IDs, Snapshots, rule = lines_p_p0_max_rule)
    model.lines_p_p1_max = Constraint(Line_IDs, Snapshots, rule = lines_p_p1_max_rule)
    model.links_p_p0_max = Constraint(Link_IDs, Snapshots, rule = links_p_p0_max_rule)
    model.links_p_p1_max = Constraint(Link_IDs, Snapshots, rule = links_p_p1_max_rule)
    
    lines_p_max_end = time.time()
    print('Line maximum capacity constraints: {} seconds'.format(round((lines_p_max_end-lines_p_max_start))))

    # H2 Addition
    ### Hydrogen transport flow constraint pipelines: H2 power flow < capacity
    # the activated was a test to limit pipelines to above a certain power (more realistically)
    # which would mean that a pipeline once it is activated also needs to be above a certain MW
    # however this turns the problem into a mixed integer model which was taking to long to compute
    def H2_transport_pipelines_p_p0_max_rule(model,l,t):
        return(model.H2_transport_pipelines_p_p0[l,t] <= H2_transport_pipelines_dict['p_max_pu'][l] * model.H2_transport_pipelines_p_nom[l]) #* model.H2_transport_activated[l])
        
        
    H2_transport_p_max_start = time.time()
    model.H2_transport_pipelines_p_p0_max = Constraint(H2_transport_pipelines_IDs, Snapshots, rule = H2_transport_pipelines_p_p0_max_rule)

    H2_transport_p_max_end = time.time()
    print('H2 transport minimum and maximum capacity constraints: {} seconds'.format(round((H2_transport_p_max_end-H2_transport_p_max_start))))


    ### Generator maximum power output constraint
    def p_max_rule(model,c,t):
        
        # Variable generator: p(t) < p_nom * CF_max(t)
        if c.split()[-1] in variable_techs:
            expr = model.p[c,t] <= g_p_max_pu_dict[c][t] * model.p_nom[c]
        
        # Flexible generator: p(t) < p_nom
        else:
            expr = model.p[c,t] <= generators_dict['p_max_pu'][c] * model.p_nom[c]
        
        return(expr)

    p_max_start = time.time()
    
    model.p_max = Constraint(Gen_IDs, Snapshots, rule = p_max_rule)
    
    p_max_end = time.time()
    print('Generator maximum power output constraints: {} seconds'.format(round((p_max_end-p_max_start))))

    ### Generator minimum power output constraint
    def p_min_rule(model,c,t):
        return(model.p[c,t] >= generators_dict['p_min_pu'][c] * model.p_nom[c])

    p_min_start = time.time()
    
    model.p_min = Constraint(Gen_IDs, Snapshots, rule = p_min_rule)
    
    p_min_end = time.time()
    print('Generator minimum power output constraints: {} seconds'.format(round((p_min_end-p_min_start))))

    # H2 Addition
    ### Electrolyser maximum power constraint, analogue to electricity modelling
    def electrolyser_max_rule(model,e, t):  
        return(model.p_H2_electrolyser[e,t] <=  electrolysers_dict['p_max_pu'][e] * model.p_H2_electrolyser_nom[e])
    
    e_max_start = time.time()
    model.electrolyser_max = Constraint(Electrolysers_IDs, Snapshots, rule = electrolyser_max_rule)
    e_max_end = time.time()

    print('Electrolyser maximum production constraints: {} seconds'.format(round((e_max_end-e_max_start))))

    ### Electrolyser minimum power constraint
    def electrolyser_min_rule(model,e, t):  
        return(model.p_H2_electrolyser[e,t] >=  electrolysers_dict['p_min_pu'][e] * model.p_H2_electrolyser_nom[e])
    
    e_min_start = time.time()
    model.electrolyser_min = Constraint(Electrolysers_IDs, Snapshots, rule = electrolyser_min_rule)
    e_min_end = time.time()

    print('Electrolyser minimum production constraints: {} seconds'.format(round((e_min_end-e_min_start))))

    ### Fuel cell maximum power constraint
    def fuel_cell_max_rule(model,fc, t):
            return(model.p_H2_fuel_cell[fc,t] <= fuel_cells_dict['p_max_pu'][fc] * model.p_H2_fuel_cell_nom[fc])
    
    ### Fuel cell minimum power constraint
    def fuel_cell_min_rule(model,fc, t):
            return(model.p_H2_fuel_cell[fc,t] >= fuel_cells_dict['p_min_pu'][fc] * model.p_H2_fuel_cell_nom[fc])


    
    fc_max_start = time.time()
    model.fuel_cell_max = Constraint(H2_fuel_cell_IDs, Snapshots, rule = fuel_cell_max_rule)
    model.fuel_cell_min = Constraint(H2_fuel_cell_IDs, Snapshots, rule = fuel_cell_min_rule)
    fc_max_end = time.time()

    print('Fuel cell maximum and minimum production constraints: {} seconds'.format(round((fc_max_end-fc_max_start))))

    ### Compressor maximum power constraint, here the compressed hydrogen energy is used to calculate the electrical power needed which is used to determine the costs of the compressor
    def compressor_max_rule(model, c, t):
        return(model.H2_compressed_p[c,t] *  H2_compressor_dict["electricity_demand"][c] <= model.H2_compressor_p_nom[c])
    
    # as far as I could find, hydrogen decompression can be done mechanically (without additional electricity demand)
    # therefore, for now the decompression amount of hydrogen is completely independent and we do not need the code snippet below
    #def decompressor_max_rule(model, c, t):
    #    return(model.H2_decompressed_p[c,t] <= model.H2_compressor_p_nom[c])

    model.compressor_max = Constraint(H2_compressor_IDs, Snapshots, rule = compressor_max_rule)

    ### Add constraint for reserve capacity margin
    # Count the total flexible generation capacity
    def flex_gen_cap(model):
        
        # Initialize expression
        expr = 0
        
        # Iterate through all flexible generators and add capacity
        for c in Gen_IDs:
            if c.split()[-1] in flex_techs:
                expr += model.p_nom[c]    

        # H2 Addition: we assume that fuel cells also contribute to the reserve margin although in theory the hydrogen storages could be empty
        for c in H2_fuel_cell_IDs:
            expr += model.p_H2_fuel_cell_nom[c]
        
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
                
        # H2 Addition: we assume that fuel cells also contribute to the reserve margin although in theory the hydrogen storages could be empty
        for c in H2_fuel_cell_IDs:
                expr += model.p_H2_fuel_cell[c,t]
        
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
        Reserve_Cap_Requirement = opts['r_cap'] * network['loads_p_set'].sum(axis=1).max()
        
        return(Flex_Cap_Unused >= Reserve_Cap_Requirement)

    reserve_capacity_start = time.time()
    
    model.reserve_capacity = Constraint(Snapshots, rule = reserve_capacity_rule)
    
    reserve_capacity_end = time.time()
    print('Reserve capacity constraints: {} seconds'.format(round((reserve_capacity_end-reserve_capacity_start))))



    ### Storage dispatch constraint
    # Constraint for storage dispatch: h(c,t) < h_nom(c)
    def h_max_rule(model,c,t):
        return(model.dispatch[c,t] <= storage_units_dict['p_max_pu'][c] * model.p_s_nom[c])

    h_max_start = time.time()
    
    model.h_max = Constraint(Sto_IDs, Snapshots, rule = h_max_rule)
    
    h_max_end = time.time()
    print('Storage dispatch constraints: {} seconds'.format(round((h_max_end-h_max_start))))

    ### Storage charging constraint
    # Constraint for storage charging: f(c,t) < h_nom(c)
    def f_max_rule(model,c,t):
        return(model.store[c,t] <= - storage_units_dict['p_min_pu'][c] * model.p_s_nom[c])

    f_max_start = time.time()
    
    model.f_max = Constraint(Sto_IDs, Snapshots, rule = f_max_rule)
    
    f_max_end = time.time()
    print('Storage charging constraints: {} seconds'.format(round((f_max_end-f_max_start))))

    ### Maximum state of charge constraint
    # Constraint for max. storage state of charge: E < P * h (discharge hours)
    def soc_max_rule(model,c,t):
        return(model.soc[c,t] <= storage_units_dict['max_hours'][c] * model.p_s_nom[c])

    soc_max_start = time.time()
    
    model.soc_max = Constraint(Sto_IDs, Snapshots, rule = soc_max_rule)
    
    soc_max_end = time.time()
    print('Storage maximum state of charge constraints: {} seconds'.format(round((soc_max_end-soc_max_start))))
    
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
        expr += storage_units_dict['efficiency_store'][c] * opts['resolution'] * model.store[c,t]
        expr -= 1 / storage_units_dict['efficiency_dispatch'][c] * opts['resolution'] * model.dispatch[c,t]
        
        if c in Inflow_IDs:
            expr += model.p_s_nom[c] * inflow_dict[c][t] * opts['resolution']
            expr -= model.spillage[c,t] * opts['resolution']
        
        return(expr == model.soc[c,t])

    soc_cons_start = time.time()
    
    model.soc_cons = Constraint(Sto_IDs, Snapshots, rule = soc_cons_rule)
    
    soc_cons_end = time.time()
    print('Storage hourly state of charge constraints: {} seconds'.format(round((soc_cons_end-soc_cons_start))))

    

    # Hydrogen Addition
    
    ### Hydrogen Storage dispatch constraint, analogue to the electricity model just for hydrogen
    # Constraint for storage dispatch: h(c,t) < h_nom(c)
    def H2_h_max_rule(model,c,t):
        return(model.H2_dispatch[c,t] <= H2_storage_units_dict['p_max_pu'][c] * model.H2_p_s_nom[c])

    h_max_start = time.time()
    
    model.H2_h_max = Constraint(H2_storage_IDs, Snapshots, rule = H2_h_max_rule)
    
    h_max_end = time.time()
    print('Hydrogen Storage dispatch constraints: {} seconds'.format(round((h_max_end-h_max_start))))

    ### Storage charging constraint
    # Constraint for storage charging: f(c,t) < h_nom(c)
    def H2_f_max_rule(model,c,t):
        return(model.H2_store[c,t] <= - H2_storage_units_dict['p_min_pu'][c] * model.H2_p_s_nom[c])

    f_max_start = time.time()
    
    model.H2_f_max = Constraint(H2_storage_IDs, Snapshots, rule = H2_f_max_rule)
    
    f_max_end = time.time()
    print('H2 Storage charging constraints: {} seconds'.format(round((f_max_end-f_max_start))))

    ### Hydrogen Maximum state of charge constraint
    # Constraint for max. storage state of charge: E < P * h (discharge hours)
    def H2_soc_max_rule(model,c,t):
        return(model.H2_soc[c,t] <= H2_storage_units_dict['max_hours'][c] * model.H2_p_s_nom[c])
    
    
    ### Hydrogen Minimum state of charge constraint
    # This constraint is currently not added!!!
    # The idea was that, as we use hydrogen also in fuel cells and former natural gas plants in the electricity reserve capacity,
    # in theory we could run out of hydrogen as we use it by different processes when we would have an emergency requiring exactly this reserve
    # Therefore, one could enforce a minimum reserve of hydrogen in each storage to avoid this
    # as storage in Switzerland was neglegible I did not investigate this further but one could add it in the future
    # Reserve capacity hydrogen if a storage exists, XX % must remain in storage for reserve capacity
    def H2_reserve_capacity_rule(model,c,t):
            return (model.H2_soc[c,t] >= model.H2_p_s_nom[c] * H2_storage_units_dict['max_hours'][c]*H2_assumptions["H2_reserve_capacity"])

    soc_max_start = time.time()
    
    model.H2_soc_max = Constraint(H2_storage_IDs, Snapshots, rule = H2_soc_max_rule)
    model.H2_soc_min = Constraint(H2_storage_IDs, Snapshots, rule = H2_reserve_capacity_rule)
    
    soc_max_end = time.time()
    print('Storage maximum and minimum state of charge constraints: {} seconds'.format(round((soc_max_end-soc_max_start))))
    
   
    ### Hydrogen Hourly state of charge constraint
    # Relates the SOC of previous hours to current hour
    # soc(c,t) = soc(c,t-1) + n(store,c) * f(c,t) - 1/n(dispatch,c) * h(c,t) + inflow(c,t) - spillage(c,t) 
    def H2_soc_cons_rule(model,c,t):
        try:
            t_1 = Snapshots[Snapshots.index(t)-1]
        except:
            t_1 = Snapshots[-1]
        
        expr = 0
        expr += model.H2_soc[c,t_1]
        expr += H2_storage_units_dict['efficiency_store'][c] * opts['resolution'] * model.H2_store[c,t]
        expr -= (1 / H2_storage_units_dict['efficiency_dispatch'][c]) * opts['resolution'] * model.H2_dispatch[c,t]
        
        return(expr == model.H2_soc[c,t])

    soc_cons_start = time.time()
    
    model.H2_soc_cons = Constraint(H2_storage_IDs, Snapshots, rule = H2_soc_cons_rule)
    
    soc_cons_end = time.time()
    print('Storage hourly state of charge constraints: {} seconds'.format(round((soc_cons_end-soc_cons_start))))

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
            #if network['links'].loc[l,'bus1'] == n:
            if links_dict['bus1'][l] == n:
                flow_sum_expr += links_dict['efficiency'][l] * model.links_p_p0[l,t]
                flow_sum_expr -= model.links_p_p1[l,t]
            
        return flow_sum_expr
    
    # Hydrogen Addition
    # Sum of h2 conversion of selected node and snapshot
    def H2_sum(model,n,t):
        H2_sum_expr = 0

        for e in Electrolysers_IDs:
            if electrolysers_dict['bus'][e] == n:
                H2_sum_expr -= model.p_H2_electrolyser[e,t] * (1/H2_assumptions["electrolyser_efficiency"]) # Electrolysis, efficiency here as the p_H2_electrolyser variable is in MWh H2
                
        for fc in H2_fuel_cell_IDs:
            if fuel_cells_dict['bus'][fc] == n:
                H2_sum_expr += model.p_H2_fuel_cell[fc,t]  # Fuel cell to electricity, already in MWh el

        return H2_sum_expr
    
    # Electricity demand for compression, depending on amount of hydrogen compressed
    # this is quite a rough estimation as it neglects temperature and other physical effects defining the electricity demand during compression
    # there might be some thermodynamic formulas giving better estimates to try in the future
    def H2_compressor_sum(model,n,t):
        H2_compressor_sum = 0
        
        for c in H2_compressor_IDs:
            if (H2_compressor_dict['bus'][c] == n):
                H2_compressor_sum += model.H2_compressed_p[c,t] *  H2_compressor_dict["electricity_demand"][c]

        return H2_compressor_sum


    # Hydrogen Addition
    # Create nodal power balances
    # added the hydrogen conversions, compressors and some additional liquefaction demand (the latter so far only for airports)
    def p_balance_rule(model,n,t):

        LHS = p_sum(model,n,t) + h_sum(model,n,t) - f_sum(model,n,t) + flow_sum(model,n,t) + H2_sum(model, n, t) - H2_compressor_sum(model, n,t)
        RHS = loads_dict[n][t] + H2_liquefiers_loads_dict[n][t]

        return(LHS == RHS)

    p_balance_start = time.time()
    
    model.p_balance = Constraint(nodes, Snapshots, rule = p_balance_rule)
    
    p_balance_end = time.time()
    print('Nodel power balance constraints: {} seconds'.format(round((p_balance_end-p_balance_start))))

    # H2 Addition
    # truck model, based on He et al, 2021 and adapted to match the rest of the model
    # limit maximum trucks on one route to avoid unrealistic traffic
    def max_trucks_on_one_route_per_hour_rule(model, l, t):
        return model.trucks_departing_full[l,t] + model.trucks_departing_empty[l,t] <= H2_transport_trucks_dict["p_max"][l]
    
    model.max_trucks_on_one_route_per_hour_rule = Constraint(H2_transport_trucks_IDs, Snapshots, rule = max_trucks_on_one_route_per_hour_rule)

    # night driving ban for trucks
    # so far (because of the resolution I used) I did not enforce a night driving ban
    # however in Switzerland (and the other countries) there are some limits on when trucks are allowed to drive (not at night/Sundays...)
    # therefore an additional night driving limit for the trucks transporting hydrogen could be added in the future
    '''def night_driving_ban_rule_full(model, l, t):
        if (int(t[11:13]) > 22) or (int(t[11:13])  < 5):
            return model.trucks_full_in_transit[l,t] == 0
        else:
            return model.trucks_full_in_transit[l,t]==model.trucks_full_in_transit[l,t]
    
    model.night_driving_ban_rule_full = Constraint(H2_transport_trucks_IDs, Snapshots, rule = night_driving_ban_rule_full)

    def night_driving_ban_rule_empty(model, l, t):
        if (int(t[11:13]) > 22) or (int(t[11:13])  < 5):
            return model.trucks_empty_in_transit[l,t] == 0
        else:
            return model.trucks_empty_in_transit[l,t]==model.trucks_empty_in_transit[l,t]
    
    model.night_driving_ban_rule_empty = Constraint(H2_transport_trucks_IDs, Snapshots, rule = night_driving_ban_rule_empty)'''
        
    
    # H2 Addition
    # the following constraints are explained in depth in the additional documentation file and the paper appendix
    # total number of trucks
    def truck_invested_rule(model, t):
        return model.invested_trucks == model.trucks_empty[t] + model.trucks_full[t]

    model.trucks_invested_rule = Constraint(Snapshots, rule = truck_invested_rule)
    
    # total number of trucks full
    def trucks_full_at_time_constraint(model, t):
        transit_trucks = 0
        staying_trucks = 0
        for l in H2_transport_trucks_IDs:
            transit_trucks += model.trucks_full_in_transit[l,t] 
        for n in nodes:
            staying_trucks += model.trucks_full_staying[n,t]
        return(model.trucks_full[t] == transit_trucks+staying_trucks)
    model.trucks_full_at_time_constraint = Constraint(Snapshots, rule=trucks_full_at_time_constraint)

    # total number of trucks empty
    def trucks_empty_at_time_constraint(model, t):
        transit_trucks = 0
        staying_trucks = 0
        for l in H2_transport_trucks_IDs:
            transit_trucks +=  model.trucks_empty_in_transit[l,t] 
        for n in nodes:
            staying_trucks += model.trucks_empty_staying[n,t]
        return(model.trucks_empty[t] == transit_trucks+staying_trucks)
    model.trucks_empty_at_time_constraint = Constraint(Snapshots, rule=trucks_empty_at_time_constraint)

    # helper function to match the driving duration to the resolution used (explained in additional documentation file)
    def closest_multiple(drive_duration): 
        if drive_duration == 0:
            return int(0)
        else: 
            multiple = opts['resolution']
            while drive_duration > multiple:
                multiple = multiple+opts['resolution']
            offset = multiple/opts['resolution']
            return int(offset)
        
    # change in full trucks (differs from original model from He, modelled similarly to a storage here)
    def change_of_full_trucks(model, node, t):
    
        try:
            t_1 = Snapshots[Snapshots.index(t)-1]
        except:
            t_1 = Snapshots[-1]

        net_arriving = 0
        for l in H2_transport_trucks_IDs:
            if H2_transport_trucks_dict['bus0'][l] == node:
                net_arriving -= model.trucks_departing_full[l, t] * opts['resolution']
            if H2_transport_trucks_dict['bus1'][l] == node:
                time_offset = closest_multiple(H2_transport_trucks_dict['transport_offset_hours'][l])
                try:
                    t_offset = Snapshots[Snapshots.index(t)-time_offset]
                except:
                    t_offset = Snapshots[-time_offset]
                net_arriving += model.trucks_departing_full[l, t_offset] * opts['resolution']

        return model.trucks_full_staying[node, t] == model.trucks_full_staying[node, t_1] + model.trucks_charged[node,t] * opts['resolution'] - model.trucks_discharged[node,t] * opts['resolution'] + net_arriving
        

    model.change_of_full_trucks = Constraint(nodes, Snapshots, rule = change_of_full_trucks)

    # change in empty trucks (differs from original model from He, modelled similarly to a storage here)
    def change_of_empty_trucks(model, node, t):
        try:
            t_1 = Snapshots[Snapshots.index(t)-1]
        except:
            t_1 = Snapshots[-1]

        net_arriving = 0
        for l in H2_transport_trucks_IDs:
            if H2_transport_trucks_dict['bus0'][l] == node:
                net_arriving -= model.trucks_departing_empty[l,t] * opts['resolution']
            if H2_transport_trucks_dict['bus1'][l] == node:
                time_offset = closest_multiple(H2_transport_trucks_dict['transport_offset_hours'][l])
                try:
                    t_offset = Snapshots[Snapshots.index(t)-time_offset]
                except:
                    t_offset = Snapshots[-time_offset]
                net_arriving += model.trucks_departing_empty[l, t_offset] * opts['resolution']

        return model.trucks_empty_staying[node, t] == model.trucks_empty_staying[node, t_1] - model.trucks_charged[node,t] * opts['resolution'] + model.trucks_discharged[node,t] * opts['resolution']+ net_arriving
        

    model.change_of_empty_trucks = Constraint(nodes, Snapshots, rule = change_of_empty_trucks)

    # change of trucks in transit (differs from original model from He, modelled similarly to a storage here))
    def change_of_full_trucks_in_transit_rule(model, l, t):
        try:
            t_1 = Snapshots[Snapshots.index(t)-1]
        except:
            t_1 = Snapshots[-1]
        
        time_offset = closest_multiple(H2_transport_trucks_dict['transport_offset_hours'][l])
        try:
            t_offset = Snapshots[Snapshots.index(t)-time_offset]
        except:
            t_offset = Snapshots[-time_offset]

        return model.trucks_full_in_transit[l, t] == model.trucks_full_in_transit[l, t_1] + model.trucks_departing_full[l, t] * opts['resolution'] - model.trucks_departing_full[l, t_offset] * opts['resolution']

    model.change_of_full_trucks_in_transit = Constraint(H2_transport_trucks_IDs, Snapshots, rule = change_of_full_trucks_in_transit_rule)

    def change_of_empty_trucks_in_transit_rule(model, l, t):
        try:
            t_1 = Snapshots[Snapshots.index(t)-1]
        except:
            t_1 = Snapshots[-1]
        
        time_offset = closest_multiple(H2_transport_trucks_dict['transport_offset_hours'][l])
        try:
            t_offset = Snapshots[Snapshots.index(t)-time_offset]
        except:
            t_offset = Snapshots[-time_offset]

        return model.trucks_empty_in_transit[l, t] == model.trucks_empty_in_transit[l, t_1]  + model.trucks_departing_empty[l, t] * opts['resolution'] - model.trucks_departing_empty[l, t_offset] * opts['resolution']

    model.change_of_empty_trucks_in_transit = Constraint(H2_transport_trucks_IDs, Snapshots, rule = change_of_empty_trucks_in_transit_rule)


    ## H2 balance per node and timestep
    # Sum of electrolyser output of selected node and snapshot
    def electrolyser_p_sum(model,n,t):
        electrolyser_p_sum_expr = 0
        for c in Electrolysers_IDs:
            if electrolysers_dict['bus'][c] == n:
                electrolyser_p_sum_expr += model.p_H2_electrolyser[c,t]
        return electrolyser_p_sum_expr
    
    # Sum of H2 storage dispatch of selected node and snapshot
    def H2_h_sum(model,n,t):
        H2_h_sum_expr = 0
        for c in H2_storage_IDs:
            if H2_storage_units_dict['bus'][c] == n:
                H2_h_sum_expr += model.H2_dispatch[c,t]
        return H2_h_sum_expr

    # Sum of storage charging of selected node and snapshot
    def H2_f_sum(model,n,t):
        H2_f_sum_expr = 0
        for c in H2_storage_IDs:
            if H2_storage_units_dict['bus'][c] == n:
                H2_f_sum_expr += model.H2_store[c,t]
        return H2_f_sum_expr
    
    # sum of electricity production from hydrogen of selected node and snapshot
    def H2_fuel_cell_gas_turbine_p_sum(model,n,t):
        H2_fuel_cell_gas_turbine_p_sum_expr = 0
        for c in H2_fuel_cell_IDs:
            if fuel_cells_dict['bus'][c] == n:
                H2_fuel_cell_gas_turbine_p_sum_expr += model.p_H2_fuel_cell[c,t] * (1/H2_assumptions["fuel_cell_efficiency"])
        # if natural gas plants now burn hydrogen or a mix, a possible further development could be to entirely move these hydrogen combustion plants
        # to a own dict (joined with the fuel cells) representing all reconversion plants
        # that way, one could add different technologies like combined cycle plants (see discussion in paper)
        if H2_assumptions['replace_nat_gas'] > 0:
            for c in Gen_IDs:
                if (generators_dict['bus'][c] == n) and (generators_dict['carrier'][c] == 'gas'):
                    H2_fuel_cell_gas_turbine_p_sum_expr += model.p[c,t] * H2_assumptions['replace_nat_gas'] * (1/H2_assumptions['efficiency_H2_gas_turbine'])

        return H2_fuel_cell_gas_turbine_p_sum_expr
    
    # Sum H2 transport that are connected to each node
    
    def H2_transport_pipelines_sum(model,n,t):
        
        H2_transport_pipelines_sum_expr = 0
        
        for l in H2_transport_pipelines_IDs:

            # if transport starts at the node
            if H2_transport_pipelines_dict['bus0'][l] == n:
                H2_transport_pipelines_sum_expr -= model.H2_transport_pipelines_p_p0[l,t] 
            
            # if transport ends at the node
            if H2_transport_pipelines_dict['bus1'][l] == n:
                H2_transport_pipelines_sum_expr += H2_transport_pipelines_dict['efficiency'][l] * model.H2_transport_pipelines_p_p0[l,t]

        return H2_transport_pipelines_sum_expr
    
    def H2_transport_trucks_sum(model,n,t):
        # old model without modelling trucks in detail
        '''H2_transport_trucks_sum_expr = 0
                
        for l in H2_transport_trucks_IDs:
                
                time_offset = closest_multiple(H2_transport_trucks_dict['transport_offset_hours'][l])
                try:
                    t_offset = Snapshots[Snapshots.index(t)-time_offset]
                except:
                    t_offset = Snapshots[-time_offset]
            
                # if transport starts at the node
                if H2_transport_trucks_dict['bus0'][l] == n:
                    #H2_transport_sum_expr += H2_transport_dict['efficiency'][l] * model.H2_transport_p_p1[l,t_offset]
                    H2_transport_trucks_sum_expr -= model.H2_transport_trucks_p_p0[l,t]
                
                # if transport ends at the node
                if H2_transport_trucks_dict['bus1'][l] == n:
                    H2_transport_trucks_sum_expr += H2_transport_trucks_dict['efficiency'][l] * model.H2_transport_trucks_p_p0[l,t_offset]
                    #H2_transport_sum_expr -= model.H2_transport_p_p1[l,t]'''
        
        # trucks discharged and charged multiplies by their transportation capacity (1100 kg of hydrogen - Reuß et al 2021) converted to MWh (1kg H2 = 33.33 kWh LHV)
        H2_transport_trucks_sum_expr =  (model.trucks_discharged[n,t] - model.trucks_charged[n,t])* 1100 * 33.33/1000
        return H2_transport_trucks_sum_expr
    
    # if hydrogen trucks are driving also powered by hydrogen then we need to take this into account here as an additional demand
    def H2_own_consumption_transport_trucks_sum(model, n, t):
        H2_transport_trucks_own_consumption_expr = 0

        for l in H2_transport_trucks_IDs:
            # if transport starts at the node
            if H2_transport_trucks_dict['bus0'][l] == n:
                #H2_transport_sum_expr += H2_transport_dict['efficiency'][l] * model.H2_transport_p_p1[l,t_offset]
                #H2_transport_trucks_own_consumption_expr += model.H2_transport_trucks_p_p0[l,t] *  H2_transport_trucks_dict['own_hydrogen_consumption'][l]
                H2_transport_trucks_own_consumption_expr += (model.trucks_departing_full[l, t] + model.trucks_departing_empty[l,t]) * 1100*33.33/1000 *  H2_transport_trucks_dict['own_hydrogen_consumption'][l]
        return H2_transport_trucks_own_consumption_expr

    # hydrogen buffer imports
    def H2_buffer_sum(model, n, t):
        H2_buffer_sum_expr = 0
        for c in H2_buffer_IDs:
            if H2_buffer_dict['bus'][c] == n:
                H2_buffer_sum_expr += model.H2_buffer_p[c,t]
        return H2_buffer_sum_expr
    
    # each pressure change is modelled by a different compressor eventhough a compressor would be able to change the pressure levels
    # the reason is the assumption that the compressors are most likely at different physical locations based on their use case (close to pipelines, truck stations etc) and therefore modelled independently
    # I modelled only three pressure levels to give an idea of the electricity demand but different storages/pipelines can have other pressures as well
    
    # hydrogen compressor exchange from or to 30 bar
    def H2_compressor_exchange_30(model,n,t):
        H2_compressor_expr = 0
        for c in H2_compressor_IDs:
            if (H2_compressor_dict['bus'][c] == n) and (H2_compressor_dict["pressure_a"][c] == 250) and (H2_compressor_dict["pressure_b"][c] == 30):
                H2_compressor_expr -= model.H2_compressed_p[c,t] 
                H2_compressor_expr += model.H2_decompressed_p[c,t]* (1-0.5/100)
            if (H2_compressor_dict['bus'][c] == n) and (H2_compressor_dict["pressure_a"][c] == 100) and (H2_compressor_dict["pressure_b"][c] == 30):
                H2_compressor_expr -= model.H2_compressed_p[c,t] 
                H2_compressor_expr += model.H2_decompressed_p[c,t] * (1-0.5/100)
        return H2_compressor_expr
    
    # hydrogen compressor exchange from or to 100 bar
    def H2_compressor_exchange_100(model,n,t):
        H2_compressor_expr = 0
        for c in H2_compressor_IDs:
            if (H2_compressor_dict['bus'][c] == n) and (H2_compressor_dict["pressure_a"][c] == 250) and (H2_compressor_dict["pressure_b"][c] == 100):
                H2_compressor_expr -= model.H2_compressed_p[c,t] 
                H2_compressor_expr += model.H2_decompressed_p[c,t] * (1-0.5/100)
            if (H2_compressor_dict['bus'][c] == n) and (H2_compressor_dict["pressure_a"][c] == 100) and (H2_compressor_dict["pressure_b"][c] == 30):
                H2_compressor_expr += model.H2_compressed_p[c,t] * (1-0.5/100)
                H2_compressor_expr -= model.H2_decompressed_p[c,t] 
        return H2_compressor_expr
    
    # hydrogen compressor exchange from or to 250 bar
    def H2_compressor_exchange_250(model,n,t):
        H2_compressor_expr = 0
        for c in H2_compressor_IDs:
            if (H2_compressor_dict['bus'][c] == n) and (H2_compressor_dict["pressure_a"][c] == 250) and (H2_compressor_dict["pressure_b"][c] == 100):
                H2_compressor_expr += model.H2_compressed_p[c,t] * (1-0.5/100)
                H2_compressor_expr -= model.H2_decompressed_p[c,t] 
            if (H2_compressor_dict['bus'][c] == n) and (H2_compressor_dict["pressure_a"][c] == 250) and (H2_compressor_dict["pressure_b"][c] == 30):
                H2_compressor_expr += model.H2_compressed_p[c,t] * (1-0.5/100)
                H2_compressor_expr -= model.H2_decompressed_p[c,t] 
        return H2_compressor_expr
    
    ### H2 balance constraint, see also additional documentation file and paper appendix
    # at 30 bar
    def H2_balance_rule_30(model, n, t):
        H2_supply = electrolyser_p_sum(model, n, t) + H2_compressor_exchange_30(model, n, t) - H2_fuel_cell_gas_turbine_p_sum(model, n, t) + H2_byproduct_dict[n][t]
        H2_demand = 0
        return(H2_demand == H2_supply)
    
    # at 100 bar
    def H2_balance_rule_100(model, n, t):
        H2_supply =  H2_compressor_exchange_100(model, n, t) + H2_transport_pipelines_sum(model, n, t) 
        H2_demand = 0
        return(H2_demand == H2_supply)
    
    # at 250 bar
    def H2_balance_rule_250(model, n, t):
        H2_supply =  H2_compressor_exchange_250(model, n, t) + H2_transport_trucks_sum(model, n, t)  + H2_h_sum(model, n, t) - H2_f_sum(model, n,t)+ H2_buffer_sum(model,n,t)
        H2_demand = H2_demand_dict[n][t] + H2_own_consumption_transport_trucks_sum(model, n, t)
        return(H2_demand == H2_supply)
    
    e_balance_start = time.time()
    model.H2_balance_30 = Constraint(nodes, Snapshots, rule = H2_balance_rule_30)
    model.H2_balance_100 = Constraint(nodes, Snapshots, rule = H2_balance_rule_100)
    model.H2_balance_250 = Constraint(nodes, Snapshots, rule = H2_balance_rule_250)
    
    e_balance_end = time.time()
    print('Nodal H2 balance constraints: {} seconds'.format(round((e_balance_end-e_balance_start))))

    ### Kirchhoff Voltage Law constraint
    cyc_cons_start = time.time()

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

    cyc_cons_end = time.time()
    print('Kirchoff voltage constraints: {} seconds'.format(round((cyc_cons_end-cyc_cons_start))))

    ### Add ramping constraints
    # Ramping up constraint: p(c,t) - p(c,t-1) <= p_nom(c) * ramp_limit(c) * resolution
    # for t in 1 -> end
    def ramp_up_rule(model,c,t):
        
        t_1 = Snapshots[Snapshots.index(t)-1]
        
        ramp_up_MW = model.p[c,t] - model.p[c,t_1]
        ramp_up_limit_MW = model.p_nom[c] * generators_dict['ramp_limit'][c] * opts['resolution']
        
        return(ramp_up_MW <= ramp_up_limit_MW)

    ramp_up_start = time.time()
    
    model.ramp_up = Constraint(inflex_GenIDs, Snapshots[1:], rule = ramp_up_rule)
    
    ramp_up_end = time.time()
    print('Ramping up constraints: {} seconds'.format(round((ramp_up_end-ramp_up_start))))

    # Ramping down constraint: p(c,t) - p(c,t-1) >= (-1) * p_nom(c) * ramp_limit(c) * resolution
    # for t in 1 -> end
    def ramp_down_rule(model,c,t):
        
        t_1 = Snapshots[Snapshots.index(t)-1]
        
        ramp_down_MW = model.p[c,t] - model.p[c,t_1]
        ramp_down_limit_MW = (-1) * model.p_nom[c] * generators_dict['ramp_limit'][c] * opts['resolution']
        
        return(ramp_down_MW >= ramp_down_limit_MW)

    ramp_down_start = time.time()
    
    model.ramp_down = Constraint(inflex_GenIDs, Snapshots[1:], rule = ramp_down_rule)
    
    ramp_down_end = time.time()
    print('Ramping up constraints: {} seconds'.format(round((ramp_down_end-ramp_down_start))))

    ### Objective function
    def gen_cap_cost(model):
        return sum(model.p_nom[c] * generators_dict['capital_cost'][c]
                   for c in Gen_IDs)

    def gen_op_cost(model):
        return sum(model.p[c, t] * generators_dict['marginal_cost'][c]
                   for c in Gen_IDs for t in Snapshots) * 8760 / len(Snapshots)
    
    # H2 addition
    def H2_cap_cost(model):
        cap_cost = sum(model.p_H2_electrolyser_nom[e] *(1/H2_assumptions["electrolyser_efficiency"]) * electrolysers_dict["capital_cost"][e]
                   for e in Electrolysers_IDs) # Electrolysers costs per MW el
        cap_cost += sum(model.H2_compressor_p_nom[e] * H2_compressor_dict["capital_cost"][e]
                   for e in H2_compressor_IDs) # Compressor
        cap_cost += H2_liquefiers_loads_dict.max().sum() * H2_assumptions["capital_cost_liquefaction"] # Liquefier
        cap_cost += sum(model.p_H2_fuel_cell_nom[fc] * fuel_cells_dict["capital_cost"][fc]
                   for fc in H2_fuel_cell_IDs) # Fuel cells
        cap_cost += sum(model.H2_p_s_nom[c] * H2_storage_units_dict['max_hours'][c] * H2_storage_units_dict['capital_cost'][c]
                   for c in H2_storage_IDs) # Storage
        cap_cost += sum(H2_transport_pipelines_dict['capital_cost'][l] * model.H2_transport_pipelines_p_nom[l] #* model.H2_transport_activated[l]
                   for l in H2_transport_pipelines_IDs)# Transport pipelines
        cap_cost += model.invested_trucks * H2_assumptions["investment_H2_transport_trucks"]

        return cap_cost
    
    def H2_op_cost(model):
        op_cost = sum(model.p_H2_electrolyser[e, t] * (1/H2_assumptions["electrolyser_efficiency"])*electrolysers_dict["marginal_cost"][e]
                   for e in Electrolysers_IDs for t in Snapshots) * 8760 / len(Snapshots) # Electrolysers
        op_cost += sum(model.p_H2_fuel_cell[fc,t] * fuel_cells_dict["marginal_cost"][fc]
                   for fc in H2_fuel_cell_IDs for t in Snapshots) * 8760 / len(Snapshots)# Fuel cells
        op_cost += sum(model.H2_dispatch[c, t] * H2_storage_units_dict['marginal_cost'][c]
                   for c in H2_storage_IDs for t in Snapshots) * 8760 / len(Snapshots) # Storage
        op_cost += sum(model.H2_transport_pipelines_p_p0[l,t] * H2_transport_pipelines_dict['marginal_cost'][l] 
                   for l in H2_transport_pipelines_IDs for t in Snapshots) * 8760 / len(Snapshots) # Transport pipelines
        op_cost += sum((model.trucks_departing_full[l,t] + model.trucks_departing_empty[l,t])* H2_transport_trucks_dict["marginal_cost"][l]
                       for l in H2_transport_trucks_IDs for t in Snapshots) *8760/len(Snapshots)
        op_cost += sum(model.H2_buffer_p[c,t] * H2_buffer_dict['marginal_cost'][c]
                       for c in H2_buffer_IDs for t in Snapshots) *8760/len(Snapshots) # Buffer imports
        return op_cost

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
        # Hydrogen Addition
        expr += H2_cap_cost(model) + H2_op_cost(model)
        
        return expr

    def GHG(model):
        expr = 0
        for c in Gen_IDs:
            for t in Snapshots:
                if network['generators']['carrier'][c] == "gas":
                    #H2 Addition: when hydrogen is burned in natural gas plants (see H2 assumptions file), we assume no CO2 emissions for the hydrogen part
                    expr += model.p[c,t] * (1-H2_assumptions['replace_nat_gas'])* impacts_dict['tCO2/MWhel'][c.split()[-1]] * 8760/len(Snapshots)
                else:
                    expr += model.p[c,t] * impacts_dict['tCO2/MWhel'][c.split()[-1]] * 8760/len(Snapshots)
                #expr += model.p[c,t] * impacts.loc[c.split()[-1],'tCO2/MWhel'] * 8760/len(Snapshots)
        return expr

    # Hydrogen network is not yet taken into account here!!!!!
    def PM10(model):
        expr = 0
        for c in Gen_IDs:
            for t in Snapshots:
                expr += model.p[c,t] * impacts_dict['kgPM10/MWhel'][c.split()[-1]] * 8760/len(Snapshots)
                #expr += model.p[c,t] * impacts.loc[c.split()[-1],'kgPM10/MWhel'] * 8760/len(Snapshots)
        return expr

    # Hydrogen network is not yet taken into account here!!!!!
    def Jobs(model):
        expr = 0
        
        # Jobs in electricity generation
        for c in Gen_IDs:
            expr += model.p_nom[c] * impacts_dict['job/MW'][c.split()[-1]]
            #expr += model.p_nom[c] * impacts.loc[c.split()[-1],'job/MW']
        
        # Jobs in storage
        for c in Sto_IDs:
            expr += model.p_s_nom[c] * impacts_dict['job/MW'][c.split()[-1]]
            #expr += model.p_s_nom[c] * impacts.loc[c.split()[-1],'job/MW']
        
        # Jobs in HVDC transmission (TWkm * job/TWkm)
        for l in Link_IDs:
            expr += model.links_p_nom[l] * links_dict['length'][l] /1e6 * impacts_dict['job/TWkm']['transmission']
            #expr += model.links_p_nom[l] * network['links'].loc[l,'length'] /1e6 * impacts.loc['transmission','job/TWkm']
        
        # Jobs in HVAC transmission (TWkm * job/TWkm)
        for l in Line_IDs:
            expr += model.lines_p_nom[l] * lines_dict['length'][l] /1e6 * impacts_dict['job/TWkm']['transmission']
            #expr += model.lines_p_nom[l] * network['lines'].loc[l,'length'] /1e6 * impacts.loc['transmission','job/TWkm']
        
        return expr

    # Hydrogen network is not yet taken into account here!!!!!
    def LandUse(model):
        expr = 0
        for c in Gen_IDs:
            expr += model.p_nom[c] * impacts_dict['m2/MW'][c.split()[-1]]
            #expr += model.p_nom[c] * impacts.loc[c.split()[-1],'m2/MW']
        for c in Sto_IDs:
            expr += model.p_s_nom[c] * impacts_dict['m2/MW'][c.split()[-1]]
            #expr += model.p_s_nom[c] * impacts.loc[c.split()[-1],'m2/MW']
        return expr

    # Assign model objectives
    impacts = create_impact_matrix()
    impacts_dict = impacts.to_dict()
    
    cost_start = time.time()
    model.cost = Objective(expr = tot_cost(model), sense = minimize)
    cost_end = time.time()
    print('Cost objective: {} seconds'.format(round((cost_end-cost_start))))

    model.ghg = Objective(expr = GHG(model), sense = minimize)
    model.pm10 = Objective(expr = PM10(model), sense = minimize)
    model.jobs = Objective(expr = Jobs(model), sense = maximize)
    model.landuse = Objective(expr = LandUse(model), sense = minimize)

    # Deactivate model objective
    model.cost.deactivate()
    model.ghg.deactivate()
    model.pm10.deactivate()
    model.jobs.deactivate()
    model.landuse.deactivate()

    return model
