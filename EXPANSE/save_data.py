"""
Copyright 2022 Jan-Philipp Sasse (UNIGE), Evelina Trutnevyte (UNIGE)
~~~~~~~~~~~~~
Save the Pyomo model results data

Inputs: Pyomo model instance
Outputs: CSV files
"""

# Import packages
import pandas as pd
import os
import glob
from pyomo.environ import *
from EXPANSE.read_settings import *
from EXPANSE.read_infrastructure import *

# Read settings
opts = read_settings()
network = read_infrastructure(opts['input_path'],opts['resolution'], opts['h2_scenario'])
sum_path = opts['output_path'] + 'Summary'

def save_all_cases(opts):
    
    '''
    Gets all scenario cases for the final overview plots
    '''

    scenarios = []
    for folder in glob.glob(opts['output_path'] + "M*"):
        scenarios.append(folder.split('/')[-1])
    scenarios.append('Current')

    cases = pd.DataFrame(index = scenarios)
    for row in cases.index:
        fname = opts['output_path'] + row + '/case.csv'
        case = pd.read_csv(fname,index_col=0)
        for col in case.columns:
            cases.loc[row,col] = case.loc[row,col]

    cases.index.name = 'Scenario'
    cases.to_csv(sum_path+'/cases.csv')
    
    return cases

def save_data(scenario,slack,results,model):
    
    '''
    Converts the Pyomo model instance into CSV files
    and the stores data in a designated scenario folder
    '''

    print("Storing scenario: %s" % scenario)
    
    # Create directory
    res_path = opts['output_path'] + scenario

    try:
        os.mkdir(res_path)
    except OSError:
        print("Creation of the directory %s failed" % res_path)
    else:
        print("Successfully created the directory %s " % res_path)

    # Get index lists
    Line_IDs = list(model.lines_p_nom_index.ordered_data())
    Link_IDs = list(model.links_p_nom_index.ordered_data())
    Sto_IDs = list(model.p_s_nom_index.ordered_data())
    Gen_IDs = list(model.p_nom_index.ordered_data())
    Snapshots = list(model.p_index_1.ordered_data())
    Spillage_IDs = list(model.spillage_index_0.ordered_data())
    nodes = list(model.p_balance_index_0.ordered_data())
    Electrolysers_IDs = list(model.p_H2_electrolyser_nom_index.ordered_data())
    H2_fuel_cell_IDs = list(model.p_H2_fuel_cell_nom_index.ordered_data())
    H2_storage_IDs = list(model.H2_p_s_nom_index.ordered_data())
    H2_transport_pipelines_IDs = list(model.H2_transport_pipelines_p_nom_index.ordered_data())
    H2_transport_trucks_IDs = list(network['H2_transport_trucks'].index)
    H2_buffer_IDs = list(network['H2_buffer'].index)
    H2_compressor_IDs = list(model.H2_compressor_p_nom_index.ordered_data())

    # Save line optimal capacity
    for l in Line_IDs:
        network['lines'].loc[l,'p_nom_opt'] = model.lines_p_nom[l].value
    network['lines'].loc[network['lines'].p_nom_opt < 0,'p_nom_opt'] = 0
    fname = res_path + '/lines.csv'
    network['lines'].to_csv(fname)

    # Save link optimal capacity
    for l in Link_IDs:
        network['links'].loc[l,'p_nom_opt'] = model.links_p_nom[l].value
    network['links'].loc[network['links'].p_nom_opt < 0,'p_nom_opt'] = 0
    fname = res_path + '/links.csv'
    network['links'].to_csv(fname)

    # Save line hourly power flow
    lines_p0 = pd.DataFrame(index = Snapshots, columns = Line_IDs)
    for l in Line_IDs:
        for t in Snapshots:
            lines_p0.loc[t,l] = model.lines_p_p0[l,t].value
    lines_p0.index.name = 'name'
    network['lines_p0'] = lines_p0
    
    lines_p1 = pd.DataFrame(index = Snapshots, columns = Line_IDs)
    for l in Line_IDs:
        for t in Snapshots:
            lines_p1.loc[t,l] = model.lines_p_p1[l,t].value
    lines_p1.index.name = 'name'
    network['lines_p1'] = lines_p1

    lines_p = pd.DataFrame(index = Snapshots, columns = Line_IDs)
    lines_p.index.name = 'name'
    lines_p = lines_p0 - lines_p1
    network['lines_p'] = lines_p

    fname = res_path + '/lines_p.csv'
    network['lines_p'].to_csv(fname)

    # H2 Addition
    # save in both directions for some checkups 
    fname = res_path + '/lines_p0.csv'
    network['lines_p0'].to_csv(fname) 

    fname = res_path + '/lines_p1.csv'
    network['lines_p1'].to_csv(fname)  

    # Save link hourly power flow
    links_p0 = pd.DataFrame(index = Snapshots, columns = Link_IDs)
    for l in Link_IDs:
        for t in Snapshots:
            links_p0.loc[t,l] = model.links_p_p0[l,t].value
    links_p0.index.name = 'name'
    network['links_p0'] = links_p0
    
    links_p1 = pd.DataFrame(index = Snapshots, columns = Link_IDs)
    for l in Link_IDs:
        for t in Snapshots:
            links_p1.loc[t,l] = model.links_p_p1[l,t].value
    links_p1.index.name = 'name'
    network['links_p1'] = links_p1
    
    links_p = pd.DataFrame(index = Snapshots, columns = Link_IDs)
    links_p.index.name = 'name'
    links_p = links_p0 - links_p1
    network['links_p'] = links_p

    fname = res_path + '/links_p.csv'
    network['links_p'].to_csv(fname)
    
    # H2 Addition
    fname = res_path + '/links_p_p0.csv'
    network['links_p0'].to_csv(fname)

    fname = res_path + '/links_p_p1.csv'
    network['links_p1'].to_csv(fname)

    # Save loads / demand
    fname = res_path + '/loads_p_set.csv'
    network['loads_p_set'].iloc[:opts['hours']:opts['resolution'], :].to_csv(fname)

    # Save storage units inflow
    fname = res_path + '/storage_inflow.csv'
    network['storage_inflow'].iloc[:opts['hours']:opts['resolution'], :].to_csv(fname)

    # Save storage spillage
    storage_spillage = pd.DataFrame(index = Snapshots, columns = Spillage_IDs)
    for c in Spillage_IDs:
        for t in Snapshots:
            storage_spillage.loc[t,c] = model.spillage[c,t].value  
    storage_spillage.index.name = 'name'
    network['storage_spillage'] = storage_spillage
    fname = res_path + '/storage_spillage.csv'
    network['storage_spillage'].to_csv(fname)

    # Save storage units optimal capacity
    for c in Sto_IDs:
        network['storage_units'].loc[c,'p_nom_opt'] = model.p_s_nom[c].value
    network['storage_units'].loc[network['storage_units'].p_nom_opt < 0,'p_nom_opt'] = 0
    fname = res_path + '/storage_units.csv'
    network['storage_units'].to_csv(fname)

    # Save buses
    fname = res_path + '/buses.csv'
    network['buses'].to_csv(fname)

    # Save H2 demand
    fname = res_path + '/Hydrogen_demand.csv'
    network['loads_H2_set_LowRes'].to_csv(fname)

    # Save byproduction
    fname = res_path + '/Hydrogen_byproduction.csv'
    network['byproduct_H2_LowRes'].to_csv(fname)

    # Save generator availability
    fname = res_path + '/generators_p_max_pu.csv'
    network['generators_p_max_pu'].iloc[:opts['hours']:opts['resolution'], :].to_csv(fname)

    # Save generator optimal capacity
    for c in Gen_IDs:
        network['generators'].loc[c,'p_nom_opt'] = model.p_nom[c].value
    network['generators'].loc[network['generators'].p_nom_opt < 0,'p_nom_opt'] = 0
    network['generators'].index.name = 'name'
    fname = res_path + '/generators.csv'
    network['generators'].to_csv(fname)

    # Save generator electricity output
    generators_p = pd.DataFrame(index = Snapshots, columns = Gen_IDs)
    for c in Gen_IDs:
        for t in Snapshots:
            generators_p.loc[t,c] = model.p[c,t].value
    generators_p.index.name = 'name'
    network['generators_p'] = generators_p
    fname = res_path + '/generators_p.csv'
    network['generators_p'].to_csv(fname)

    # Save storage unit output
    storage_units_p = pd.DataFrame(index = Snapshots, columns = Sto_IDs)
    for c in Sto_IDs:
        for t in Snapshots:
            storage_units_p.loc[t,c] = model.dispatch[c,t].value-model.store[c,t].value
    storage_units_p.index.name = 'name'
    network['storage_units_p'] = storage_units_p
    fname = res_path + '/storage_units_p.csv'
    network['storage_units_p'].to_csv(fname)

    # save storage dispatch
    storage_units_p_disp = pd.DataFrame(index = Snapshots, columns = Sto_IDs)
    for c in Sto_IDs:
        for t in Snapshots:
            storage_units_p_disp.loc[t,c] = model.dispatch[c,t].value
    storage_units_p_disp.index.name = 'name'
    fname = res_path + '/storage_units_dispatch_p.csv'
    storage_units_p_disp.to_csv(fname)

    # Save storage unit SOC
    storage_units_soc = pd.DataFrame(index = Snapshots, columns = Sto_IDs)
    for c in Sto_IDs:
        for t in Snapshots:
            storage_units_soc.loc[t,c] = model.soc[c,t].value
    storage_units_soc.index.name = 'name'
    network['storage_units_soc'] = storage_units_soc
    fname = res_path + '/storage_units_soc.csv'
    network['storage_units_soc'].to_csv(fname)

    ### Save hourly curtailment
    # Variable generation technologies
    variable_techs = []
    for col in network['generators_p_max_pu'].columns:
        variable_techs.append(col.split()[-1])
    variable_techs = list(set(variable_techs))

    network['generators_p_curtailed'] = pd.DataFrame(index = network['generators_p'].index, columns = network['generators_p'].columns)
    
    for col in network['generators_p_curtailed'].columns:
        
        if col.split()[-1] in variable_techs:
            network['generators_p_curtailed'][col] = network['generators_p_max_pu'][col]*network['generators'].loc[col,'p_nom_opt']
            network['generators_p_curtailed'][col] -= network['generators_p'][col]
        
        else:
            network['generators_p_curtailed'][col] = network['generators'].loc[col,'p_nom_opt']
            network['generators_p_curtailed'][col] -= network['generators_p'][col]
    
    fname = res_path + '/generators_p_curtailed.csv'
    network['generators_p_curtailed'].to_csv(fname)

    # H2 Addition
    # Save electrolyser optimal capacity

    for e in Electrolysers_IDs:
        network['electrolysers'].loc[e,'p_nom_opt'] = model.p_H2_electrolyser_nom[e].value
    network['electrolysers'].loc[network['electrolysers'].p_nom_opt < 0,'p_nom_opt'] = 0
    network['electrolysers'].index.name = 'name'
    fname = res_path + '/electrolyser_capacity.csv'
    network['electrolysers'].to_csv(fname)

    # Save electrolyser hydrogen output
    electrolysers_p = pd.DataFrame(index = Snapshots, columns = Electrolysers_IDs)
    for e in Electrolysers_IDs:
        for t in Snapshots:
            electrolysers_p.loc[t,e] = model.p_H2_electrolyser[e,t].value
    electrolysers_p.index.name = 'name'
    network['electrolysers_p'] = electrolysers_p
    fname = res_path + '/electrolysers_p.csv'
    network['electrolysers_p'].to_csv(fname)

    # Save fuel cell optimal capacity
    for e in H2_fuel_cell_IDs:
        network['fuel_cells'].loc[e,'p_nom_opt'] = model.p_H2_fuel_cell_nom[e].value
    network['fuel_cells'].loc[network['fuel_cells'].p_nom_opt < 0,'p_nom_opt'] = 0
    network['fuel_cells'].index.name = 'name'
    fname = res_path + '/fuel_cells_capacity.csv'
    network['fuel_cells'].to_csv(fname)


    # Save fuel cell electricity output
    H2_fuel_cells_p = pd.DataFrame(index = Snapshots, columns = H2_fuel_cell_IDs)
    for e in H2_fuel_cell_IDs:
        for t in Snapshots:
            H2_fuel_cells_p.loc[t,e] = model.p_H2_fuel_cell[e,t].value
    H2_fuel_cells_p.index.name = 'name'
    network['H2_fuel_cells_p'] = H2_fuel_cells_p
    fname = res_path + '/H2_fuel_cells_p.csv'
    network['H2_fuel_cells_p'].to_csv(fname)

    # Save H2 storage unit output
    H2_storage_units_p = pd.DataFrame(index = Snapshots, columns = H2_storage_IDs)
    for c in H2_storage_IDs:
        for t in Snapshots:
            H2_storage_units_p.loc[t,c] = model.H2_dispatch[c,t].value-model.H2_store[c,t].value
    H2_storage_units_p.index.name = 'name'
    network['H2_storage_units_p'] = H2_storage_units_p
    fname = res_path + '/H2_storage_units_p.csv'
    network['H2_storage_units_p'].to_csv(fname)

    
    # save H2 storage dispatch
    H2_storage_units_p_disp = pd.DataFrame(index = Snapshots, columns = H2_storage_IDs)
    for c in H2_storage_IDs:
        for t in Snapshots:
            H2_storage_units_p_disp.loc[t,c] = model.H2_dispatch[c,t].value
    H2_storage_units_p_disp.index.name = 'name'
    fname = res_path + '/H2_storage_units_dispatch_p.csv'
    H2_storage_units_p_disp.to_csv(fname)

    # Save H2 storage unit SOC
    H2_storage_units_soc = pd.DataFrame(index = Snapshots, columns = H2_storage_IDs)
    for c in H2_storage_IDs:
        for t in Snapshots:
            H2_storage_units_soc.loc[t,c] = model.H2_soc[c,t].value
    H2_storage_units_soc.index.name = 'name'
    network['H2_storage_units_soc'] = H2_storage_units_soc
    fname = res_path + '/H2_storage_units_soc.csv'
    network['H2_storage_units_soc'].to_csv(fname)

    # Save H2 storage units optimal capacity
    for c in H2_storage_IDs:
        network['H2_storage_units'].loc[c,'p_nom_opt'] = model.H2_p_s_nom[c].value
    network['H2_storage_units'].loc[network['H2_storage_units'].p_nom_opt < 0,'p_nom_opt'] = 0
    fname = res_path + '/H2_storage_units.csv'
    network['H2_storage_units'].to_csv(fname)

    
    # Save H2 transport optimal capacity
    for l in H2_transport_pipelines_IDs:
        network['H2_transport_pipelines'].loc[l,'p_nom_opt'] = model.H2_transport_pipelines_p_nom[l].value
    #network['H2_transport'].loc[network['H2_transport'].p_nom_opt < 0,'p_nom_opt'] = 0
    fname = res_path + '/H2_transport_pipelines.csv'
    network['H2_transport_pipelines'].to_csv(fname)


    fname = res_path + '/H2_transport_trucks.csv'
    network['H2_transport_trucks'].to_csv(fname)


    # Save hourly H2 transport pipelines
    H2_transport_pipelines_p0 = pd.DataFrame(index = Snapshots, columns = H2_transport_pipelines_IDs)
    for l in H2_transport_pipelines_IDs:
        for t in Snapshots:
            H2_transport_pipelines_p0.loc[t,l] = model.H2_transport_pipelines_p_p0[l,t].value
    H2_transport_pipelines_p0.index.name = 'name'
    network['H2_transport_piplines_p0'] = H2_transport_pipelines_p0

    # H2 Addition
    fname = res_path + '/H2_transport_pipelines_p0.csv'
    network['H2_transport_piplines_p0'].to_csv(fname) 

    # Save hourly H2 transport trucks
    # old implementation
    '''H2_transport_trucks_p0 = pd.DataFrame(index = Snapshots, columns = H2_transport_trucks_IDs)
    for l in H2_transport_trucks_IDs:
        for t in Snapshots:
            H2_transport_trucks_p0.loc[t,l] = model.H2_transport_trucks_p_p0[l,t].value
    H2_transport_trucks_p0.index.name = 'name'
    network['H2_transport_trucks_p0'] = H2_transport_trucks_p0
    # H2 Addition
    fname = res_path + '/H2_transport_trucks_p0.csv'
    network['H2_transport_trucks_p0'].to_csv(fname) '''

    # Save buffer H2 imports
    H2_buffer_p = pd.DataFrame(index = Snapshots, columns = H2_buffer_IDs)
    for l in H2_buffer_IDs:
        for t in Snapshots:
            H2_buffer_p.loc[t,l] = model.H2_buffer_p[l,t].value
    H2_buffer_p.index.name = 'name'
    network['H2_buffer_p'] = H2_buffer_p
    fname = res_path + '/H2_buffer_p.csv'
    network['H2_buffer_p'].to_csv(fname) 
        
    fname = res_path + '/loads_H2_liquefier_set_LowRes.csv'
    network['loads_H2_liquefier_set_LowRes'].to_csv(fname)

    # Save compressor optimal capacity

    for e in H2_compressor_IDs:
        network['H2_compressor'].loc[e,'p_nom_opt'] = model.H2_compressor_p_nom[e].value
    network['H2_compressor'].loc[network['H2_compressor'].p_nom_opt < 0,'p_nom_opt'] = 0
    network['H2_compressor'].index.name = 'name'
    fname = res_path + '/compressor_capacity.csv'
    network['H2_compressor'].to_csv(fname)

    # H2 Addition
    # Save compressor hydrogen data
    compressed_p = pd.DataFrame(index = Snapshots, columns = H2_compressor_IDs)
    for e in H2_compressor_IDs:
        for t in Snapshots:
            compressed_p.loc[t,e] = model.H2_compressed_p[e,t].value
    compressed_p.index.name = 'name'
    network['compressed_p'] = compressed_p
    fname = res_path + '/compressed_p.csv'
    network['compressed_p'].to_csv(fname)

    decompressed_p = pd.DataFrame(index = Snapshots, columns = H2_compressor_IDs)
    for e in H2_compressor_IDs:
        for t in Snapshots:
            decompressed_p.loc[t,e] = model.H2_decompressed_p[e,t].value
    decompressed_p.index.name = 'name'
    network['decompressed_p'] = decompressed_p
    fname = res_path + '/decompressed_p.csv'
    network['decompressed_p'].to_csv(fname)


    # Save H2 input parameters
    H2_assumptions = opts['hydrogen_assumptions']
    H2_assumptions = H2_assumptions[opts['h2_scenario']]
    fname = res_path + '/H2_scenario_variables.csv'
    H2_assumptions.to_csv(fname)


    # save H2 truck transport data
    # for simplicity, as not all these values were needed for analysis later they are left in the original formatting
    # the ones needed are then converted in the analysis file
    pd.Series(model.invested_trucks.extract_values(), name=model.invested_trucks.name).to_csv(res_path+ "/invested_trucks.csv")
    pd.Series(model.trucks_empty.extract_values(), name=model.trucks_empty.name).to_csv(res_path+ "/trucks_empty.csv")
    pd.Series(model.trucks_full.extract_values(), name=model.trucks_full.name).to_csv(res_path+ "/trucks_full.csv")
    pd.Series(model.trucks_full_in_transit.extract_values(), name=model.trucks_full_in_transit.name).to_csv(res_path+ "/trucks_full_in_transit.csv")
    pd.Series(model.trucks_full_staying.extract_values(), name=model.trucks_full_staying.name).to_csv(res_path+ "/trucks_full_staying.csv")
    
    pd.Series(model.trucks_empty_in_transit.extract_values(), name=model.trucks_empty_in_transit.name).to_csv(res_path+ "/trucks_empty_in_transit.csv")
    pd.Series(model.trucks_empty_staying.extract_values(), name=model.trucks_empty_staying.name).to_csv(res_path+ "/trucks_empty_staying.csv")
    pd.Series(model.trucks_charged.extract_values(), name=model.trucks_charged.name).to_csv(res_path+ "/trucks_charged.csv")
    pd.Series(model.trucks_discharged.extract_values(), name=model.trucks_discharged.name).to_csv(res_path+ "/trucks_discharged.csv")

    pd.Series(model.trucks_departing_full.extract_values(), name=model.trucks_departing_full.name).to_csv(res_path+ "/trucks_departing_full.csv")
    #pd.Series(model.trucks_arriving_full.extract_values(), name=model.trucks_arriving_full.name).to_csv(res_path+ "/trucks_arriving_full.csv")
    pd.Series(model.trucks_departing_empty.extract_values(), name=model.trucks_departing_empty.name).to_csv(res_path+ "/trucks_departing_empty.csv")
    #pd.Series(model.trucks_arriving_empty.extract_values(), name=model.trucks_arriving_empty.name).to_csv(res_path+ "/trucks_arriving_empty.csv")
    


    
    # Save the solver status
    solver_status = pd.DataFrame()
    solver_status.loc['termination','value'] = results.solver.termination_condition.value
    solver_status.loc['status','value'] = results.solver.status.value

    status_path = res_path + '/solver_status.csv'
    solver_status.index.name = 'parameter'
    solver_status.to_csv(status_path)

    # Save total impacts
    case = pd.DataFrame()
    case.loc[scenario,'Cost slack (%)'] = slack
    case.loc[scenario,'Cost (BEUR)'] = round(model.cost()/1e9,3)
    case.loc[scenario,'Jobs (thousand)'] = round(model.jobs()/1e3,3)
    case.loc[scenario,'GHG (MtCO2)'] = round(model.ghg()/1e6,3)
    case.loc[scenario,'PM10 (ktPM10)'] = round(model.pm10()/1e6,3)
    case.loc[scenario,'Land (km2)'] = round(model.landuse()/1e6,3)
    case.loc[scenario,'termination'] = results.solver.termination_condition.value
    case.loc[scenario,'status'] = results.solver.status.value

    case_path = res_path + '/case.csv'
    case.index.name = 'scenario'
    case.to_csv(case_path)
    
    # Print successful completion
    print("Successfully stored the data.")
