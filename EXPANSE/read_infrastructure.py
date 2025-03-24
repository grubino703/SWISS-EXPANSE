"""
Copyright 2022 Jan-Philipp Sasse (UNIGE), Evelina Trutnevyte (UNIGE)
~~~~~~~~~~~~~
Read electricity system infrastructure

Inputs: individual csv files
Outputs: dictionary with all csv files -> 'network'
"""

# Import required packages
import pandas as pd
import geopandas as gpd
from shapely import wkt

def read_infrastructure(input_path, resolution, scenario):
    
    # Read hourly capacity factors
    fname = input_path + 'generators_p_max_pu.csv'
    generators_p_max_pu = pd.read_csv(fname, index_col=[0])
    generators_p_max_pu = generators_p_max_pu.astype('float64')
    generators_p_max_pu = generators_p_max_pu.round(decimals=3)
    
    # Read hourly demand
    fname = input_path + 'loads_p_set.csv'
    loads_p_set = pd.read_csv(fname, index_col=[0])
    loads_p_set = loads_p_set.astype('float64')

    # Read loads at specific temporal resolution
    loads_p_set_LowRes = loads_p_set.iloc[::resolution, :].copy()

    # get scenario settings

    # H2 Addition

    # Read hourly airport hydrogen liquefaction electricity demand
    fname = input_path + '2050_'+ scenario + '_H2_additional_airport_energy_liquefaction_MW.csv'
    loads_H2_liquefier_set = pd.read_csv(fname, index_col=[0])
    loads_H2_liquefier_set = loads_H2_liquefier_set.astype('float64')

    # Read liquefaction loads at specific temporal resolution
    loads_H2_liquefier_set_LowRes = loads_H2_liquefier_set.iloc[::resolution, :].copy()

    # Read hourly H2 demand
    fname = input_path + '2050_'+ scenario + '_H2_demand_per_hour_node_MW.csv'
    loads_H2_set = pd.read_csv(fname, index_col = [0])
    loads_H2_set = loads_H2_set.astype('float64')

    # Read H2 loads at specific temporal resolution
    loads_H2_set_LowRes = loads_H2_set.iloc[::resolution, :].copy()

    # Read hourly H2 byproduction
    fname = input_path + '2050_'+ scenario + '_byproduction_per_hour_node.csv'
    byproduct_H2 = pd.read_csv(fname, index_col = [0])
    byproduct_H2 = byproduct_H2.astype('float64')

    byproduct_H2_LowRes = byproduct_H2.iloc[::resolution, :].copy()
    
    # Read storage units without H2 as this is treated separately
    fname = input_path + 'storage_units_without_H2.csv'
    storage_units = pd.read_csv(fname, index_col=[0])

    # read H2 storage units
    fname = input_path + '2050_'+ scenario + '_H2_storage_units.csv'
    H2_storage_units = pd.read_csv(fname, index_col=[0])
    
    # Read storage inflow
    fname = input_path + 'storage_inflow.csv'
    storage_inflow = pd.read_csv(fname, index_col=[0])
    storage_inflow = storage_inflow.astype('float64')
    
    # Read generators
    fname = input_path + '2050_'+scenario+'_generators.csv'
    generators = pd.read_csv(fname, index_col=[0])

    ## H2 Addition
    # Read electrolysers
    fname = input_path + '2050_'+ scenario + '_electrolysers.csv'
    electrolysers = pd.read_csv(fname, index_col=[0])

    # Read fuel cells
    fname = input_path + '2050_'+ scenario + '_fuel_cells.csv'
    fuel_cells = pd.read_csv(fname, index_col=[0])
    
    # Read buses
    fname = input_path + 'buses.csv'
    buses = pd.read_csv(fname, index_col=[0])
    buses['geometry'] = buses['geometry'].apply(wkt.loads)
    buses = gpd.GeoDataFrame(buses, geometry='geometry')
    
    # Read HVAC transmission lines
    fname = input_path + 'lines.csv'
    lines = pd.read_csv(fname, index_col=[0])
    lines['geometry'] = lines['geometry'].apply(wkt.loads)
    lines = gpd.GeoDataFrame(lines, geometry='geometry')
    
    # Read HVDC transmission links
    fname = input_path + 'links.csv'
    links = pd.read_csv(fname, index_col=[0])
    links['geometry'] = links['geometry'].apply(wkt.loads)
    links = gpd.GeoDataFrame(links, geometry='geometry')

    # H2 Addition
    # Read H2 transport paths (pipelines and trucks, depending on storyline)
    fname = input_path + '2050_' + scenario + '_H2_transport_pipelines.csv'
    H2_transport_pipelines = pd.read_csv(fname, index_col = [0])
    H2_transport_pipelines['geometry'] = H2_transport_pipelines['geometry'].apply(wkt.loads)
    H2_transport_pipelines = gpd.GeoDataFrame(H2_transport_pipelines, geometry='geometry')

    fname = input_path + '2050_' + scenario + '_H2_transport_trucks.csv'
    H2_transport_trucks = pd.read_csv(fname, index_col = [0])
    H2_transport_trucks['geometry'] = H2_transport_trucks['geometry'].apply(wkt.loads)
    H2_transport_trucks = gpd.GeoDataFrame(H2_transport_trucks, geometry='geometry')


    # Read hydrogen buffer imports from outside of our five modelled countries
    fname = input_path + '2050_' + scenario + '_H2_buffer_imports.csv'
    H2_buffer = pd.read_csv(fname, index_col = [0])

# Read compressor information (per node and per pressure change)
    fname = input_path + '2050_' + scenario + '_H2_compressors.csv'
    H2_compressor = pd.read_csv(fname, index_col = [0])


    # Store all dataframes into a dictionary
    network = dict()
    network['loads_p_set'] = loads_p_set
    network['loads_p_set_LowRes'] = loads_p_set_LowRes
    network['loads_H2_set'] = loads_H2_set
    network['loads_H2_set_LowRes'] = loads_H2_set_LowRes
    network['byproduct_H2_LowRes'] = byproduct_H2_LowRes
    network['loads_H2_liquefier_set_LowRes'] = loads_H2_liquefier_set_LowRes
    network['generators'] = generators
    network['electrolysers'] = electrolysers
    network['fuel_cells'] = fuel_cells
    network['H2_storage_units'] = H2_storage_units
    network['generators_p_max_pu'] = generators_p_max_pu
    network['storage_units'] = storage_units
    network['storage_inflow'] = storage_inflow
    network['buses'] = buses
    network['lines'] = lines
    network['links'] = links
    network['H2_transport_pipelines'] = H2_transport_pipelines
    network['H2_transport_trucks'] = H2_transport_trucks
    network['H2_buffer'] = H2_buffer
    network['H2_compressor'] = H2_compressor


    return network
