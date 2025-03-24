"""
Copyright 2021 Jan-Philipp Sasse (UNIGE), Evelina Trutnevyte (UNIGE)
~~~~~~~~~~~~~
Plot summary figures

"""

# Import packages
import matplotlib.pyplot as plt
import seaborn as sns
from EXPANSE.read_settings import *
import cartopy.crs as ccrs
import cartopy
from matplotlib.patches import Wedge, Circle
from matplotlib.collections import LineCollection, PatchCollection
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely import wkt
import os

# Read settings
opts = read_settings()
sum_path = opts['output_path'] + 'Summary'


colors = {"onwind" : "#235ebc", "solar_roof" : "#f9d002","solar_facade" : "#ffef60","offwind" : "#74c6f2",
            "biogas" : '#23932d',"woodybiomass" : '#06540d',"waste" : 'olive', 
            "gas" : "#d35050","nuclear" : "darkorange","ror" : "#4adbc8","hydro" : "#08ad97", 
            "oil" : "#262626","geothermal" : "#ba91b1","lignite" : "#9e5a01","coal" : "#707070", 
            "solar_battery" : "#f902bb", "battery" : "#b8ea04", "H2" : "#a9acd1", "PHS" : "#a31597","hydro" : "#08ad97",
            "solar_battery charge" : "#f902bb", "battery charge" : "#b8ea04", "H2 charge" : "#a9acd1", "PHS charge" : "#a31597",
            "solar_battery discharge" : "#f902bb","battery discharge" : "#b8ea04", "H2 discharge" : "#a9acd1", "PHS discharge" : "#a31597"}

def get_scenario(path,scenario):
    
    # Get scenario results files
    ParPath = os.path.dirname(os.path.dirname(opts['input_path']))
    fname = ParPath + '/data/admin_boundaries/swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET.shp'
    regions = gpd.read_file(fname)
    regions = regions.to_crs('epsg:4326')

    # Define path
    res_path = path + scenario
    
    # read generator capacity
    generators = pd.read_csv(res_path+'/generators.csv',index_col=0)

    # read and add generator electricity output to generator capacity file
    generators_p = pd.read_csv(res_path+'/generators_p.csv',index_col=0)
    scaling_factor = 8760/len(generators_p)
    
    for gen in generators.index:
        generators.loc[gen,'p_annual'] = generators_p[gen].sum()*scaling_factor

    # read storage capacity
    storage_units = pd.read_csv(res_path+'/storage_units.csv',index_col=0)

    # read hourly storage operation
    storage_units_p = pd.read_csv(res_path+'/storage_units_p.csv',index_col=0)

    # get only dispatch (charging doesnt add to the costs)
    for sto in storage_units.index:
        storage_units.loc[sto,'p_annual'] = storage_units_p[storage_units_p>0][sto].sum()*scaling_factor

    # read line capacity
    lines = pd.read_csv(res_path+'/lines.csv',index_col=0)

    # read line hourly operation
    lines_p = pd.read_csv(res_path+'/lines_p.csv',index_col=0)

    # get absolute values of hourly line operation (flow in both directions adds to the costs)
    lines_p_abs = abs(lines_p)

    # add line operation to line capacity file
    for tra in lines.index:
        lines.loc[tra,'p_annual'] = lines_p_abs[str(tra)].sum()*scaling_factor

    # read link capacity
    links = pd.read_csv(res_path+'/links.csv',index_col=0)

    # read link hourly operation
    links_p = pd.read_csv(res_path+'/links_p.csv',index_col=0)

    # get absolute values of hourly link operation (flow in both directions adds to the costs)
    links_p_abs = abs(links_p)

    # add link operation to link capacity file
    for tra in links.index:
        links.loc[tra,'p_annual'] = links_p_abs[str(tra)].sum()*scaling_factor

    # calculate generator capacity expansion
    generators['p_nom_exp'] = generators['p_nom_opt'] - generators['p_nom_cur']

    # calculate storage capacity expansion
    storage_units['p_nom_exp'] = storage_units['p_nom_opt']-storage_units['p_nom_cur']

    # calculate line capacity expansion
    lines['p_nom_exp'] = lines['p_nom_opt'] - lines['p_nom_min']

    # calculate link capacity expansion
    links['p_nom_exp'] = links['p_nom_opt'] - links['p_nom_min']
    
    # read storage unit soc
    storage_units_soc = pd.read_csv(res_path+'/storage_units_soc.csv',index_col=0)
    
    # read buses
    buses = pd.read_csv(res_path+'/buses.csv',index_col=0)
    
    # read loads
    loads_p_set = pd.read_csv(res_path+'/loads_p_set.csv',index_col=0)

    # read curtailment
    generators_p_curtailed = pd.read_csv(res_path+'/generators_p_curtailed.csv',index_col=0)
    
    n = dict()
    n['generators'] = generators
    n['generators_p'] = generators_p
    n['generators_p_curtailed'] = generators_p_curtailed
    n['storage_units'] = storage_units
    n['storage_units_p'] = storage_units_p
    n['storage_units_soc'] = storage_units_soc
    n['lines'] = lines
    n['lines_p'] = lines_p
    n['links'] = links
    n['links_p'] = links_p
    n['buses'] = buses
    n['loads_p_set'] = loads_p_set
    n['regions'] = regions
    
    return n

def plot_operation(n, path, scenario):
    
    # Plot hourly operation figure
    p_by_carrier = n['generators_p'].groupby(n['generators'].carrier, axis=1).sum()
    p_by_carrier = abs(p_by_carrier)
    s_by_carrier = n['storage_units_p'].groupby(n['storage_units'].carrier, axis=1).sum()
    s_by_carrier_discharge = s_by_carrier.copy()
    s_by_carrier_discharge[s_by_carrier_discharge < 0] = 0
    s_by_carrier_charge = s_by_carrier.copy()
    s_by_carrier_charge[s_by_carrier_charge > 0] = 0

    s_by_carrier_charge = s_by_carrier_charge.rename(columns={"H2": "H2 charge", "PHS": "PHS charge", 
                                                              "battery": "battery charge", "solar_battery": "solar_battery charge"})
    
    s_by_carrier_charge = s_by_carrier_charge.drop(columns=['hydro'])

    s_by_carrier_discharge = s_by_carrier_discharge.rename(columns={"H2": "H2 discharge", "PHS": "PHS discharge", 
                                                                    "battery": "battery discharge", "solar_battery": "solar_battery discharge"})
    
    p_by_carrier = p_by_carrier.merge(s_by_carrier_discharge,left_index=True,right_index=True)
    p_by_carrier = p_by_carrier.merge(s_by_carrier_charge,left_index=True,right_index=True)
    
    # reorder
    cols = ["solar_battery charge","battery charge","H2 charge","PHS charge","nuclear","geothermal","lignite","coal","woodybiomass","biogas","waste",
            "hydro","ror","gas","oil","onwind","offwind","solar_roof","solar_facade","solar_battery discharge","battery discharge","H2 discharge","PHS discharge"]

    for col in cols:
        if col not in p_by_carrier.columns:
            p_by_carrier[col] = 0

    p_by_carrier = p_by_carrier[cols]
    p_by_carrier.index = pd.to_datetime(p_by_carrier.index)

    Snapshots = list(n['loads_p_set'].index)
    demand_GW = pd.DataFrame(n['loads_p_set'].sum(axis=1)/1e3).rename(columns={0: "demand"})
    demand_GW.index = pd.to_datetime(demand_GW.index)
    demand_GW = demand_GW.loc[Snapshots]
    
    fig, ax = plt.subplots(1,1)
    fig.suptitle('Hourly generation')
    (p_by_carrier/1e3).plot(kind="area",
                            ax=ax,
                            linewidth=0.1,
                            color=[colors[col] for col in p_by_carrier.columns])
    demand_GW.plot(ax=ax,linewidth=1,color='black')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], ncol=1,loc="upper left",bbox_to_anchor=(1.01, 1))
    ax.set_ylabel("GW")
    ax.set_xlabel("")
    
    f_name = path + scenario + '/hourly_operation.png'
    fig.savefig(f_name, dpi=300, bbox_inches='tight')

def plot_storage_soc(n, path, scenario): 
    
    # Plot storage soc levels

    storage_units_soc_copy = n['storage_units_soc'].copy()
    storage_units_soc_copy['PHS'] = storage_units_soc_copy.filter(regex='PHS', axis=1).sum(axis=1)
    storage_units_soc_copy['hydro'] = storage_units_soc_copy.filter(regex='hydro', axis=1).sum(axis=1)
    storage_units_soc_copy['solar_battery'] = storage_units_soc_copy.filter(regex='solar_battery', axis=1).sum(axis=1)
    storage_units_soc_copy['battery'] = storage_units_soc_copy.filter(regex='battery', axis=1).sum(axis=1)
    storage_units_soc_copy['H2'] = storage_units_soc_copy.filter(regex='H2', axis=1).sum(axis=1)
    storage_units_soc_copy.index = pd.to_datetime(storage_units_soc_copy.index)
    storage_units_soc_percent = storage_units_soc_copy[['PHS','hydro','battery','H2','solar_battery']].copy()
    storage_units_soc_percent.loc[:,'PHS'] = storage_units_soc_percent.loc[:,'PHS']/storage_units_soc_percent['PHS'].max()*100
    storage_units_soc_percent.loc[:,'hydro']= storage_units_soc_percent.loc[:,'hydro']/storage_units_soc_percent['hydro'].max()*100
    storage_units_soc_percent.loc[:,'solar_battery']= storage_units_soc_percent.loc[:,'solar_battery']/storage_units_soc_percent['solar_battery'].max()*100
    storage_units_soc_percent.loc[:,'battery'] = storage_units_soc_percent.loc[:,'battery']/storage_units_soc_percent['battery'].max()*100
    storage_units_soc_percent.loc[:,'H2'] = storage_units_soc_percent.loc[:,'H2']/storage_units_soc_percent['H2'].max()*100
    
    fig, ax = plt.subplots(1,1)
    fig.suptitle('Storage state of charge')
    storage_units_soc_percent[:].plot(ax=ax,alpha=0.5,color=[colors[col] for col in storage_units_soc_percent.columns])
    ax.set_ylabel('SOC [%]')
    ax.set_xlabel('')

    f_name = path + scenario + '/hourly_soc.png'
    fig.savefig(f_name, dpi=300, bbox_inches='tight')

def plot_storage_operation(n, path, scenario):
    
    # plot storage operation
    fig, axes = plt.subplots(2,3)
    fig.suptitle('Storage operation')

    storage_units_p_copy = n['storage_units_p'].copy()
    storage_units_p_copy['PHS'] = storage_units_p_copy.filter(regex='PHS', axis=1).sum(axis=1)
    storage_units_p_copy['hydro'] = storage_units_p_copy.filter(regex='hydro', axis=1).sum(axis=1)
    storage_units_p_copy['solar_battery'] = storage_units_p_copy.filter(regex='solar_battery', axis=1).sum(axis=1)
    storage_units_p_copy['battery'] = storage_units_p_copy.filter(regex='battery', axis=1).sum(axis=1)
    storage_units_p_copy['H2'] = storage_units_p_copy.filter(regex='H2', axis=1).sum(axis=1)
    storage_units_p_copy.index = pd.to_datetime(storage_units_p_copy.index)
    storage_units_p_copy['PHS'][:].plot(ax=axes[0,0],xlabel='PHS')
    storage_units_p_copy['hydro'][:].plot(ax=axes[0,1],xlabel='Hydro')
    storage_units_p_copy['solar_battery'][:].plot(ax=axes[0,2],xlabel='solar_battery')
    storage_units_p_copy['battery'][:].plot(ax=axes[1,0],xlabel='Battery')
    storage_units_p_copy['H2'][:].plot(ax=axes[1,1],xlabel='Hydrogen')

    fig.tight_layout()
    f_name = path + scenario + '/storage_operation.png'
    fig.savefig(f_name, dpi=300, bbox_inches='tight')

def plot_map(n, path, scenario):

    # convert geometry to spatial data format 
    n['lines']['geometry'] = n['lines']['geometry'].apply(wkt.loads)
    n['lines'] = gpd.GeoDataFrame(n['lines'], geometry='geometry',crs='epsg:4326')
    n['links']['geometry'] = n['links']['geometry'].apply(wkt.loads)
    n['links'] = gpd.GeoDataFrame(n['links'], geometry='geometry',crs='epsg:4326')
    n['buses']['geometry'] = n['buses']['geometry'].apply(wkt.loads)
    n['buses'] = gpd.GeoDataFrame(n['buses'],geometry='geometry',crs='epsg:4326')
    
    fig,ax = plt.subplots(1,1,subplot_kw={"projection":ccrs.PlateCarree()})
    
    ## DATA
    
    tech_colors = colors
    
    n['generators'].loc[n['generators']['p_nom_opt']<0,'p_nom_opt']=0
    n['storage_units']['p_nom_cur'] = n['storage_units']['p_nom_cur']
    n['lines']['p_nom_cur'] = n['lines']['p_nom_min']
    n['links']['p_nom_cur'] = n['links']['p_nom_min']
    
    bus_sizes = pd.concat((n['generators'][n['generators'].bus.str[:2]=='CH'].groupby(['bus', 'carrier'])['p_nom_opt'].sum(),
        n['storage_units'][n['storage_units'].bus.str[:2]=='CH'].groupby(['bus', 'carrier'])['p_nom_opt'].sum()))
    
    ## FORMAT
    bus_size_factor = 5e+4

    ## PLOT
    n['regions'].plot(ax=ax, color='antiquewhite', zorder=3, linewidth=0.1, edgecolor='black') # base map
    ax.add_feature(cartopy.feature.LAND.with_scale('50m'),facecolor='whitesmoke', zorder=1) # outside land
    ax.add_feature(cartopy.feature.BORDERS.with_scale('50m'),linewidth=0.2, zorder=4) # national borders
    ax.coastlines(linewidth=0.2, resolution='50m', zorder=4) # coast lines
    n['lines'].plot(color='#a31597', ax=ax, alpha = 0.5, linewidth = n['lines']['p_nom_opt']/1000, zorder=10) # transmission lines
    n['links'].plot(color='#08ad97', ax=ax, alpha = 1, linewidth = n['links']['p_nom_opt']/1000, zorder=10) # transmission lines
    
    bus_colors = pd.Series(colors)
    bus_sizes = bus_sizes.sort_index(level=0, sort_remaining=False)
    bus_sizes = bus_sizes/bus_size_factor
    x, y = n['buses']["x"], n['buses']["y"]
    bus_alpha = 1
    patches = []
    
    for b_i in bus_sizes.index.levels[0]:
        s = bus_sizes.loc[b_i]
        radius = s.sum()**0.5
        
        if radius == 0.0:
            ratios = s
        else:
            ratios = s/s.sum()

        start = 0.25
        for i, ratio in ratios.iteritems():
            patches.append(Wedge((x.at[b_i], y.at[b_i]), radius,
                                 360*start, 360*(start+ratio),
                                 facecolor=bus_colors[i], edgecolor = 'black', linewidth = 0.1, 
                                 alpha=bus_alpha, zorder=10000))
            start += ratio
            bus_collection = PatchCollection(patches, match_original=True, zorder=10000)
            ax.add_collection(bus_collection)
    
    ax.set_aspect('equal')

    f_name = path + scenario + '/capacity_map.png'
    fig.savefig(f_name, dpi=300, bbox_inches='tight')
