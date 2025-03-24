
# Find EXPANSE code
import os
CurPath = os.getcwd()
ParPath = os.path.dirname(CurPath)
print("Parent dir",ParPath)
import sys
sys.path.append(ParPath)

# Import EXPANSE
print('Importing EXPANSE code ...')
from EXPANSE.init_model import *
from EXPANSE.save_data import *
from EXPANSE.solve_model import *
from EXPANSE.create_directories import *

# Set temp directory for LP files
from pyomo.common.tempfiles import TempfileManager
TempfileManager.tempdir = '../../scratch/Temp/'

# Create directories
create_directories()

# Initialize model instance
print('Initalizing model instance with Pyomo...')
start = time.time()
model = init_model()
end = time.time()
print('Time to initialize model: {} seconds'.format(round((end-start))))

# Set constraint that solar_battery and solar PV combined cannot go above total potential
storage_dict = network['storage_units'].to_dict()

def solar_rule(model, cur_bus):
    
    cur_gen_list = list(network['generators'][(network['generators'].carrier=='solar_roof')&
                                              (network['generators'].bus==cur_bus)].index)
    
    c_batt = '{} solar_battery'.format(cur_bus)
    
    return(sum(model.p_nom[c] for c in cur_gen_list) + model.p_s_nom[c_batt] <= storage_dict['p_nom_max'][c_batt])

sol_list = list(network['storage_units'][network['storage_units'].carrier=='solar_battery']['bus'].unique())

model.solar_constraint = Constraint(sol_list, rule = solar_rule)

# gen tech phase-out
for c in list(model.p_nom_index):
    if (c[:2] == 'CH') &(c.split()[1] == 'nuclear'):
        model.p_nom[c].fix(network['generators'].loc[c,'p_nom_max'] * 0.0)
        print('~ ~ ~ CH phase-out: ' + c)
###

# Add RES target
print('Building RES target ...')

Gen_IDs = list(network['generators'].index) # Generation technologies
Sto_IDs = list(network['storage_units'].index) # Storage technologies
Snapshots = list(network['loads_p_set_LowRes'].index[:int(opts['hours']/opts['resolution'])]) # Snapshots to optimize

#amendment to solar_battery: spillage(t) <= inflow(t)
network['storage_inflow_LowRes'] = network['storage_inflow'].iloc[::opts['resolution'], :].copy()
network['storage_inflow_LowRes'] = network['storage_inflow_LowRes'].iloc[:int(opts['hours']/opts['resolution'])]
inflow_dict = network['storage_inflow_LowRes'].to_dict()
solar_bat_IDs = list(network['storage_inflow'].filter(like='battery').columns)
def solar_bat_rule_1(model,c,t):
    return(
        model.spillage[c,t] <= model.p_s_nom[c] *  inflow_dict[c][t]
    )
    
model.spillage_max = Constraint(solar_bat_IDs, Snapshots, rule = solar_bat_rule_1)

# add CH bio target
def bio_target_expr(model, country):

    expr = 0

    if country == 'CH':
        
        bio_target_techs = ['woodybiomass','biogas','waste']

        # Iterate through RES generation technologies
        for c in Gen_IDs:
            if (c[:2] == country) & (c.split()[-1] in bio_target_techs):
                for t in Snapshots:
                    expr += model.p[c, t] * 8760 / len(Snapshots)

    return expr

model.CH_bio_target = Constraint(expr = bio_target_expr(model,'CH') <= 7.5 * 1e6)
###

# add neighbor's CO2 target
# H2 Addition: when hydrogen is burned in natural gas plants (see H2 assumptions file), we assume no CO2 emissions for the hydrogen part

print(create_impact_matrix())
impacts_dict = create_impact_matrix().to_dict()

def CO2_target_expr(model, country):
    expr = 0
    for c in Gen_IDs:
        if (c[:2] == country):
            for t in Snapshots:
                if network['generators']['carrier'][c] == "gas":
                    expr += model.p[c,t] * (1-H2_assumptions['replace_nat_gas'])* impacts_dict['tCO2/MWhel'][c.split()[-1]] * 8760/len(Snapshots)
                else:
                    expr += model.p[c,t] * impacts_dict['tCO2/MWhel'][c.split()[-1]] * 8760/len(Snapshots)
    return expr
model.DE_CO2_target = Constraint(expr = CO2_target_expr(model, 'DE') <= 356.5 *1e6 *0.05 )
model.FR_CO2_target = Constraint(expr = CO2_target_expr(model, 'FR') <= 85.5 *1e6 *0.05 )
model.IT_CO2_target = Constraint(expr = CO2_target_expr(model, 'IT') <= 119.8 *1e6 *0.05 )
model.AT_CO2_target = Constraint(expr = CO2_target_expr(model, 'AT') <= 0 )
# 1990 electricity production from https://ourworldindata.org/energy/country/germany?country=DEU~FRA~ITA#what-sources-does-the-country-get-its-electricity-from
# emission intensity of electricify generation from https://www.eea.europa.eu/en/analysis/indicators/greenhouse-gas-emission-intensity-of-1 
###

# Solve cost-optimal scenario
print('Solving model instance ...')
model,results = solve_model_mincost(model)


# make sure title of output is same of title of parent folder - Giacomo 7 Dec
parent_directory_name = os.path.basename(os.path.dirname(os.path.abspath('../run_Swiss_EXPANSE_2035_MinCost.py')))

# Store cost-optimal scenario
print('Saving run case file to scenario path ...')
save_data('output_'+parent_directory_name,0,results,model)

 