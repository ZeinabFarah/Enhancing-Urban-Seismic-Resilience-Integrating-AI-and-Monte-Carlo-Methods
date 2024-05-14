
import numpy as np
import pandas as pd
import wntr
from scipy.stats import lognorm
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger('wntr.sim.hydraulics')
logger.setLevel(logging.ERROR)

class HydraulicSimulator:
    def __init__(self, epanet_file_path, scenarios_df):
        """
        Initializes the HydraulicSimulator with the water network model and earthquake scenario data.

        Parameters:
        -----------
        epanet_file_path : str
            File path to the EPANET input file.
        scenarios_df : pandas.DataFrame
            A DataFrame containing Peak Ground Acceleration (PGA) and Peak Ground Velocity (PGV) scenario data.
        """
        self.epanet_file_path = epanet_file_path
        self.scenarios_df = scenarios_df

    def run_hydraulic_simulation(self, duration, leak_start_time, leak_end_time, minimum_pressure, required_pressure):
        """
        Runs hydraulic simulations for each scenario, updates the water network model accordingly,
        and computes the serviceability of each component based on the simulation results.

        Parameters:
        -----------
        duration : int
            Duration of simulation in hours.
        leak_start_time : int
            The simulation time (in hours) when leaks start.
        leak_end_time : int
            The simulation time (in hours) when leaks stop.
        minimum_pressure : float
            Minimum pressure required in the network.
        required_pressure : float
            Required pressure in the network.

        Returns:
        --------
        dict
            A dictionary containing cumulative serviceability of each component across all scenarios.
        """
        serviceability_dict = {}
        for scenario_index, scenario_group in self.scenarios_df.groupby('scenario_index'):

            wn = wntr.network.WaterNetworkModel(self.epanet_file_path)
            wn.options.hydraulic.demand_model = 'PDD'
            wn.options.time.duration = duration
            wn.options.hydraulic.minimum_pressure = minimum_pressure
            wn.options.hydraulic.required_pressure = required_pressure

            pipelines_orifice_areas = self.apply_damage_states(wn, scenario_group)

            # Apply orifice areas for pipelines
            for pipe_id, orifice_area in pipelines_orifice_areas.items():
                pipe = wn.get_link(pipe_id)
                wn = wntr.morph.split_pipe(wn, str(pipe), f"{pipe}_A", f"Leak_{pipe}")
                leak_node = wn.get_node(f"Leak_{pipe}")
                leak_node.add_leak(wn, area=orifice_area, start_time=leak_start_time, end_time=leak_end_time)

            # Run a hydraulic simulation
            sim = wntr.sim.WNTRSimulator(wn)
            results = sim.run_sim()

            # Calculate serviceability
            serviceability = {}
            for node_name in wn.node_name_list:
                if node_name in wn.junction_name_list:
                    # For junctions, serviceability is the ratio of satisfied demand to required demand
                    satisfied_demand = results.node['demand'].loc[:, node_name].mean()
                    required_demand = wntr.metrics.expected_demand(wn)[node_name].mean()
                    if required_demand == 0:
                        # Skip the node if the required demand is zero
                        continue
                    serviceability[node_name] = satisfied_demand / required_demand

            # Accumulate serviceability from this scenario
            for node, value in serviceability.items():
                if node not in serviceability_dict:
                    serviceability_dict[node] = []
                serviceability_dict[node].append(value)

        wide_serviceability_df = pd.DataFrame.from_dict(serviceability_dict).transpose().reset_index().rename(columns={'index': 'site_id'})
        serviceability_df = wide_serviceability_df.melt(id_vars=["site_id"], var_name="scenario_index", value_name="serviceability").astype({'scenario_index': int})

        return serviceability_df

    def apply_damage_states(self, wn, scenario_group):
        """
        Applies damage states to tanks, pumps, and pipelines based on the intensity measures from the given scenarios.

        Parameters:
        -----------
        wn : wntr.network.WaterNetworkModel
            Water network model object.
        scenario_group : pandas.DataFrame
            A subset of the full scenario DataFrame filtered for a specific scenario index.

        Returns:
        --------
        tuple
            A tuple containing dictionaries for operational levels of tanks, pumps, and pipelines.
        """

        pipelines_orifice_areas = {}
        for index, row in scenario_group.iterrows():
            # Calculate orifice area for pipelines
            if row['site_id'] in wn.pipe_name_list:
                pipe_name = row['site_id']
                repair_rate = 0.000241 * row['PGV'] / 100
                link_length = wn.get_link(pipe_name).length * 0.3048  # Convert to meters
                num_damages = np.random.poisson(repair_rate * link_length)
                leaks, breaks = int(0.85 * num_damages), int(0.15 * num_damages)
                diameter = wn.get_link(pipe_name).diameter
                area = np.pi * (diameter / 2) ** 2
                orifice_area = min((0.03 * leaks + 0.2 * breaks) * area, area)
                pf = 1 - np.exp(-repair_rate * link_length)
                # Check if the pipeline fails
                if np.random.uniform() < pf:
                    pipelines_orifice_areas[row['site_id']] = orifice_area

        return pipelines_orifice_areas
