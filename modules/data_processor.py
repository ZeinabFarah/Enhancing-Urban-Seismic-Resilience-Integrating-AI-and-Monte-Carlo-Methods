
import pandas as pd
import wntr
import numpy as np

class DataProcessor:
    def __init__(self, epanet_file_path, scenarios_df, serviceability_df, population_df):
        self.wn = wntr.network.WaterNetworkModel(epanet_file_path)
        self.scenarios_df = scenarios_df
        self.serviceability_df = serviceability_df
        self.population_df = population_df

    def hydraulic_network_data(self):
        sim = wntr.sim.WNTRSimulator(self.wn)
        results = sim.run_sim()

        data = []
        for node_name in self.wn.node_name_list:
            node = self.wn.get_node(node_name)
            connected_pipes = self.wn.get_links_for_node(node_name)
            incoming = []
            outgoing = []
            pipe_diameters = []
            pipe_lengths = []
            pipe_materials = []
            
            for pipe_id in connected_pipes:
                component = self.wn.get_link(pipe_id) 
                if component.end_node == node:
                    incoming.append(pipe_id)
                if component.start_node == node:
                    outgoing.append(pipe_id)

                # Check the type of the component and access only relevant attributes
                if hasattr(component, 'diameter'):  # Check if the 'diameter' attribute exists
                    pipe_diameters.append(component.diameter)
                    pipe_lengths.append(component.length)
                    # pipe_materials.append(getattr(component, 'material', 'N/A'))
                else:
                    # Handle other types of components (like pumps)
                    pipe_diameters.append(0)
                    pipe_lengths.append(0)
                    # pipe_materials.append(None)
            
            # material_mode = max(set(pipe_materials), key=pipe_materials.count) if pipe_materials else None
            
            node_data = {
                'site_id': node_name,
                'Total Incoming Pipe Count': len(incoming),
                'Total Outgoing Pipe Count': len(outgoing),
                'Average Diameter of Connecting Pipes': np.mean(pipe_diameters) if pipe_diameters else 0,
                'Sum of Pipe Lengths': sum(pipe_lengths),
                # 'Material Mode': material_mode,
                'Maximum Pressure': max(results.node['pressure'][node_name]),
                'Minimum Pressure': min(results.node['pressure'][node_name])
            }
            data.append(node_data)
        return pd.DataFrame(data)

    def generate_combined_dataframe(self):
        # Get hydraulic network data
        network_df = self.hydraulic_network_data()

        combined_df = pd.merge(self.serviceability_df, self.scenarios_df, on=['scenario_index', 'site_id'], how='left')
        combined_df = pd.merge(combined_df, self.population_df, on=['scenario_index', 'site_id'], how='left')

        final_df = pd.merge(combined_df, network_df, on='site_id', how='left')
        final_df = final_df[[
            'scenario_index', 'site_id', 'source_id', 'magnitude', 'distance', 'PGA', 'PGV',
            'population_served', 'population_impacted', 'serviceability',
            'Total Incoming Pipe Count', 'Total Outgoing Pipe Count',
            'Average Diameter of Connecting Pipes', 'Sum of Pipe Lengths',
            'Maximum Pressure', 'Minimum Pressure'
        ]]

        return final_df
