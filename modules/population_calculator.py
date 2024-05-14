
import geopandas as gpd
import pandas as pd

class PopulationCalculator:
    def __init__(self, census_file_path, water_nodes_file_path, scenarios_df, serviceability_df):
        """
        Initialize the PopulationCalculator with necessary data for population calculations.

        Parameters:
        -----------
        census_file_path : str
            File path to the shapefile containing census tract boundaries and population information.
        water_nodes_file_path : str
            File path to the shapefile containing water node locations.
        epanet_file_path : str
            File path to the EPANET input file (.inp) containing the water network model.
        scenarios_df : pandas.DataFrame
            DataFrame containing seismic scenario data including PGA, PGV, source_id, magnitude, distance, etc.
        serviceability_df : df
            Pandas df containing serviceability data for each scenario and for each node in the water network model.
        """
        self.census_file_path = census_file_path
        self.water_nodes_file_path = water_nodes_file_path
        self.scenarios_df = scenarios_df
        self.serviceability_df = serviceability_df

    def calculate_population_served(self):
        """
        Calculate the population served by each water node based on spatial analysis of census tracts and water node locations.

        Parameters:
        -----------
        census_file_path : str
            File path to the shapefile containing census tract boundaries and population information.
        water_nodes_file_path : str
            File path to the shapefile containing water node locations.

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the site_id (identifier for water nodes) and the corresponding population served by each water node.
        """
        # Load the shapefiles
        census_tracts = gpd.read_file(self.census_file_path)
        water_nodes = gpd.read_file(self.water_nodes_file_path)

        # Ensure the CRS is the same and appropriate
        census_tracts = census_tracts.to_crs(epsg=2274)  # State Plane CRS for TN is 2274
        water_nodes = water_nodes.to_crs(epsg=2274)

        # Define a fixed radius for service areas around each node in feet
        radius_in_feet = 1000
        water_nodes['service_area'] = water_nodes.geometry.buffer(radius_in_feet)

        # Perform a spatial join between the nodes' service areas and the census tracts
        service_areas = gpd.sjoin(water_nodes, census_tracts, how='left', predicate='intersects')

        # Dissolve by 'site_id' while summing the population
        node_population = service_areas.dissolve(by='site_id', aggfunc={'Population': 'sum'})
        node_population.reset_index(inplace=True)

        # Join the aggregated population data back to the water nodes GeoDataFrame
        water_nodes_with_population = water_nodes.merge(
            node_population[['site_id', 'Population']], on='site_id', how='left'
        )

        # Rename the population column to 'population_served'
        water_nodes_with_population.rename(columns={'Population': 'population_served'}, inplace=True)

        # Replace NaN with 0 for nodes that don't serve any population
        water_nodes_with_population['population_served'].fillna(0, inplace=True)

        return water_nodes_with_population[['site_id', 'population_served']]

    def calculate_population_impacted(self):
        """
        Calculate the population impacted by an earthquake at each water node based on seismic scenarios, serviceability, and expected water demand.

        Parameters:
        -----------
        wn : wntr.network.WaterNetworkModel
            Water network model object.
        census_file_path : str
            File path to the shapefile containing census tract boundaries and population information.
        water_nodes_file_path : str
            File path to the shapefile containing water node locations.

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing scenario index, site_id, source_id, magnitude, distance, PGA, PGV, population_served, serviceability, and population_impacted.
        """

        # self.serviceability_df['site_id'] = self.serviceability_df['site_id'].astype(int)
        # self.scenarios_df['site_id'] = self.scenarios_df['site_id'].astype(int)

        population_served = self.calculate_population_served()
        population_df = pd.merge(self.serviceability_df, self.scenarios_df, on=['scenario_index', 'site_id'], how='left')
        population_df = pd.merge(population_df, population_served, on='site_id', how='left')
        population_df['population_impacted'] = (1 - population_df['serviceability']) * population_df['population_served']


        population_served = self.calculate_population_served()
        population_df = pd.merge(
            self.serviceability_df[['scenario_index', 'site_id', 'serviceability']],
            population_served,
            on='site_id',
            how='left'
        )
        population_df['population_impacted'] = (1 - population_df['serviceability']) * population_df['population_served']
        population_df = population_df[['scenario_index', 'site_id', 'population_served', 'population_impacted']]

        return population_df
