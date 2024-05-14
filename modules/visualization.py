
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import geopandas as gpd
import seaborn as sns
from scipy.stats import linregress
import contextily as ctx

class Visualization:
    def __init__(self, setup, scenarios_dict, return_period):
        """
        Initializes the SeismicHazardVisualization with necessary parameters for visualization.

        Parameters:
        -----------
        setup : object
            An instance of setup class containing site and source data.
        scenarios_dict : dict
            A dictionary of intensity measures for each site and scenario.
        return_period : int
            The return period for seismic hazard analysis in years.
        """
        self.setup = setup
        self.scenarios_dict = scenarios_dict
        self.return_period = return_period

    def prepare_im_data(self, exceedance_probability, ground_motion_type):
        """
        Prepare the data for plotting based on the specified exceedance probability and ground motion type.

        Parameters:
        -----------
        exceedance_probability : float
            The exceedance probability for which to prepare the data.
        ground_motion_type : str
            The type of ground motion ('PGA' or 'PGV') to prepare data for.

        Returns:
        --------
        list
            A list of tuples containing latitude, longitude, and exceedance value for each site.
        """
        annual_exceedance_probability = 1 - (1 - exceedance_probability)**(1/self.return_period)

        processed_data = []
        for site_id in self.setup.site_data['id']:
            # Filter intensity measures for the current site and ground motion type
            site_im_values = [im[ground_motion_type] for (idx, id, *rest), im in self.scenarios_dict.items()
                              if id == site_id and ground_motion_type in im]

            exceedance_value = np.percentile(site_im_values, 100 * (1 - annual_exceedance_probability))

            try:
                row = self.setup.site_data[self.setup.site_data['id'] == site_id].iloc[0]
                lat, lon = row['latitude'], row['longitude']
                processed_data.append((lat, lon, exceedance_value))
            except IndexError:
                print(f"Site ID {site_id} not found in site_data")

        return processed_data

    def plot_contour_map(self, exceedance_probability, ground_motion_type):
        """
        Plot a contour map representing seismic hazard based on the exceedance probability.

        Parameters:
        -----------
        exceedance_probability : float
            The exceedance probability for which to generate the contour map.
        """
        # Prepare the data
        data = self.prepare_im_data(exceedance_probability, ground_motion_type)
        lats, lons, ims = zip(*data)

        # Generate a grid to interpolate onto
        grid_lons, grid_lats = np.meshgrid(np.linspace(min(lons), max(lons), 100),
                                           np.linspace(min(lats), max(lats), 100))

        # Interpolate the data onto the grid
        grid_ims = griddata((lons, lats), ims, (grid_lons, grid_lats), method='cubic')

        # Create the contour plot
        plt.figure(figsize=(10, 6))
        contour = plt.contourf(grid_lons, grid_lats, grid_ims, levels=100, cmap='viridis')
        plt.colorbar(contour)

        # for source_id, source_info in self.setup.source_data.items():
        #     plt.plot(source_info['lon'], source_info['lat'], '*', color='yellow', markersize=15, label=f'{source_id}')

        # Annotations and titles
        plt.title(f"Seismic Hazard Contour Map - {self.return_period} years, {exceedance_probability*100}% Exceedance")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.show()

    def plot_population_impacts(self, dataframe, impact_column, color_by=None):
        """
        Plots the relationship between a specified impact column and population impacted, with optional color coding and annotations.
        
        Parameters:
        -----------
        dataframe : DataFrame
            The DataFrame containing the data to plot.
        impact_column : str
            The column in the dataframe to plot against population impacted.
        color_by : str, optional
            The column by which to color the points. Default is None, which uses a single color.
        
        Returns:
        --------
        A plot showing the relationship between the specified impact and the population impacted, including a regression line and annotations for outliers.
        """
        plt.figure(figsize=(12, 8))
        
        if color_by:
            # Create a scatter plot colored by a specified column
            scatter = plt.scatter(dataframe[impact_column], dataframe['population_impacted'],
                                  c=dataframe[color_by], cmap='viridis', alpha=0.6, edgecolors='w', linewidth=0.5)
            plt.colorbar(scatter, label=color_by)
        else:
            # Regular scatter plot
            plt.scatter(dataframe[impact_column], dataframe['population_impacted'],
                        color='blue', alpha=0.6, edgecolors='w', linewidth=0.5)

        # Fit and plot a regression line
        slope, intercept, r_value, p_value, std_err = linregress(dataframe[impact_column], dataframe['population_impacted'])
        line = slope * dataframe[impact_column] + intercept
        plt.plot(dataframe[impact_column], line, 'r', label=f'y={slope:.2f}x+{intercept:.2f} (R²={r_value**2:.2f})')
        
        plt.xlabel(impact_column.capitalize())
        plt.ylabel('Population Impacted')
        plt.title(f'Relationship Between {impact_column.capitalize()} and Population Impacted')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_serviceability_map(self, tracts_path, nodes_path, pipes_path, serviceability_df, scenario_index):
        """
        Plots a map showing the serviceability of water distribution nodes for a specified scenario post-earthquake.

        Parameters:
        -----------
        tracts_path : str
            Path to the census tract shapefile.
        nodes_path : str
            Path to the water distribution nodes shapefile.
        pipes_path : str
            Path to the water distribution pipes shapefile.
        serviceability_df : DataFrame
            DataFrame containing serviceability data along with site_ids and scenario indices.
        scenario_index : int
            The specific scenario index to visualize.
        """
        # Load the shapefiles
        tracts = gpd.read_file(tracts_path)
        nodes = gpd.read_file(nodes_path)
        pipes = gpd.read_file(pipes_path)

        # Filter the serviceability DataFrame for the selected scenario
        scenario_data = serviceability_df[serviceability_df['scenario_index'] == scenario_index]

        # Merge the nodes GeoDataFrame with the serviceability data
        nodes = nodes.merge(scenario_data, left_on='site_id', right_on='site_id')

        # Ensure CRS compatibility and add a basemap
        tracts = tracts.to_crs(epsg=4326)
        nodes = nodes.to_crs(epsg=4326)
        pipes = pipes.to_crs(epsg=4326)

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        tracts.plot(ax=ax, color='none', edgecolor='gray', alpha=0.4) 
        pipes.plot(ax=ax, color='black', linewidth=0.5) 

        # Plot nodes with serviceability levels
        nodes.plot(ax=ax, column='serviceability', cmap='coolwarm', legend=True, legend_kwds={'label': "Serviceability Level", 'orientation': "horizontal"})

        # Add a basemap for geographical context
        ctx.add_basemap(ax, crs=nodes.crs.to_string(), source=ctx.providers.CartoDB.Positron)

        # Set spatial limits based on nodes' extent
        ax.set_xlim(nodes.total_bounds[[0, 2]])
        ax.set_ylim(nodes.total_bounds[[1, 3]])

        ax.set_title(f'Serviceability Map for Scenario {scenario_index}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        plt.show()

    def plot_infrastructure_map(self, tracts_path, nodes_path, pipes_path):
        """
        Plots census tracts with population data and water distribution system (nodes and pipes) overlaid,
        including a vertical color bar for population.

        Parameters:
        -----------
        tracts_path : str
            Path to the census tract shapefile.
        nodes_path : str
            Path to the water distribution nodes shapefile.
        pipes_path : str
            Path to the water distribution pipes shapefile.
        """
        # Load the shapefiles
        tracts = gpd.read_file(tracts_path)
        nodes = gpd.read_file(nodes_path)
        pipes = gpd.read_file(pipes_path)

        # Ensure that the CRS matches for all layers and is appropriate for adding basemaps (usually WGS84 - EPSG:4326)
        tracts = tracts.to_crs(epsg=4326)
        nodes = nodes.to_crs(epsg=4326)
        pipes = pipes.to_crs(epsg=4326)

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot the tracts as a base layer colored by population
        tract_plot = tracts.plot(ax=ax, column='Population', cmap='OrRd', alpha=0.6, legend=True,
                    legend_kwds={'label': "Population by Census Tract", 'orientation': "vertical"})

        # Plot pipes
        pipes.plot(ax=ax, color='blue', linewidth=1, label='Water Pipes')

        # Plot nodes
        nodes.plot(ax=ax, marker='o', color='red', markersize=10, label='Distribution Nodes')

        # Add a basemap for geographical context
        ctx.add_basemap(ax, crs=tracts.crs.to_string(), source=ctx.providers.CartoDB.Positron)

        # Setting spatial limits based on tracts' bounds to avoid stray elements outside the area of interest
        ax.set_xlim(tracts.total_bounds[[0, 2]])
        ax.set_ylim(tracts.total_bounds[[1, 3]])

        # Adding labels, title, and legend
        ax.set_title('Census Tracts and Water Distribution System Overlay')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend()

        plt.show()

    def plot_feature_importances(self, importances, feature_names):
        """
        Plots a scatter plot of feature importances from a random forest model.

        Parameters:
        -----------
        importances : array-like
            An array of feature importance scores from the random forest model.
        feature_names : list
            A list of names corresponding to the features used in the model.
        """
        # Sorting the features by importance
        indices = np.argsort(importances)

        # Create the plot
        plt.figure(figsize=(10, 8))
        plt.title('Feature Importances in Random Forest Model')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()
