
import numpy as np
import json

class SeismicSourceSimulator:
    def __init__(self, source_data_path, num_faults, m_min, m_max, lat_range, lon_range, depth_range, dip_range, strike_range):
        self.source_data_path  = source_data_path
        self.num_faults = num_faults
        self.m_min = m_min
        self.m_max = m_max
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.depth_range = depth_range
        self.dip_range = dip_range
        self.strike_range = strike_range

    def generate_seismic_source(self):
        return {
            "M_min": np.random.uniform(self.m_min, self.m_min + 1),
            "M_max": np.random.uniform(self.m_max, self.m_max - 1),
            "nu": 0.001,  # Fixed value based on typical seismicity rate studies
            "lat": np.random.uniform(*self.lat_range),
            "lon": np.random.uniform(*self.lon_range),
            "depth": np.random.uniform(*self.depth_range),
            "dip": np.random.uniform(*self.dip_range),
            "strike": np.random.uniform(*self.strike_range),
            "mechanism": np.random.choice(["SS", "NS", "RS"]),
            "event_type": np.random.choice(["interface", "intraslab"])
        }

    def generate_all_sources(self):
        fault_data = {}
        for i in range(1, self.num_faults + 1):
            fault_name = f"Fault {i}"
            fault_data[fault_name] = self.generate_seismic_source()
        return fault_data

    def save_to_json(self, fault_data, filename=None):
        if filename is None:
            filename = self.source_data_path
        with open(filename, 'w') as outfile:
            json.dump(fault_data, outfile, indent=4)
