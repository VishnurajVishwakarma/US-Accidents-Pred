import numpy as np

def create_grid(lat_min, lat_max, lng_min, lng_max, grid_size=0.01):
    """Creates a spatial grid for risk binning."""
    lats = np.arange(lat_min, lat_max, grid_size)
    lngs = np.arange(lng_min, lng_max, grid_size)
    return lats, lngs

def get_grid_index(lat, lng, lat_min, lng_min, grid_size=0.01):
    """Returns grid indices for a given lat/lng."""
    lat_idx = int((lat - lat_min) / grid_size)
    lng_idx = int((lng - lng_min) / grid_size)
    return lat_idx, lng_idx

class RiskMap:
    def __init__(self, grid_size=0.01):
        self.grid_size = grid_size
        self.risk_grid = {} # Map (lat_idx, lng_idx) -> risk_score
        self.lat_min = 0
        self.lng_min = 0
        
    def build_from_predictions(self, df):
        """Populates the risk map from a dataframe containing Start_Lat, Start_Lng, risk_score."""
        print("Building risk map...")
        self.lat_min = df['Start_Lat'].min()
        self.lat_max = df['Start_Lat'].max()
        self.lng_min = df['Start_Lng'].min()
        self.lng_max = df['Start_Lng'].max()
        
        # Aggregate risks per grid
        grid_risks = {}
        for _, row in df.iterrows():
            idx = get_grid_index(row['Start_Lat'], row['Start_Lng'], self.lat_min, self.lng_min, self.grid_size)
            if idx not in grid_risks:
                grid_risks[idx] = []
            grid_risks[idx].append(row['risk_score'])
            
        for k, v in grid_risks.items():
            self.risk_grid[k] = np.mean(v)
            
    def get_risk(self, lat, lng):
        """Returns risk score for a lat/lng, or a default small risk if not found."""
        try:
            idx = get_grid_index(lat, lng, self.lat_min, self.lng_min, self.grid_size)
            return self.risk_grid.get(idx, 0.05) # base minimal risk
        except Exception:
            return 0.05
