import networkx as nx
import osmnx as ox

def assign_risk_to_edges(G, risk_map, alpha=1.0):
    """
    Assigns a 'safe_weight' to each edge based on distance and risk.
    safe_weight = length * (1 + alpha * risk)
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        # approximate edge midpoint for risk lookup
        if 'geometry' in data:
            coords = list(data['geometry'].coords)
            mid = coords[len(coords)//2]
            lon, lat = mid[0], mid[1]
        else:
            lat1, lon1 = G.nodes[u]['y'], G.nodes[u]['x']
            lat2, lon2 = G.nodes[v]['y'], G.nodes[v]['x']
            lat, lon = (lat1 + lat2)/2.0, (lon1 + lon2)/2.0
            
        risk = risk_map.get_risk(lat, lon)
        length = data.get('length', 1.0)
        
        # calculate safe weight
        data['safe_weight'] = length * (1.0 + alpha * risk)
        data['risk_score'] = risk
    return G

def find_shortest_path(G, orig_lat, orig_lng, dest_lat, dest_lng):
    """Finds path minimizing distance."""
    orig_node = ox.distance.nearest_nodes(G, orig_lng, orig_lat)
    dest_node = ox.distance.nearest_nodes(G, dest_lng, dest_lat)
    
    try:
        route = nx.shortest_path(G, orig_node, dest_node, weight='length')
        return route
    except nx.NetworkXNoPath:
        return None

def find_safest_path(G, orig_lat, orig_lng, dest_lat, dest_lng):
    """Finds path minimizing safe_weight (distance + risk)."""
    orig_node = ox.distance.nearest_nodes(G, orig_lng, orig_lat)
    dest_node = ox.distance.nearest_nodes(G, dest_lng, dest_lat)
    
    try:
        route = nx.shortest_path(G, orig_node, dest_node, weight='safe_weight')
        return route
    except nx.NetworkXNoPath:
        return None
