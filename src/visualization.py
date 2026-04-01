import folium

def plot_routes_on_map(G, shortest_route, safest_route):
    """
    Plots the shortest (red) and safest (green) routes on a Folium map.
    """
    if safest_route:
        start_node = G.nodes[safest_route[0]]
    elif shortest_route:
        start_node = G.nodes[shortest_route[0]]
    else:
        # Default view (approx LA/Santa Monica)
        return folium.Map(location=[34.015, -118.49], zoom_start=13)

    m = folium.Map(location=[start_node['y'], start_node['x']], zoom_start=13)
    
    # Plot Shortest Route
    if shortest_route:
        locs = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in shortest_route]
        folium.PolyLine(locs, color='red', weight=4, opacity=0.7, popup="Shortest Route").add_to(m)
        
        # Start and end markers
        folium.Marker(locs[0], popup="Origin", icon=folium.Icon(color='blue')).add_to(m)
        folium.Marker(locs[-1], popup="Destination", icon=folium.Icon(color='red')).add_to(m)
    
    # Plot Safest Route
    if safest_route:
        locs = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in safest_route]
        # Offset slightly if they overlap, but Folium handles drawing order.
        folium.PolyLine(locs, color='green', weight=6, opacity=0.9, popup="Safest Route").add_to(m)
        if not shortest_route:
            folium.Marker(locs[0], popup="Origin", icon=folium.Icon(color='blue')).add_to(m)
            folium.Marker(locs[-1], popup="Destination", icon=folium.Icon(color='green')).add_to(m)
        
    return m
