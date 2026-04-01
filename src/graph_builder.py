import osmnx as ox
import networkx as nx
import os

def download_graph(place_name="Santa Monica, California, USA", network_type='drive'):
    print(f"Downloading road network for {place_name}...")
    try:
        # Simplification might take time, we do it to keep graph manageable
        G = ox.graph_from_place(place_name, network_type=network_type, simplify=True)
        return G
    except Exception as e:
        print(f"Failed to download graph for {place_name}: {e}")
        # fallback to a tiny bounding box if place name fails
        point = (34.015, -118.49) # Approx Santa Monica
        print(f"Fallback: downloading 5km radius around {point}")
        G = ox.graph_from_point(point, dist=5000, network_type=network_type, simplify=True)
        return G

def save_graph(G, filepath):
    # save as graphml
    ox.save_graphml(G, filepath)
    print(f"Graph saved to {filepath}")

def load_graph(filepath):
    print(f"Loading graph from {filepath}...")
    G = ox.load_graphml(filepath)
    return G

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    graph_path = 'data/la_network.graphml'
    if not os.path.exists(graph_path):
        G = download_graph() 
        save_graph(G, graph_path)
    else:
        print("Graph already exists.")
