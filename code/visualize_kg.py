import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

def visualize_structured_graph_v2(file_path, output_file='/storage/nvme/Customized_LLM/structured_viz.png'):
    try:
        G = nx.read_gml(file_path)
        print(f"Read graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # --- 1. Node Classification & Data Extraction ---
    super_nodes = []
    events = []
    entities = []
    
    node_type_map = {}
    
    for n, data in G.nodes(data=True):
        n_type = data.get('type', 'Unknown')
        node_type_map[n] = n_type
        
        if n_type == 'SuperNode':
            timestamp = data.get('timestamp', '0')
            super_nodes.append((n, timestamp))
        elif n_type == 'Event':
            events.append(n)
        elif n_type == 'Entity':
            entities.append(n)

    # Sort SuperNodes by timestamp
    super_nodes.sort(key=lambda x: x[1])
    sorted_super_node_ids = [n[0] for n in super_nodes]

    # --- 2. Custom Layout Calculation ---
    pos = {}
    
    # A. SuperNodes (Center Spine)
    # Increased spacing for larger nodes
    center_y = 0
    spacing_x = 8.0  # Increased from 4.0
    start_x = -((len(sorted_super_node_ids) - 1) * spacing_x) / 2
    
    for i, node_id in enumerate(sorted_super_node_ids):
        pos[node_id] = np.array([start_x + i * spacing_x, center_y])

    # B. Events (Orbit)
    # Map Events to SuperNodes
    scene_event_map = {sn: [] for sn in sorted_super_node_ids}
    
    for evt in events:
        found_scene = False
        # Check outgoing edges (Event -> SuperNode)
        for neighbor in G.neighbors(evt):
            if neighbor in sorted_super_node_ids:
                scene_event_map[neighbor].append(evt)
                found_scene = True
                break
        # Check incoming edges (SuperNode -> Event)
        if not found_scene:
            for pred in G.predecessors(evt):
                if pred in sorted_super_node_ids:
                    scene_event_map[pred].append(evt)
                    break
    
    # Arrange Events around SuperNodes
    radius_event = 3.0 # Increased from 1.2
    
    for sn_id, evt_list in scene_event_map.items():
        if not evt_list: continue
        sn_pos = pos[sn_id]
        num_evts = len(evt_list)
        # Distribute over a full circle or semi-circle? Full circle is usually better for orbit.
        # But to avoid overlapping with horizontal spine, maybe avoid 0 and 180 degrees if possible?
        # Let's stick to uniform distribution for now.
        angle_step = 2 * np.pi / num_evts
        start_angle = np.pi / 2 
        
        for i, evt in enumerate(evt_list):
            theta = start_angle + i * angle_step
            dx = radius_event * np.cos(theta)
            dy = radius_event * np.sin(theta)
            pos[evt] = sn_pos + np.array([dx, dy])

    # C. Entities (Periphery)
    evt_entity_map = {} 
    
    for ent in entities:
        connected_evts = []
        all_neighbors = list(G.predecessors(ent)) + list(G.successors(ent))
        for nb in all_neighbors:
            if nb in events:
                connected_evts.append(nb)
        
        if connected_evts:
            valid_positions = [pos[e] for e in connected_evts if e in pos]
            if valid_positions:
                avg_pos = np.mean(valid_positions, axis=0)
            else:
                avg_pos = np.array([start_x, 0])

            # Determine direction (Up or Down)
            direction = 1 if avg_pos[1] >= 0 else -1
            
            # Increased offset for larger nodes and larger orbit
            base_offset_y = 5.0 # Increased from 2.5
            
            # If the calculated position is too close to the spine (y=0), push it out
            current_y = avg_pos[1]
            if abs(current_y) < radius_event + 1.0:
                 # Ensure it's outside the event orbit
                 target_y = (radius_event + 2.0) * direction
            else:
                 target_y = current_y + (2.0 * direction) # Push out further
            
            # Apply layout
            pos[ent] = np.array([avg_pos[0], target_y])
            
            # Add separation jitter
            pos[ent] += np.random.rand(2) * 0.8 # Increased jitter
        else:
            # Isolated entities
            pos[ent] = np.array([start_x + np.random.rand() * 10, -6])

    # --- 3. Drawing ---
    plt.figure(figsize=(24, 14)) # Increased canvas size
    
    c_super = '#e31a1c' 
    c_event = '#1f78b4' 
    c_entity = '#33a02c' 
    
    # Draw Edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', arrows=True, arrowsize=15, width=1.0)
    
    # Draw Nodes with increased sizes
    # SuperNodes: 2500
    nx.draw_networkx_nodes(G, pos, nodelist=sorted_super_node_ids, 
                           node_color=c_super, node_size=2500, label='SuperNode')
    
    # Events: 1000
    nx.draw_networkx_nodes(G, pos, nodelist=events, 
                           node_color=c_event, node_size=1000, label='Event')
    
    # Entities: 1800
    nx.draw_networkx_nodes(G, pos, nodelist=entities, 
                           node_color=c_entity, node_size=1800, label='Entity')
    
    # Labels
    # SuperNode Labels (Inside)
    nx.draw_networkx_labels(G, pos, labels={n: n for n in sorted_super_node_ids}, 
                            font_color='white', font_weight='bold', font_size=10)
    
    # Entity Labels (Below/Above or Inside if fits) - Let's put them inside for clarity with large nodes
    entity_labels = {n: G.nodes[n].get('name', n) for n in entities}
    nx.draw_networkx_labels(G, pos, labels=entity_labels, font_size=11, font_weight='bold', font_color='white') # White text on Green node
    
    # Event Labels? Maybe indices or small text. Let's skip to keep clean, or maybe very small.
    # User didn't ask for event labels explicitly, but larger nodes might look empty. 
    # Let's leave them empty to emphasize structure, or add summary if short.
    
    # # Time Axis
    # plt.arrow(start_x - 2, -2.5, (len(super_nodes) * spacing_x) + 2, 0, 
    #           head_width=0.4, head_length=0.8, fc='#dddddd', ec='#dddddd', zorder=0)
    # plt.text(start_x, -3.0, "Time Progression ->", color='gray', fontsize=14, weight='bold')

    # Legend with increased spacing
    # Creating custom handles
    legend_handles = [
        mlines.Line2D([], [], color='white', marker='o', markersize=18, markerfacecolor=c_super, label='SuperNode (Scene)'),
        mlines.Line2D([], [], color='white', marker='o', markersize=14, markerfacecolor=c_event, label='Event'),
        mlines.Line2D([], [], color='white', marker='o', markersize=16, markerfacecolor=c_entity, label='Entity')
    ]
    
    # labelspacing controls vertical space between legend entries
    plt.legend(handles=legend_handles, loc='upper right', title="Node Types", fontsize=14, title_fontsize=16, labelspacing=2.0)

    plt.title("Knowledge Graph: Scenes -> Events -> Entities", fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print("Visualization saved.")

visualize_structured_graph_v2('/storage/nvme/Customized_LLM/knowledge_graph.gml')