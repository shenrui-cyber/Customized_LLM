import json
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

def visualize_graph(json_file='/storage/nvme/Customized_LLM/my_dataset/structral_data.json', output_file='/storage/nvme/Customized_LLM/kg_check.png'):
    # 1. 加载数据
    with open(json_file, 'r') as f:
        data = json.load(f)

    G = nx.DiGraph()
    events = []

    # 2. 构建图谱 (简化版用于可视化)
    print("正在构建图谱结构...")
    for i, entry in enumerate(data):
        # 定义事件节点
        event_id = f"Event_{i}"
        original = entry['original_data']
        scene = entry['scene_graph']
        timestamp = datetime.fromisoformat(original['timestamp'])
        
        # 截取简短摘要用于显示
        summary = scene['event_summary']
        short_label = summary[:10] + ".." if len(summary) > 10 else summary
        
        G.add_node(event_id, type='Event', label=short_label, color='skyblue')
        events.append({'id': event_id, 'time': timestamp, 'video': original['image_file'].split('_')[0]})

        # 定义实体节点并连接
        for p in scene['participants']:
            entity_id = p  # 简单以名字作为ID
            if not G.has_node(entity_id):
                G.add_node(entity_id, type='Entity', label=p, color='lightgreen')
            G.add_edge(event_id, entity_id)

    # 3. 添加时序边
    events.sort(key=lambda x: x['time'])
    for i in range(len(events) - 1):
        curr = events[i]
        nxt = events[i+1]
        # 如果是同一个视频源或时间很近，建立连接
        if curr['video'] == nxt['video']:
            G.add_edge(curr['id'], nxt['id'], relation='next')

    # 4. 绘图
    print(f"正在绘制 {len(G.nodes)} 个节点和 {len(G.edges)} 条边...")
    plt.figure(figsize=(15, 10))
    
    # 布局算法：Spring Layout 适合展示节点聚类关系
    pos = nx.spring_layout(G, k=0.6, iterations=50)

    # 分类绘制节点
    event_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'Event']
    entity_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'Entity']

    nx.draw_networkx_nodes(G, pos, nodelist=event_nodes, node_color='#A0CBE2', node_size=2000, label='Events')
    nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes, node_color='#98FB98', node_size=1000, label='Entities')
    
    # 绘制边和标签
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, arrows=True)
    
    # 节点标签
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=9)

    plt.title("Knowledge Graph Inspection: Events & Participants")
    plt.axis('off')
    
    # 保存
    plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
    print(f"可视化图片已保存为: {output_file}")

if __name__ == "__main__":
    visualize_graph()