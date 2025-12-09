import json
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import networkx as nx
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class KnowledgeGraphBuilder:
    def __init__(self):
        # 初始化图结构
        self.graph = nx.DiGraph()
        # 加载向量化模型 (用于保留主观/隐晦情感内容 以及 实体对齐)
        print("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        # 实体映射表 (用于初步实体对齐 )
        self.entity_registry = {} 

    def load_data(self, json_file):
        with open(json_file, 'r') as f:
            return json.load(f)

    def vectorize_text(self, text):
        """将原始文档/摘要向量化，作为属性保存 """
        # embedder.encode 默认返回 numpy array
        return self.embedder.encode(text)

    def align_entity_initial(self, participant_name, image_id):
        """
        初步实体对齐逻辑 (基于精确字符串匹配):
        先做简单的字符串归一化和注册，后续会有向量合并步骤进行深度对齐。
        """
        # 简单清洗，去除首尾空格
        participant_name = participant_name.strip()
        
        if participant_name not in self.entity_registry:
            self.entity_registry[participant_name] = {
                "id": f"ENT_{len(self.entity_registry)}",
                "name": participant_name,
                "appearances": []
            }
        
        # 记录出现过的图片ID (去重)
        if image_id not in self.entity_registry[participant_name]["appearances"]:
            self.entity_registry[participant_name]["appearances"].append(image_id)
            
        return self.entity_registry[participant_name]["id"]

    def build_initial_graph(self, data):
        """第一阶段：构建基础图谱（Event 和 Entity 节点，包含 Relation 处理）"""
        print("Building initial graph...")
        
        # --- 遍历数据构建节点 ---
        for entry in data:
            original = entry['original_data']
            scene = entry['scene_graph']
            
            # 1. 生成唯一事件ID
            event_id = original['image_file']
            
            # 2. 处理时间戳
            try:
                timestamp = datetime.fromisoformat(original['timestamp'])
            except ValueError:
                timestamp = datetime.strptime(original['timestamp'], "%Y-%m-%d %H:%M:%S")

            # 3. 向量化 (保留情感氛围)
            content_to_embed = f"{original['caption']} Atmosphere: {', '.join(scene['atmosphere'])}"
            vector = self.vectorize_text(content_to_embed)
            
            # 4. 创建 Event 节点
            node_attrs = {
                "type": "Event",
                "timestamp": timestamp,
                "location_str": scene['location'],
                "geo": [float(original['latitude']), float(original['longitude'])],
                "summary": scene['event_summary'],
                "vector": vector, 
                "atmosphere": scene['atmosphere']
            }
            
            self.graph.add_node(event_id, **node_attrs)
            
            # 用于当前 Event 内实体去重，防止同一张图对同一实体重复添加 Event->Entity 边
            seen_entity_ids = set()

            # --- 辅助函数：处理单个实体的添加与连接 ---
            def process_entity(name):
                ent_id = self.align_entity_initial(name, event_id)
                
                # 如果节点不存在，添加 Entity 节点
                if not self.graph.has_node(ent_id):
                    self.graph.add_node(ent_id, type="Entity", name=name)
                
                # 如果在这个事件中还没连接过该实体，则建立 Event -> Entity 连接
                if ent_id not in seen_entity_ids:
                    self.graph.add_edge(event_id, ent_id, relation="has_participant")
                    seen_entity_ids.add(ent_id)
                
                return ent_id

            # 5. 处理参与者 (Participants)
            if 'participants' in scene:
                for p in scene['participants']:
                    process_entity(p)

            # 6. [新增逻辑] 处理关系 (Relations) 中的 Subject 和 Object
            # 提取 relations 中的实体，并建立实体间的交互边
            if 'relations' in scene:
                for rel in scene['relations']:
                    subj_name = rel.get('subject')
                    obj_name = rel.get('object')
                    action = rel.get('action')

                    if subj_name and obj_name:
                        # 处理 Subject 实体
                        subj_id = process_entity(subj_name)
                        # 处理 Object 实体
                        obj_id = process_entity(obj_name)

                        # 建立实体间的关系边: Subject --[action]--> Object
                        # key=event_id 确保如果同一对实体在不同事件中有相同动作，不会覆盖，而是作为多重边存在（NetworkX DiGraph默认会覆盖，这里用 add_edge 属性区分）
                        # 这里我们简化处理，添加 event_ref 属性说明这个关系发生在哪个事件
                        self.graph.add_edge(subj_id, obj_id, 
                                          relation=action, 
                                          event_ref=event_id,
                                          timestamp=timestamp) # 可选：带上时间戳

        print(f"Initial graph built: {len(self.graph.nodes)} nodes (Events + Entities).")

    def merge_semantic_entities(self, threshold=0.85):
        """
        功能 1：基于向量相似度合并 Entity 节点
        解决 'tam-tam' vs 'tam-tams', 'boy' vs 'child' 等问题
        """
        print(f"Merging semantic entities (Threshold: {threshold})...")
        
        # 1. 提取所有 Entity 节点
        entity_nodes = []
        entity_names = []
        
        for n, attr in self.graph.nodes(data=True):
            if attr.get('type') == 'Entity':
                entity_nodes.append(n)
                entity_names.append(attr.get('name', ''))
        
        if not entity_names:
            return

        # 2. 批量计算名字的向量
        embeddings = self.embedder.encode(entity_names)
        
        # 3. 计算相似度矩阵
        similarity_matrix = cosine_similarity(embeddings)
        
        # 4. 构建相似度图用于查找连通分量 (合并组)
        sim_graph = nx.Graph()
        sim_graph.add_nodes_from(entity_nodes)
        
        num_entities = len(entity_nodes)
        for i in range(num_entities):
            for j in range(i + 1, num_entities):
                if similarity_matrix[i][j] > threshold:
                    sim_graph.add_edge(entity_nodes[i], entity_nodes[j])
        
        # 5. 执行合并
        merge_count = 0
        for component in nx.connected_components(sim_graph):
            if len(component) > 1:
                nodes_list = list(component)
                # 策略：保留名字最短的作为主节点 (通常是词根)
                nodes_list.sort(key=lambda x: len(self.graph.nodes[x].get('name', '')))
                target_node = nodes_list[0]
                
                # 更新目标节点的名字（有时候保留较长的更有描述性，但为了统一起见，这里维持原逻辑）
                target_name = self.graph.nodes[target_node]['name']
                
                for source_node in nodes_list[1:]:
                    # source_name = self.graph.nodes[source_node]['name']
                    # 将 source_node 的所有边转移到 target_node
                    nx.contracted_nodes(self.graph, target_node, source_node, self_loops=False, copy=False)
                    merge_count += 1
                    
        print(f"Merged {merge_count} duplicate entities.")

    def create_spatiotemporal_supernodes(self):
        """
        功能 2：将时空接近的事件连接到“超节点”，并按时间顺序连接超节点
        """
        print("Creating spatiotemporal SuperNodes...")
        
        # 1. 收集现有 Event 节点数据
        events = []
        # 过滤出 Event 类型的节点
        event_node_ids = [n for n, attr in self.graph.nodes(data=True) if attr.get('type') == 'Event']
        
        for nid in event_node_ids:
            attr = self.graph.nodes[nid]
            ts = attr['timestamp'] 
            
            # 生成位置签名
            geo = attr.get('geo')
            if geo and isinstance(geo, list):
                loc_sig = f"{round(geo[0], 4)}_{round(geo[1], 4)}"
            else:
                loc_sig = attr.get('location_str', 'unknown')
                
            events.append({'id': nid, 'time': ts, 'loc': loc_sig})
            
        # 2. 排序与聚类
        events.sort(key=lambda x: (x['loc'], x['time']))
        
        clusters = []
        if events:
            current_cluster = [events[0]]
            for i in range(1, len(events)):
                prev = current_cluster[-1]
                curr = events[i]
                
                time_diff = (curr['time'] - prev['time']).total_seconds()
                
                # 聚类逻辑：地点相同 且 时间间隔 < 1小时
                if curr['loc'] == prev['loc'] and 0 <= time_diff < 3600:
                    current_cluster.append(curr)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [curr]
            clusters.append(current_cluster)

        # 3. 创建 SuperNodes
        super_nodes_list = []
        
        # 移除原有的事件间时序边 (如果存在)
        edges_to_remove = []
        for u, v, attr in self.graph.edges(data=True):
            if attr.get('relation') in ['same_video_sequence', 'next_step']:
                if self.graph.nodes[u].get('type') == 'Event':
                    edges_to_remove.append((u, v))
        self.graph.remove_edges_from(edges_to_remove)
        
        print(f"Generated {len(clusters)} SuperNodes.")

        for idx, cluster in enumerate(clusters):
            # 创建 SuperNode ID
            super_id = f"SCENE_{idx}"
            start_time = cluster[0]['time']
            # end_time = cluster[-1]['time']
            
            # 获取位置描述
            first_event_node = self.graph.nodes[cluster[0]['id']]
            loc_str = first_event_node.get('location_str', 'Unknown')
            
            # 添加 SuperNode
            self.graph.add_node(super_id,
                                type="SuperNode",
                                timestamp=start_time,
                                summary=f"Sequence at {loc_str}",
                                event_count=len(cluster))
            
            super_nodes_list.append({'id': super_id, 'time': start_time})
            
            # 连接 Event -> SuperNode
            for event in cluster:
                self.graph.add_edge(event['id'], super_id, relation="belongs_to")

        # 4. 连接超节点 (Sequence Flow)
        super_nodes_list.sort(key=lambda x: x['time'])
        for i in range(len(super_nodes_list) - 1):
            u = super_nodes_list[i]['id']
            v = super_nodes_list[i+1]['id']
            time_diff = (super_nodes_list[i+1]['time'] - super_nodes_list[i]['time']).total_seconds()
            
            self.graph.add_edge(u, v, relation="sequence_flow", time_diff=time_diff)

    def process_pipeline(self, data):
        """执行完整的构建流程"""
        # 1. 基础构建 (含 Relations 实体提取)
        self.build_initial_graph(data)
        
        # 2. 功能集成：语义实体合并
        # 因为现在加入了 Object (如 snowboard, camera 等)，合并功能会更加重要
        self.merge_semantic_entities(threshold=0.85)
        
        # 3. 功能集成：时空超节点构建
        self.create_spatiotemporal_supernodes()

    def save_graph(self, filename="/storage/nvme/Customized_LLM/knowledge_graph.gml"):
        # 1. 创建副本，避免修改原图
        G_to_save = self.graph.copy()
        
        print("Sanitizing graph data for GML export...")

        # --- 清洗节点属性 ---
        for n, data in G_to_save.nodes(data=True):
            # 删除无法保存的向量数据
            if 'vector' in data:
                del data['vector']
            
            # 遍历剩余属性进行格式转换
            for k, v in list(data.items()):
                # 处理时间格式
                if isinstance(v, datetime):
                    data[k] = v.strftime("%Y-%m-%d %H:%M:%S")
                # 处理列表 (GML只支持纯数字列表，对于字符串列表如atmosphere，必须转为字符串)
                elif isinstance(v, list):
                    # 如果是 geo (纯数字列表)，保留
                    if k == 'geo' and all(isinstance(x, (int, float)) for x in v):
                        continue
                    # 其他列表 (如 atmosphere) 转为逗号分隔字符串
                    else:
                        data[k] = ", ".join(map(str, v))
                # 关键修复：处理元组或其他非基本类型 (这就是报错 ('ENT_10', 'ENT_9') 的根源)
                elif isinstance(v, tuple):
                    data[k] = str(v)
                # 处理 None
                elif v is None:
                    data[k] = "null"
                # 兜底：如果不是 int/float/str/bool，统统转字符串
                elif not isinstance(v, (int, float, str, bool)):
                    data[k] = str(v)

        # --- 清洗边属性 ---
        # 注意：这里去掉了 keys=True，修复了上一个报错
        for u, v, data in G_to_save.edges(data=True):
            for k, v in list(data.items()):
                # 处理时间格式
                if isinstance(v, datetime):
                    data[k] = v.strftime("%Y-%m-%d %H:%M:%S")
                # 关键修复：GML 边属性极其严格，遇到任何复杂结构(元组/列表)直接转字符串
                elif not isinstance(v, (int, float, str, bool)):
                    data[k] = str(v)

        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        try:
            # 写入 GML
            nx.write_gml(G_to_save, filename)
            print(f"Graph saved successfully to {filename}")
            print(f"Final Stats: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges.")
        except Exception as e:
            print(f"Error saving GML: {e}")
            # 如果依然报错，尝试打印出有问题的节点以便调试（通常上面清洗后不会再报错）
            import traceback
            traceback.print_exc()
            
# --- 执行构建 ---
if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()
    
    # 路径配置
    json_path = '/storage/nvme/Customized_LLM/my_dataset/scene_graph.json'
    output_path = '/storage/nvme/Customized_LLM/knowledge_graph.gml'
    
    if os.path.exists(json_path):
        data = builder.load_data(json_path)
        builder.process_pipeline(data)
        builder.save_graph(output_path)
    else:
        print(f"File not found: {json_path}")