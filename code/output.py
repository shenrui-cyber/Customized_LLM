import os
import json
import math
import networkx as nx
import torch
from datetime import datetime
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- 工具函数：计算地理距离 (Haversine Formula) ---
def haversine_distance(lat1, lon1, lat2, lon2):
    if lat1 is None or lat2 is None: return float('inf')
    R = 6371  # 地球半径 (km)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

class GraphRAGAgent:
    def __init__(self, kg_path, caption_path, image_root, model_path):
        self.kg_path = kg_path
        self.caption_path = caption_path
        self.image_root = image_root
        self.model_path = model_path
        
        # 1. 加载数据
        print(f"Loading Knowledge Graph from {self.kg_path}...")
        self.G = nx.read_gml(self.kg_path)
        
        print(f"Loading Captions from {self.caption_path}...")
        with open(self.caption_path, 'r') as f:
            raw_data = json.load(f)
        self.caption_map = {item['image_file']: item for item in raw_data}

        # 2. 预处理 SuperNode 属性 (Aggregation)
        # 目的：将 Event 的属性(Geo, Entities) 上浮聚合到 SuperNode
        print("Indexing SuperNodes...")
        self.supernode_index = {} # {node_id: {'geo': (lat, lon), 'time': datetime, 'entities': set(), 'events': []}}
        
        self._build_supernode_index()

        # 3. 加载模型
        print(f"Loading Qwen3-VL from {self.model_path}...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa", # 避免 flash_attn 报错
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        print("System ready.")

    def _build_supernode_index(self):
        """
        离线建立索引：SuperNode -> 包含的事件列表、聚合的实体集合、平均经纬度
        """
        for node_id, data in self.G.nodes(data=True):
            if data.get('type') == 'SuperNode':
                self.supernode_index[node_id] = {
                    'events': [],
                    'entities': set(),
                    'geo_lat': [],
                    'geo_lon': [],
                    'timestamp': None,
                    'summary': data.get('summary', '')
                }
                # 解析 SuperNode 自身的时间戳
                try:
                    t_str = data.get('timestamp')
                    if t_str:
                        self.supernode_index[node_id]['timestamp'] = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
                except:
                    pass

        # 遍历所有边，找到 Event -> belongs_to -> SuperNode 的关系
        # 同时找到 Event -> has_participant -> Entity 的关系
        for u, v, data in self.G.edges(data=True):
            relation = data.get('relation')
            
            # 情况1: Event 属于 SuperNode (Source: u=Event, Target: v=SuperNode)
            if relation == 'belongs_to' and v in self.supernode_index:
                event_node = self.G.nodes[u]
                self.supernode_index[v]['events'].append(u)
                
                # 收集经纬度
                if 'geo' in event_node: 
                    # GML 中如果有两个 geo 属性，NetworkX 可能会读成列表或取最后值，这里做鲁棒处理
                    # 假设我们读取到的 geo 是 float 或者 list
                    # 简化处理：从 GML 属性读取 (根据你的文件结构，geo 是多值属性，nx 可能处理为 list)
                    # 实际上你的 caption.json 里有明确 lat/lon，更可靠，但为了纯图谱逻辑，我们尝试读图
                    pass 

                # 为了准确，我们反向查 caption.json 里的经纬度（因为 GML 解析多值 geo 比较麻烦）
                img_label = event_node.get('label')
                if img_label in self.caption_map:
                    cap_data = self.caption_map[img_label]
                    self.supernode_index[v]['geo_lat'].append(cap_data['latitude'])
                    self.supernode_index[v]['geo_lon'].append(cap_data['longitude'])

                # 情况2: Event 包含 Entity (Source: u=Event, Target: v=Entity)
                # 我们需要找到该 Event 所属的 SuperNode，然后把 Entity 加进去
                # 由于这是遍历边，比较麻烦。不如直接遍历 Event 节点找邻居。

        # 补充实体聚合逻辑
        for sn_id, sn_data in self.supernode_index.items():
            # 计算平均经纬度
            if sn_data['geo_lat']:
                sn_data['avg_lat'] = sum(sn_data['geo_lat']) / len(sn_data['geo_lat'])
                sn_data['avg_lon'] = sum(sn_data['geo_lon']) / len(sn_data['geo_lon'])
            else:
                sn_data['avg_lat'] = None
                sn_data['avg_lon'] = None

            # 聚合实体：遍历该 SuperNode 下的所有 Event
            for event_id in sn_data['events']:
                # 找 Event 的邻居 (Entity)
                neighbors = self.G.neighbors(event_id)
                for n_id in neighbors:
                    if self.G.nodes[n_id].get('type') == 'Entity':
                        name = self.G.nodes[n_id].get('name', '').lower()
                        sn_data['entities'].add(name)

    def retrieve_subgraph(self, query, user_lat=None, user_lon=None, user_time=None, top_k=2):
        """
        基于 SuperNode 的加权检索
        """
        print(f"Retrieving Scenes for query: '{query}' | Loc: ({user_lat}, {user_lon})")
        
        # 1. 提取 Query 中的实体 (含代词映射)
        query_lower = query.lower()
        query_entities = set()
        
        # 简单分词 (可替换为更高级的 NER)
        query_tokens = set(query_lower.replace("?", "").replace(".", "").split())
        
        # 代词映射
        pronoun_map = {'i', 'me', 'my', 'myself', 'mine'}
        if any(p in query_tokens for p in pronoun_map):
            query_entities.add('user') # 映射到图谱中的 "User" 实体
            
        # 扫描图谱中所有已知实体名，看是否在 Query 中
        all_kg_entities = set()
        for _, data in self.G.nodes(data=True):
            if data.get('type') == 'Entity':
                all_kg_entities.add(data.get('name', '').lower())
        
        for ent in all_kg_entities:
            if ent in query_lower: # 字符串包含匹配
                query_entities.add(ent)
        
        print(f"Detected Query Entities: {query_entities}")

        # 2. 为每个 SuperNode 打分
        scores = []
        
        # 权重配置 (可调整)
        W_ENTITY = 0.6
        W_GEO = 0.3
        W_TIME = 0.1
        
        for sn_id, data in self.supernode_index.items():
            # A. 实体分数 (Jaccard 或 简单交集)
            # 交集包含
            match_count = len(query_entities.intersection(data['entities']))
            # 归一化：如果没有匹配，分为0；有匹配则越高越好。
            # 这里用简单的 sigmoid 变体或者直接用 count，为了平衡，使用 count / (len(query)+1)
            score_ent = match_count / (len(query_entities) + 1e-5) if query_entities else 0.0
            
            # B. 地理分数 (距离越近分数越高)
            score_geo = 0.0
            if user_lat is not None and data['avg_lat'] is not None:
                dist = haversine_distance(user_lat, user_lon, data['avg_lat'], data['avg_lon'])
                # 使用 1 / (1 + distance) 归一化，距离0时分为1
                score_geo = 1.0 / (1.0 + dist/10.0) # /10.0 是为了让公里数不至于让分数下降太快
            
            # C. 时间分数 (时间越近分数越高)
            score_time = 0.0
            if user_time is not None and data['timestamp'] is not None:
                # 假设 user_time 是 datetime 对象
                diff_hours = abs((user_time - data['timestamp']).total_seconds()) / 3600.0
                score_time = 1.0 / (1.0 + diff_hours/24.0) # 以天为尺度衰减

            # D. 总分
            # 如果用户没提供经纬度/时间，相应权重自动降为0 (或者重新分配，这里简单处理)
            final_score = (W_ENTITY * score_ent) + (W_GEO * score_geo) + (W_TIME * score_time)
            
            scores.append((sn_id, final_score, match_count))

        # 3. 排序并取 Top-K
        # 优先按总分排，如果总分一样，按实体匹配数量排
        scores.sort(key=lambda x: x[1], reverse=True)
        top_nodes = scores[:top_k]
        
        # 4. 构建上下文 (在选中的 SuperNode 中找 Best Event)
        retrieved_context = []
        images_to_process = []
        
        for sn_id, score, _ in top_nodes:
            sn_data = self.supernode_index[sn_id]
            if not sn_data['events']: continue

            # 在该 Scene 中找到最匹配 Query 的 Event
            # 策略：看哪个 Event 连接的 Entity 在 Query 中出现最多
            best_event_id = None
            max_overlap = -1
            
            for event_id in sn_data['events']:
                # 获取该 Event 的直接实体
                event_entities = set()
                for n in self.G.neighbors(event_id):
                    if self.G.nodes[n].get('type') == 'Entity':
                        event_entities.add(self.G.nodes[n].get('name', '').lower())
                
                overlap = len(query_entities.intersection(event_entities))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_event_id = event_id
            
            # 获取 Best Event 的信息
            if best_event_id:
                event_node = self.G.nodes[best_event_id]
                img_filename = event_node.get('label')
                
                # 获取原始信息
                raw_caption = "N/A"
                if img_filename in self.caption_map:
                    raw_caption = self.caption_map[img_filename]['caption']
                
                # 图片路径
                if img_filename and img_filename.strip():
                    full_img_path = os.path.join(self.image_root, img_filename)
                    if os.path.isfile(full_img_path):
                        images_to_process.append(full_img_path)

                context_block = (
                    f"--- Scene Context (Match Score: {score:.2f}) ---\n"
                    f"Scene Summary: {sn_data['summary']}\n"
                    f"Relevant Event Time: {event_node.get('timestamp')}\n"
                    f"Location: {event_node.get('location_str')}\n"
                    f"Atmosphere: {event_node.get('atmosphere')}\n"
                    f"Event Detail: {event_node.get('summary')}\n"
                    f"Original Caption: {raw_caption}\n"
                )
                retrieved_context.append(context_block)

        return "\n".join(retrieved_context), images_to_process

    def generate_response(self, user_query, user_lat=None, user_lon=None, user_time=None):
        """
        生成入口，增加了用户状态参数
        """
        # 1. 检索
        context_text, image_paths = self.retrieve_subgraph(user_query, user_lat, user_lon, user_time)
        
        if not context_text:
            return "Based on your memories, I couldn't find any relevant scenes matching your criteria."

        # 2. 构建 Prompt
        system_prompt = (
            "You are a personal memory assistant. "
            "The user is asking a question about their past experiences. "
            "You are provided with retrieved 'Scenes' (SuperNodes) that match the user's query based on "
            "entities, location, and time. Use the Scene Summary and the specific Event Details provided to answer."
        )
        
        # 处理图片输入 (Qwen-VL)
        user_content = []
        # 限制图片数量防止显存溢出 (例如最多2张)
        for img_path in image_paths[:2]:
            user_content.append({"type": "image", "image": img_path})
            
        text_payload = (
            f"Context from Memory Graph:\n{context_text}\n\n"
            f"User Current Context: [Loc: {user_lat},{user_lon}, Time: {user_time}]\n"
            f"User Question: {user_query}"
        )
        user_content.append({"type": "text", "text": text_payload})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # 3. 推理
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]

# --- Main Execution ---
if __name__ == "__main__":
    KG_PATH = "/storage/nvme/Customized_LLM/knowledge_graph.gml"
    CAPTION_JSON = "/storage/nvme/Customized_LLM/my_dataset/caption.json"
    IMAGE_DIR = "/storage/nvme/Customized_LLM/my_dataset/images"
    MODEL_PATH = "/storage/nvme/Qwen3-VL-4B-Instruct"

    agent = GraphRAGAgent(KG_PATH, CAPTION_JSON, IMAGE_DIR, MODEL_PATH)

    # 模拟场景 1: 用户在餐厅问自己做了什么 (实体匹配 + 地点匹配)
    # 假设用户当前坐标接近上海某处 (31.23, 121.47)
    q1 = "What did I do here?"
    # 注意：这里我们传入了经纬度，agent 会自动计算距离 SuperNode 的相似度
    print(f"\n--- Query: {q1} (with location context) ---")
    res1 = agent.generate_response(q1, user_lat=44.4586, user_lon=128.2111, user_time=datetime(2023, 1, 23))
    print(f"Response:\n{res1}")

    # 模拟场景 2: 仅文本询问
    q2 = "Tell me about the canoeing competition."
    print(f"\n--- Query: {q2} ---")
    res2 = agent.generate_response(q2)
    print(f"Response:\n{res2}")