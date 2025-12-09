import json
import os
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

# ================= 路径配置区域 =================

# 1. 模型本地路径 (保持不变)
MODEL_PATH = "/storage/nvme/Qwen3-VL-4B-Instruct"

# 2. 数据集根目录 (保持不变)
DATASET_ROOT = "/storage/nvme/Customized_LLM/my_dataset"

# 3. 图片文件夹路径
IMAGE_DIR = os.path.join(DATASET_ROOT, "images")

# 4. 输入的元数据文件
INPUT_JSON_FILE = os.path.join(DATASET_ROOT, "caption.json") 

# 5. 输出结果文件路径
OUTPUT_FILE = os.path.join(DATASET_ROOT, "scene_graph.json")

# ================= 核心逻辑 =================

def load_model():
    """
    加载 Qwen3-VL 模型
    修复点: 使用 Qwen2VLForConditionalGeneration 而非 AutoModelForCausalLM
    """
    print(f"正在从本地加载 Qwen3 模型: {MODEL_PATH} ...")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"错误: 找不到模型路径 {MODEL_PATH}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16, 
        device_map="cuda:0",
    )


    # 加载处理器
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return model, processor

def build_prompt_content(caption):
    """构建 Prompt 文本内容"""
    system_text = f"""
Analyze the image with the caption: "{caption}".
    Extract structured information into a strictly valid JSON format.
    
    ### STRICT GUIDELINES:
    1. **Identity Retention**: 
       - If the caption contains "I", "me", or "my", the participant MUST be labeled as "User".
       - If the caption contains a name (e.g., "Tom", "Jack"), use the Name. Do NOT change "Jack" to "boy".
    2. **Format Consistency**:
       - "participants": A list of strings.
       - "relations": Include < "subject", "action", "object" >, all participant MUST appear in relations.
       - "atmosphere": ALWAYS a list of strings (e.g., ["sunny", "happy"]). Do not use a single string.
    3. **Entity Consistency**:
       - Use the exact terminology from the caption for objects if available (e.g., if caption says "canoe", do not output "boat").

    ### JSON SCHEMA:
    {{
        "event_summary": "Short phrase summarizing the event",
        "location": "Where is this happening? (Infer from image + caption)",
        "participants": ["List of main objects/people"],
        "relations": [
            {{"subject": "Entity A", "action": "interaction verb", "object": "Entity B"}}
        ],
        "atmosphere": ["vibe1", "vibe2", "vibe3"]
    }}
    
    Output ONLY the raw JSON string.
    """
    return system_text.strip()

def parse_model_output(output_text):
    """清洗模型输出"""
    clean_text = output_text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        try:
            start = clean_text.find('{')
            end = clean_text.rfind('}') + 1
            if start != -1 and end != -1:
                return json.loads(clean_text[start:end])
        except:
            pass
        # 仅打印前50字符防止刷屏
        print(f"Warning: JSON decode failed. Raw: {clean_text[:50]}...")
        return {"raw_text_fallback": clean_text, "error": "JSON_DECODE_ERROR"}

def clean_scene_graph(struct_data):
    """
    清洗逻辑：
    1. 去除重复的 participants。
    2. 移除未在 relations 中出现的 participants。
    """
    # 如果解析出错或数据为空，直接返回
    if not struct_data or "error" in struct_data:
        return struct_data

    # 获取原始数据
    raw_participants = struct_data.get("participants", [])
    relations = struct_data.get("relations", [])

    # 1. 收集所有在 relations 中“活跃”出现的实体
    active_entities = set()
    for rel in relations:
        # 收集 subject
        if "subject" in rel and isinstance(rel["subject"], str):
            active_entities.add(rel["subject"])
        # 收集 object
        if "object" in rel and isinstance(rel["object"], str):
            active_entities.add(rel["object"])

    # 2. 清洗 participants
    # 逻辑: 先转 set 去重，再遍历检查是否在 active_entities 中
    final_participants = []
    
    # 使用 set 去重 raw_participants (防止列表里有 ["Man", "Man"])
    unique_raw_participants = list(set(raw_participants))

    for p in unique_raw_participants:
        # 只有当 participant 存在于关系网络中时才保留
        if p in active_entities:
            final_participants.append(p)
            
    # 3. 覆盖原数据
    struct_data["participants"] = final_participants
    
    return struct_data

def main():
    with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"数据条数: {len(data)}")
    print("第一条数据样例:")
    print(json.dumps(data[0], ensure_ascii=False, indent=2))

    # 1. 初始化模型
    try:
        model, processor = load_model()
    except Exception as e:
        print(f"❌ 模型加载严重错误: {e}")
        print("请确认 transformers 版本是否更新: pip install --upgrade transformers")
        return

    # 2. 检查输入文件
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"错误: 找不到输入文件 {INPUT_JSON_FILE}")
        return

    with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    print(f"🚀 开始处理，共 {len(data)} 条数据...")

    MAX_RETRIES = 3

    # 3. 遍历数据
    for i, entry in enumerate(data):
        # 兼容不同的键名
        image_filename = entry.get('image_file') or entry.get('file_name')
        caption = entry.get('caption', '')
        
        if not image_filename:
            continue

        image_path = os.path.join(IMAGE_DIR, image_filename)
        if not os.path.exists(image_path):
            print(f"跳过: 图片不存在 {image_path}")
            continue

        # 构造 Qwen3/Qwen2-VL 格式消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": build_prompt_content(caption)},
                ],
            }
        ]

        # 预处理
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = inputs.to(model.device)

        struct_data = {}
        
        for attempt in range(MAX_RETRIES):
            try:
                with torch.no_grad():
                    # 关键修改: 开启 do_sample=True，让每次生成的 token 有微小变化
                    # temperature 可以随重试次数微调，增加变数
                    current_temp = 0.6 + (attempt * 0.1) 
                    
                    generated_ids = model.generate(
                        **inputs, 
                        max_new_tokens=512,
                        do_sample=True,       # 必须开启，否则每次重试结果都一样
                        temperature=current_temp, 
                        top_p=0.9
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_text = processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]

                # 尝试解析
                parsed_result = parse_model_output(output_text)

                # 检查是否有错误标记 (基于 parse_model_output 的返回逻辑)
                if "error" in parsed_result and parsed_result["error"] == "JSON_DECODE_ERROR":
                    print(f"⚠️ [索引 {i}] 第 {attempt + 1} 次生成 JSON 解析失败，正在重试...")
                    # 如果是最后一次尝试依然失败
                    if attempt == MAX_RETRIES - 1:
                        print(f"❌ [索引 {i}] 重试耗尽，保留原始错误文本。")
                        struct_data = parsed_result
                else:
                    # 解析成功，跳出重试循环
                    struct_data = clean_scene_graph(parsed_result)
                    # 如果是重试后成功的，打印一下提示
                    if attempt > 0:
                        print(f"✅ [索引 {i}] 在第 {attempt + 1} 次尝试后成功修复。")
                    break
            
            except Exception as e:
                print(f"推理过程发生未知错误: {e}")
                if attempt == MAX_RETRIES - 1:
                    struct_data = {"error": str(e)}
        
        final_entry = {
            "original_data": entry,
            "scene_graph": struct_data
        }
        results.append(final_entry)
        
        if i % 10 == 0:
            print(f"进度: {i}/{len(data)} 已完成")
            if i > 0:
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

    # 4. 最终保存
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 处理完成！结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()