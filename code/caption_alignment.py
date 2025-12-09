import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ================= 配置路径 =================
MODEL_PATH = "/storage/nvme/Qwen2.5-1.5B-Instruct"
INPUT_FILE = "/storage/nvme/Customized_LLM/my_dataset/scene_graph.json"
OUTPUT_FILE = "/storage/nvme/Customized_LLM/my_dataset/structral_data.json"

# ================= 1. 加载模型 =================
print(f"正在加载模型: {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype="auto", 
    device_map="auto"
)
print("模型加载完成！")

# ================= 2. 定义 Prompt (优化版) =================
# 修改点：
# 1. 把 Named Entity 提到第一位。
# 2. 增加了 "Do NOT use 'User' if..." 的警告。
# 3. 示例中同时包含了 "Jack" 和 "I" 的情况。
system_prompt = """You are a data alignment expert. Update the Scene Graph entities to match the Caption.

### PRIORITY RULES:
1. **Named Entity**: If the caption contains a specific human name (e.g., "Jack", "Tom", "Mary"), you MUST rename the generic subject (e.g., "boy", "man") to that specific Name. 

2. **First Person**: If (and ONLY if) the caption contains "I", "Me", or "My", rename the subject to "User".

3. **PRESERVE DATA**: 
   - You MUST include ALL items in the 'participants' list from the input. Do NOT drop any items (like 'table', 'boat').
   - Keep the structure exactly as is.

### Examples:

[Case 1: Name]
Input Caption: "Jack is seated at a table."
Input Participants: ["boy", "table"]
Input Relations: [{"subject": "boy", "action": "seated", "object": "table"}]
Output:
{
  "participants": ["Jack", "table"],
  "relations": [{"subject": "Jack", "action": "seated", "object": "table"}]
}

[Case 2: First Person]
Input Caption: "I am paddling a boat."
Input Participants: ["man", "boat"]
Input Relations: [{"subject": "man", "action": "paddling", "object": "boat"}]
Output:
{
  "participants": ["User", "boat"],
  "relations": [{"subject": "User", "action": "paddling", "object": "boat"}]
}
"""

def generate_correction(caption, sg_data):
    user_content = f"""
    Task: Update Scene Graph based on Caption.
    
    Input Caption: "{caption}"
    Input Participants: {json.dumps(sg_data.get('participants', []))}
    Input Relations: {json.dumps(sg_data.get('relations', []))}
    
    Output JSON:
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        do_sample=False, # 贪婪解码，保持逻辑稳定
        repetition_penalty=1.05
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# ================= 3. 主处理流程 =================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到输入文件 {INPUT_FILE}")
        return

    print(f"正在读取数据: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"开始处理 {len(data)} 条数据...")
    
    success_count = 0
    error_count = 0

    for entry in tqdm(data, desc="Processing"):
        try:
            caption = entry['original_data']['caption']
            scene_graph = entry['scene_graph']
            
            # 调用大模型
            raw_response = generate_correction(caption, scene_graph)
            
            # 清洗输出
            clean_json_str = raw_response.replace("```json", "").replace("```", "").strip()
            
            # 解析 JSON
            new_sg_data = json.loads(clean_json_str)
            
            # 增量更新 (只更新模型返回的字段，保留其他字段)
            if 'participants' in new_sg_data:
                scene_graph['participants'] = new_sg_data['participants']
            if 'relations' in new_sg_data:
                scene_graph['relations'] = new_sg_data['relations']
                
            success_count += 1

        except json.JSONDecodeError:
            error_count += 1
            continue
        except Exception as e:
            print(f"\n未知错误: {e}")
            error_count += 1
            continue

    # ================= 4. 保存结果 =================
    print(f"\n处理完成！成功: {success_count}, 失败: {error_count}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()