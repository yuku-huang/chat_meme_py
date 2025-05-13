# 可以回覆並印出圖片，但回覆方式死板
import json
import subprocess
import os
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# 讀取標注資料
def load_meme_annotations(path='meme_annotations.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 篩選符合情緒與主題的梗圖
def find_relevant_memes(meme_data, emotion=None, topic=None):
    results = []
    for fname, info in meme_data.items():
        if emotion and emotion not in info.get('emotion', []):
            continue
        if topic and topic not in info.get('topics', []):
            continue
        results.append(fname)
    return results

# Ollama 呼叫，分析情緒與主題
def analyze_emotion_topic(user_text, model="gemmapro"):
    prompt = f"""
你是梗圖回應員，我希望你可以用適當的梗圖回應使用者讓他們感受驚喜與歡笑。請根據使用者的輸入\"\"\"{user_text}\"\"\"，提供吐槽風格的回覆方向。
並用 JSON 格式回覆包含兩個欄位：emotion 與 topic。
emotion 的值必須是以下選項之一：吐槽、無奈、困惑、生氣、開心、悲傷、焦慮、無聊。

topic 的值必須是以下選項之一：朋友、生活趣事、社交評論、學業、工作、感情、天氣、社會時事。

請只輸出 JSON 格式，例如：
{{
  "emotion": "無奈",
  "topic": "生活趣事"
}}
"""
    response = ollama_query(prompt, model)
    return response

# Ollama 執行指令
def ollama_query(prompt, model="gemmapro"):
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        if result.returncode != 0:
            print("Error:", result.stderr)
            return None
        return result.stdout.strip()
    except Exception as e:
        print("Exception:", e)
        return None

# 取得圖片完整路徑
def get_meme_path(filename, meme_dir='memes'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, meme_dir, filename)

# 主程式
if __name__ == "__main__":
    meme_data = load_meme_annotations('meme_annotations.json')
    user_input = input("你好啊，今天有什麼想聊的?\n")

    analysis_json_str = analyze_emotion_topic(user_input)
    print("\n=== 吐嘈 ===")
    print(analysis_json_str)

    try:
        analysis = json.loads(analysis_json_str)
        user_emotion = analysis.get('emotion')
        user_topic = analysis.get('topic')
    except Exception as e:
        print("解析情緒與主題時出錯，使用預設值。", e)
        user_emotion = None
        user_topic = None

    matched_memes = find_relevant_memes(meme_data, user_emotion, user_topic)

    if matched_memes:
        selected_meme = matched_memes[0]  # 可改成隨機選或其他策略
        meme_path = get_meme_path(selected_meme)
        print(f"\n=== AI 選擇的梗圖 ===\n檔案名稱: {selected_meme}\n檔案位置: {meme_path}")
        Image.open(meme_path).show()
        # 你可以這裡用你想的方式回覆圖片，例如聊天機器人傳圖片，或本地展示
    else:
        print("找不到符合的梗圖。")