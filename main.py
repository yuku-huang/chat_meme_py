#可以玩角色扮演
# .\.venv\Scripts\Activate.ps1
import subprocess

def create_prompt(user_text):
    return f"""
    你是一個擅長製作梗圖回覆的AI對話角色，請根據使用者的輸入，分析其語氣與情緒，並提供吐槽風格的回覆方向。

    請根據以下輸入，輸出以下三項：
    1. 使用者的「情緒」：可選如 [開心、生氣、沮喪、無奈、厭世、焦慮、無聊] 等
    2. 「主題」：可選如 [工作、上課、朋友、感情、天氣、社會時事、日常生活]
    3. 適合的「吐槽式回覆語句」：請帶有幽默與反差，能與使用者產生共鳴感，並符合梗圖的語氣風格

    範例輸入：「今天上班被主管唸到炸掉，我到底為什麼要努力」
    {{
    "emotion": "無奈",
    "topic": "上班",
    "reply": "主管是不是吃太飽沒事做？你還是來這裡找梗圖比較有建設性。"
    }}

    文本內容：\"\"\"{user_text}\"\"\"
    """

def ollama_query(prompt, model="gemmapro"):
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",  # 指定使用utf-8解碼
            # timeout=30
        )
        if result.returncode != 0:
            print("Error:", result.stderr)
            return None
        return result.stdout.strip()
    except Exception as e:
        print("Exception:", e)
        return None

# if __name__ == "__main__":
#     prompt = "你好"
#     response = ollama_query(prompt)
#     if response:
#         print("Model response:", response)
#     else:
#         print("No response received.")
if __name__ == "__main__":
    user_input = input("你好啊，今天有什麼想聊的?\n")
    prompt = create_prompt(user_input)
    response = ollama_query(prompt)
    if response:
        print("\n=== Emotion Analysis Report ===")
        print(response)
    else:
        print("No response received from the model.")
