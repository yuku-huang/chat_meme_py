#使用向量空間讀取資料，第一版，已能不錯的實現功能，但尚未有資料夾功能
import json
import subprocess
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import random

# --- 設定 ---
INDEX_FILE = 'faiss_index.index' # 預先建立的 FAISS 索引檔
MAPPING_FILE = 'index_to_filename.json' # 索引 ID 到檔案名稱的對應檔
MEME_DIR = 'memes' # 梗圖圖片所在的資料夾
# 使用與建立索引時相同的嵌入模型
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
OLLAMA_MODEL = "gemmapro" # 你使用的 Ollama 模型名稱
NUM_RESULTS_TO_CONSIDER = 3 # 向量搜尋後考慮前 N 個結果
# --- 設定結束 ---

# --- Ollama 相關函式 (與你原本的類似) ---
def ollama_query(prompt, model):
    """執行 Ollama 指令並取得輸出"""
    try:
        # 注意：確保 prompt 字串正確傳遞給命令列
        # 使用 subprocess.run 可以更安全地處理參數
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8", # 確保正確處理 UTF-8
            check=True # 如果命令失敗會拋出 CalledProcessError
        )
        return result.stdout.strip()
    except FileNotFoundError:
        print(f"錯誤：找不到 'ollama' 指令。請確認 Ollama 已安裝並在系統路徑中。")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Ollama 指令執行錯誤 (Return Code: {e.returncode}):")
        print(f"Stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"執行 Ollama 時發生未預期錯誤: {e}")
        return None

def analyze_response_description(user_text, model=OLLAMA_MODEL):
    """請 Ollama 根據使用者輸入，生成理想梗圖回應的描述"""
    # 這個 Prompt 是關鍵，引導 LLM 輸出描述性文字
    prompt = f"""
你是一位幽默的梗圖吐嘈回應大師，專門用最貼切、最有趣的梗圖來回應使用者。
請仔細閱讀並理解以下使用者輸入的內容：
\"\"\"
{user_text}
\"\"\"

現在，請不要直接回覆，而是生成一段**描述性文字**，說明**什麼樣的梗圖最適合用來回應**這段使用者輸入，且帶有吐嘈的感覺。
這段描述應該捕捉到回應梗圖應有的**核心涵義、情緒、語氣和可能的情境**。

例如，如果使用者說「我今天走在路上被鳥大便砸到頭」，你可能生成描述：「一個笑到快岔氣、表情像在說『你就活該』的損友型梗圖，帶著強烈的幸災樂禍氣息，好像上天也看不下去你最近太囂張。」

如果使用者說「我昨天打LOL被小學生虐爆還被對面加好友嘲諷」，你可能生成描述：「一個滿臉無語、眼神像在說『你還是去打單機吧』的嘲諷梗圖，有種朋友想幫你擦眼淚但自己也快笑死的矛盾感。」

如果使用者說「我剛剛告白結果對方只回一句『謝謝你』」，你可能生成描述：「一個已經準備好三秒內爆笑但又假裝安慰你的梗圖，像是內心狂喊『我早就知道會這樣』，表面卻故作鎮定說『至少你勇敢過』的假掰感。」

請直接輸出這段描述性文字，力求自然流暢且語意清晰。
"""
    print("\n=== 正在請 Ollama 生成回應描述... ===")
    response = ollama_query(prompt, model)
    if response:
        print(f"Ollama 回應描述: {response}")
    else:
        print("無法從 Ollama 取得回應描述。")
    return response

# --- 向量搜尋相關函式 ---
def load_faiss_index(index_filepath):
    """載入 FAISS 索引"""
    try:
        index = faiss.read_index(index_filepath)
        print(f"成功從 {index_filepath} 載入 FAISS 索引，包含 {index.ntotal} 個向量。")
        return index
    except Exception as e:
        print(f"錯誤：載入 FAISS 索引 {index_filepath} 失敗: {e}")
        return None

def load_index_mapping(mapping_filepath):
    """載入索引 ID 到檔案名稱的對應檔"""
    try:
        with open(mapping_filepath, 'r', encoding='utf-8') as f:
            # JSON 讀取時 key 會是字串，需要轉回整數
            mapping = {int(k): v for k, v in json.load(f).items()}
        print(f"成功從 {mapping_filepath} 載入 ID 對應關係。")
        return mapping
    except FileNotFoundError:
        print(f"錯誤：找不到 ID 對應檔 {mapping_filepath}")
        return None
    except json.JSONDecodeError:
        print(f"錯誤：無法解析 JSON 檔案 {mapping_filepath}")
        return None
    except Exception as e:
        print(f"載入 ID 對應檔時發生錯誤: {e}")
        return None

def search_similar_memes(query_text, index, model, id_to_filename, k=NUM_RESULTS_TO_CONSIDER):
    """將查詢文字轉為向量，並在 FAISS 索引中搜尋最相似的 k 個結果"""
    if not query_text or index is None or model is None or id_to_filename is None:
        print("搜尋前缺少必要元素（查詢文字、索引、模型或對應表）。")
        return []

    try:
        # 1. 將查詢文字轉為向量
        print("正在生成查詢向量...")
        query_vector = model.encode([query_text], convert_to_numpy=True)
        query_vector = query_vector.astype('float32') # 確保是 float32
        print("查詢向量生成完畢。")

        # 2. 在 FAISS 索引中搜尋
        print(f"正在 FAISS 索引中搜尋前 {k} 個最相似的梗圖...")
        # search 方法回傳兩個陣列：D (distances) 和 I (indices)
        distances, indices = index.search(query_vector, k)
        print("搜尋完成。")

        # 3. 整理結果
        results = []
        if indices.size > 0:
            for i in range(indices.shape[1]):
                idx = indices[0, i]
                dist = distances[0, i]
                if idx in id_to_filename: # 確保索引 ID 有效
                    filename = id_to_filename[idx]
                    results.append({'filename': filename, 'distance': float(dist), 'index_id': int(idx)})
                else:
                    print(f"警告：在對應表中找不到索引 ID {idx}。")
        print(f"找到 {len(results)} 個相似結果：{results}")
        return results

    except Exception as e:
        print(f"向量搜尋過程中發生錯誤: {e}")
        return []

# --- 其他輔助函式 ---
def get_meme_path(filename, meme_dir=MEME_DIR):
    """取得梗圖的完整檔案路徑"""
    # 取得目前腳本所在的目錄
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, meme_dir, filename)

# --- 主程式 ---
if __name__ == "__main__":
    # 1. 載入必要的資源
    print("--- 載入資源 ---")
    faiss_index = load_faiss_index(INDEX_FILE)
    index_to_filename_map = load_index_mapping(MAPPING_FILE)
    print(f"正在載入嵌入模型: {EMBEDDING_MODEL_NAME}...")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("嵌入模型載入完成。")
    except Exception as e:
        print(f"載入嵌入模型時發生錯誤: {e}")
        embedding_model = None

    if not faiss_index or not index_to_filename_map or not embedding_model:
        print("錯誤：缺少必要的資源（索引、對應檔或模型），無法繼續執行。")
        exit()

    print("--- 資源載入完成 ---")

    # 2. 取得使用者輸入
    user_input = input("\n你好啊，今天有什麼想聊的?\n> ")

    # 3. 請 Ollama 生成回應描述
    response_description = analyze_response_description(user_input)

    if not response_description:
        print("無法從 Ollama 取得回應描述，嘗試使用原始輸入進行搜尋...")
        response_description = user_input # 後備方案：直接用使用者輸入搜尋

    # 4. 進行向量搜尋
    similar_memes = search_similar_memes(
        response_description,
        faiss_index,
        embedding_model,
        index_to_filename_map,
        k=NUM_RESULTS_TO_CONSIDER
    )

    # 5. 選擇並顯示梗圖
    if similar_memes:
        # 簡單策略：選擇距離最近的 (distance 最小)
        # 你也可以加入隨機性，例如在前 N 個相似度高的結果中隨機選
        selected_meme_info = min(similar_memes, key=lambda x: x['distance'])
        selected_meme_filename = selected_meme_info['filename']
        meme_path = get_meme_path(selected_meme_filename, MEME_DIR)

        print(f"\n=== AI 選擇的梗圖 (基於向量相似度) ===")
        print(f"檔案名稱: {selected_meme_filename}")
        print(f"檔案位置: {meme_path}")
        print(f"相似度距離 (越小越相似): {selected_meme_info['distance']:.4f}")

        # 顯示圖片
        try:
            img = Image.open(meme_path)
            img.show()
            # 在實際應用中，你可能會將圖片路徑或內容傳回給前端或聊天介面
        except FileNotFoundError:
            print(f"錯誤：找不到梗圖檔案 {meme_path}")
        except Exception as e:
            print(f"顯示圖片時發生錯誤: {e}")

    else:
        print("\n找不到符合的梗圖。")

    print("\n--- 程式執行完畢 ---")
