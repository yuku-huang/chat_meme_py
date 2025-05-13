#向量空間第二板，可以搜尋資料夾
import json
import subprocess
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import random

# --- 設定 ---
ANNOTATION_FILE = 'meme_annotations_enriched.json' # 包含 embedding_text 和 folder 的完整標註檔
INDEX_FILE = 'faiss_index.index' # 預先建立的 FAISS 索引檔
MAPPING_FILE = 'index_to_filename.json' # 索引 ID 到檔案名稱的對應檔
MEME_ROOT_DIR = 'memes' # 存放所有梗圖子資料夾的根目錄 (例如 'memes/spongebob', 'memes/patrick')
# 使用與建立索引時相同的嵌入模型
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
OLLAMA_MODEL = "gemmapro" # 你使用的 Ollama 模型名稱
NUM_RESULTS_TO_CONSIDER = 3 # 向量搜尋後考慮前 N 個結果
# --- 設定結束 ---

# --- Ollama 相關函式 (與之前相同) ---
def ollama_query(prompt, model):
    """執行 Ollama 指令並取得輸出"""
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True
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

# --- 向量搜尋與資料載入相關函式 ---
def load_annotations(filepath):
    """載入完整的 JSON 標註檔"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功從 {filepath} 載入 {len(data)} 筆完整標註。")
        return data
    except FileNotFoundError:
        print(f"錯誤：找不到標註檔 {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"錯誤：無法解析 JSON 檔案 {filepath}")
        return None
    except Exception as e:
        print(f"載入標註檔時發生錯誤: {e}")
        return None

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
        print("正在生成查詢向量...")
        query_vector = model.encode([query_text], convert_to_numpy=True)
        query_vector = query_vector.astype('float32')
        print("查詢向量生成完畢。")

        print(f"正在 FAISS 索引中搜尋前 {k} 個最相似的梗圖...")
        distances, indices = index.search(query_vector, k)
        print("搜尋完成。")

        results = []
        if indices.size > 0:
            for i in range(indices.shape[1]):
                idx = indices[0, i]
                dist = distances[0, i]
                if idx in id_to_filename:
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
def get_meme_path(filename, folder, meme_root_dir=MEME_ROOT_DIR):
    """
    取得梗圖的完整檔案路徑 (包含 folder)。
    Args:
        filename (str): 梗圖的檔案名稱 (例如 '640.jpg').
        folder (str): 梗圖所在的子資料夾名稱 (例如 'patrick').
        meme_root_dir (str): 存放所有梗圖子資料夾的根目錄.
    Returns:
        str: 完整的檔案路徑.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 組合路徑：根目錄 / 子資料夾 / 檔案名稱
    # os.path.join 會自動處理路徑分隔符
    # 如果 folder 是空字串，os.path.join 會正確處理
    return os.path.join(base_dir, meme_root_dir, folder, filename)

# --- 主程式 ---
if __name__ == "__main__":
    # 1. 載入必要的資源
    print("--- 載入資源 ---")
    # 載入完整的標註檔，以便查詢 folder
    all_meme_annotations = load_annotations(ANNOTATION_FILE)
    faiss_index = load_faiss_index(INDEX_FILE)
    index_to_filename_map = load_index_mapping(MAPPING_FILE)

    print(f"正在載入嵌入模型: {EMBEDDING_MODEL_NAME}...")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("嵌入模型載入完成。")
    except Exception as e:
        print(f"載入嵌入模型時發生錯誤: {e}")
        embedding_model = None

    # 檢查所有必要資源是否都成功載入
    if not all_meme_annotations or not faiss_index or not index_to_filename_map or not embedding_model:
        print("錯誤：缺少必要的資源（標註檔、索引、對應檔或模型），無法繼續執行。")
        exit()

    print("--- 資源載入完成 ---")

    # 2. 取得使用者輸入
    user_input = input("\n你好啊，今天有什麼想聊的?\n> ")

    # 3. 請 Ollama 生成回應描述
    response_description = analyze_response_description(user_input)

    if not response_description:
        print("無法從 Ollama 取得回應描述，嘗試使用原始輸入進行搜尋...")
        response_description = user_input

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
        selected_meme_info = min(similar_memes, key=lambda x: x['distance'])
        selected_meme_filename = selected_meme_info['filename']

        # *** 新增：從完整標註檔中查找對應的 folder ***
        selected_meme_folder = all_meme_annotations.get(selected_meme_filename, {}).get('folder', '')
        if not selected_meme_folder:
             print(f"警告：在標註檔中找不到檔案 '{selected_meme_filename}' 的 'folder' 資訊，將嘗試在根目錄 '{MEME_ROOT_DIR}' 下尋找。")

        # *** 修改：使用包含 folder 的路徑取得函式 ***
        meme_path = get_meme_path(selected_meme_filename, selected_meme_folder, MEME_ROOT_DIR)

        print(f"\n=== AI 選擇的梗圖 (基於向量相似度) ===")
        print(f"檔案名稱: {selected_meme_filename}")
        print(f"所在資料夾: {selected_meme_folder if selected_meme_folder else '(未指定)'}")
        print(f"檔案位置: {meme_path}")
        print(f"相似度距離 (越小越相似): {selected_meme_info['distance']:.4f}")

        # 顯示圖片
        try:
            img = Image.open(meme_path)
            img.show()
        except FileNotFoundError:
            print(f"錯誤：找不到梗圖檔案 {meme_path}。請檢查檔案是否存在以及 folder 標籤是否正確。")
        except Exception as e:
            print(f"顯示圖片時發生錯誤: {e}")

    else:
        print("\n找不到符合的梗圖。")

    print("\n--- 程式執行完畢 ---")
