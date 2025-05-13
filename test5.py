#向量空間第三板，可以搜尋資料夾，並給出不會沒有脈絡的答案
import json
import subprocess
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import random

# --- 設定 ---
ANNOTATION_FILE = 'meme_annotations_enriched.json' # 包含 embedding_text, folder, meme_description 等的完整標註檔
INDEX_FILE = 'faiss_index.index' # 預先建立的 FAISS 索引檔
MAPPING_FILE = 'index_to_filename.json' # 索引 ID 到檔案名稱的對應檔
MEME_ROOT_DIR = 'memes' # 存放所有梗圖子資料夾的根目錄
# 使用與建立索引時相同的嵌入模型
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
OLLAMA_MODEL = "gemmapro" # 你使用的 Ollama 模型名稱
NUM_RESULTS_TO_CONSIDER = 3 # 向量搜尋後考慮前 N 個結果
# --- 設定結束 ---

# --- Ollama 相關函式 ---
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
        # 清理 Ollama 可能加入的引號或其他不必要的字符
        return result.stdout.strip().strip('"')
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
    """
    (用於向量搜尋) 請 Ollama 根據使用者輸入，生成理想梗圖回應的 *語意描述*。
    這個描述將用於向量搜尋，找到語意上最接近的梗圖。
    """
    prompt = f"""
任務：為梗圖搜尋生成查詢描述。
使用者輸入：
\"\"\"
{user_text}
\"\"\"
你是一位幽默的梗圖吐嘈回應大師，專門用最貼切、最有趣的梗圖來回應使用者。
例如，如果使用者說「我今天走在路上被鳥大便砸到頭」，你可能生成描述：「一個笑到快岔氣、表情像在說『你就活該』的損友型梗圖，帶著強烈的幸災樂禍氣息，好像上天也看不下去你最近太囂張。」

如果使用者說「我昨天打LOL被小學生虐爆還被對面加好友嘲諷」，你可能生成描述：「一個滿臉無語、眼神像在說『你還是去打單機吧』的嘲諷梗圖，有種朋友想幫你擦眼淚但自己也快笑死的矛盾感。」

如果使用者說「我剛剛告白結果對方只回一句『謝謝你』」，你可能生成描述：「一個已經準備好三秒內爆笑但又假裝安慰你的梗圖，像是內心狂喊『我早就知道會這樣』，表面卻故作鎮定說『至少你勇敢過』的假掰感。」

請生成一段簡短的文字，描述一個最適合用來回應上述輸入的梗圖的**核心涵義、情緒和語氣**。這段描述將用於向量搜尋。
例如：「一個表達極度震驚和不敢置信的梗圖」或「一個表示計劃得逞、沾沾自喜的梗圖」。
請直接輸出這段描述文字。
"""
    print("\n=== 正在請 Ollama 生成搜尋描述... ===")
    response = ollama_query(prompt, model)
    if response:
        print(f"Ollama 搜尋描述: {response}")
    else:
        print("無法從 Ollama 取得搜尋描述。")
    return response

def generate_final_response(user_text, meme_info, model=OLLAMA_MODEL):
    """
    (用於最終回覆) 請 Ollama 結合使用者輸入和選定的梗圖資訊，生成一段有趣且串連上下文的文字回覆。
    """
    meme_filename = meme_info.get('filename', '未知檔案')
    meme_description = meme_info.get('meme_description', '（無描述）')
    embedding_text = meme_info.get('embedding_text', '（無描述）')
    # 可以考慮加入更多梗圖資訊，例如 title 或 core_meaning_sentiment
    # meme_title = meme_info.get('title', '')
    # meme_meaning = meme_info.get('core_meaning_sentiment', '')

    # 這個 Prompt 是關鍵，引導 LLM 串連上下文並解釋笑點
    prompt = f"""
你是個幽默風趣的梗圖聊天機器人。你的任務是根據使用者的話，以及你找到的梗圖，生成一段詼諧幽默的回覆，就是朋友聊天互相調侃那樣。

使用者說：
\"\"\"
{user_text}
\"\"\"

你找到了一個很搭的梗圖：
檔名：{meme_filename}
梗圖描述：{meme_description}
梗圖意涵：{embedding_text}

現在，請生成一段**回覆文字**：請根據梗圖描述與梗圖意涵，結合使用者的話，來吐槽或調侃使用者。若是梗圖原文就能很好回應，則直接輸出梗圖的話，若梗圖的概念與使用者的原話之間存在比較大的差異，則請用你自己的話來表達，表達出你為什麼這張梗圖可以用來作為回覆。語氣要幽默、有趣、像朋友聊天一樣且以簡短為原則。**只要輸出最終的回覆文字**，不要包含任何前言或解釋。
範例：
若使用者說「我今天因為作業寫不完所以不能出去玩」，梗圖意涵是「一位男子誇張地按著胸口說出「我們的肉體 是受到禁錮的」，這張圖通常用來戲劇性地表達壓力、困境或對社會不滿，語氣中二、誇張又搞笑。適合朋友之間誇大地訴苦或自嘲，不適合嚴肅情境。」，你可以回覆：
「我們的肉體，是受到禁錮的」

若使用者說「我好喜歡那個女生，但我不敢告白」，梗圖意涵是「男子激動地問「你心中有愛嗎」，對方冷漠持塑膠袋無回應，形成強烈情緒反差。用於諷刺冷血行為或搞笑探問情感，具戲劇張力。」，你可以回覆：
「你心中還有愛嗎?有愛就不要只在這邊內耗，做出行動」

若使用者說「你好笨，還不如我家的狗」，梗圖意涵是「圖中人物雙手一攤說著「我無所謂 我完全無所謂啊」，這張梗圖通常用來表達表面無所謂、實際可能另有情緒的敷衍態度。語氣帶有無奈與反諷，常在朋友間互動或情感對話中出現，關鍵字包括：無所謂、敷衍、假裝不在意、幽默。」，你可以回覆：
「我無所謂 我完全無所謂啊」

請開始生成你的回覆文字：
"""
    print("\n=== 正在請 Ollama 生成最終回應文字... ===")
    final_text = ollama_query(prompt, model)
    if final_text:
        print(f"Ollama 最終回應: {final_text}")
    else:
        print("無法從 Ollama 取得最終回應文字。")
        final_text = "呃，我好像詞窮了..." # 提供一個預設回覆
    return final_text


# --- 向量搜尋與資料載入相關函式 (與之前相同) ---
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
            # 只取回傳的索引，距離暫時不用
            found_indices = indices[0]
            for idx in found_indices:
                # 確保索引 ID 在我們的映射範圍內
                if idx >= 0 and idx in id_to_filename:
                    filename = id_to_filename[idx]
                    results.append({'filename': filename, 'index_id': int(idx)}) # 暫時移除 distance
                else:
                    print(f"警告：在對應表中找不到索引 ID {idx} 或索引無效。")
        print(f"找到 {len(results)} 個相似結果的文件名：{[r['filename'] for r in results]}")
        return results

    except Exception as e:
        print(f"向量搜尋過程中發生錯誤: {e}")
        return []

# --- 其他輔助函式 ---
def get_meme_path(filename, folder, meme_root_dir=MEME_ROOT_DIR):
    """取得梗圖的完整檔案路徑 (包含 folder)。"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, meme_root_dir, folder, filename)

# --- 主程式 ---
if __name__ == "__main__":
    # 1. 載入必要的資源
    print("--- 載入資源 ---")
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

    if not all_meme_annotations or not faiss_index or not index_to_filename_map or not embedding_model:
        print("錯誤：缺少必要的資源（標註檔、索引、對應檔或模型），無法繼續執行。")
        exit()

    print("--- 資源載入完成 ---")

    # 2. 取得使用者輸入 (改成迴圈可以持續對話)
    while True:
        user_input = input("\n你：")
        if user_input.lower() in ['quit', 'exit', '掰掰', '再見']:
            print("掰掰！下次再聊！")
            break
        if not user_input:
            continue

        # 3. 請 Ollama 生成用於 *搜尋* 的描述
        search_description = analyze_response_description(user_input)

        if not search_description:
            print("無法生成搜尋描述，跳過本次回應。")
            continue

        # 4. 進行向量搜尋
        similar_memes_found = search_similar_memes(
            search_description,
            faiss_index,
            embedding_model,
            index_to_filename_map,
            k=1 # 我們只需要最相關的那一個來生成最終回覆
        )

        # 5. 選擇梗圖並生成最終回應
        if similar_memes_found:
            # 取得最相似的梗圖檔案名
            top_meme_filename = similar_memes_found[0]['filename']

            # 從完整標註檔中查找該梗圖的詳細資訊
            meme_details = all_meme_annotations.get(top_meme_filename)

            if not meme_details:
                print(f"錯誤：在標註檔中找不到檔案 '{top_meme_filename}' 的詳細資訊。")
                final_text_response = "糟了，我找不到這個梗圖的說明..."
                meme_path_to_show = None
            else:
                # 提取需要的資訊給最終回應生成函式
                info_for_final_prompt = {
                    'filename': top_meme_filename,
                    'meme_description': meme_details.get('meme_description', '（無描述）'),
                    # 你可以選擇性加入更多資訊
                    'title': meme_details.get('title', ''),
                    'core_meaning_sentiment': meme_details.get('core_meaning_sentiment', '')
                }

                # 6. 請 Ollama 生成最終的文字回應
                final_text_response = generate_final_response(user_input, info_for_final_prompt)

                # 7. 取得梗圖路徑以供顯示
                selected_meme_folder = meme_details.get('folder', '')
                meme_path_to_show = get_meme_path(top_meme_filename, selected_meme_folder, MEME_ROOT_DIR)

                # 8. 輸出最終回應文字
                print(f"\n機器人：{final_text_response}")

                # 9. 顯示圖片
                if meme_path_to_show:
                    try:
                        img = Image.open(meme_path_to_show)
                        img.show()
                        print(f"(梗圖：{top_meme_filename})") # 提示顯示了哪個梗圖
                    except FileNotFoundError:
                        print(f"錯誤：找不到梗圖檔案 {meme_path_to_show}。請檢查檔案是否存在以及 folder 標籤是否正確。")
                    except Exception as e:
                        print(f"顯示圖片時發生錯誤: {e}")
                else:
                     print(f"警告：無法確定梗圖 '{top_meme_filename}' 的路徑。")

        else:
            print("\n機器人：嗯... 這次我沒找到特別適合的梗圖耶。")
            # 在這裡可以選擇讓 LLM 生成一個純文字的回應作為後備

    print("\n--- 程式執行完畢 ---")
# 範例：
# 若使用者說「我今天報告被老闆釘在牆上，超慘」，梗圖描述是「派大星眼神空洞地說『我就是好想逃離那一切』...」，你可以回覆：
# 「拍拍你，被老闆釘的感覺我懂，真的會讓人像派大星那樣眼神死，只想大喊『我就是好想逃離那一切』啊！下次報告前多拜拜？」

# 若使用者說「我昨天考試矇對一題超難的選擇題！」，梗圖描述是「派大星指著畫面外...文字是『他媽媽一定感到很驕傲』...反諷意味...」，你可以回覆：
# 「哇！矇對難題太神啦！這運氣簡直讓我想用派大星那張『他媽媽一定感到很驕傲』的圖來（假裝）稱讚你一下了，太厲害（的反諷）了吧！XD」
