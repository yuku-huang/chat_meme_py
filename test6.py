#gemeni修正
# 向量空間第四版，整合更豐富的梗圖資訊到 LLM Prompt
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
NUM_RESULTS_TO_CONSIDER_FOR_LLM_CHOICE = 1 # 修改為1，因為目前流程是向量搜尋後直接用第一個結果。若要讓LLM選擇，此數值應 >1
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
    # 修改提示，引導LLM生成更側重於梗圖特性的描述
    prompt = f"""
任務：為梗圖向量搜尋生成精準的查詢描述。
使用者輸入：
\"\"\"
{user_text}
\"\"\"
你是一位幽默的梗圖分析師，你的目標是理解使用者的意圖和情緒，然後精準描述一個最能對此做出幽默回應的梗圖所應具備的**核心特徵、畫面元素、人物情緒、或它所傳達的獨特氛圍和諷刺點**。這段描述將被用來從一個巨大的梗圖資料庫中找出最匹配的梗圖。

範例：
使用者：「我今天走在路上被鳥大便砸到頭」
你的描述：「一個角色露出幸災樂禍、憋笑或毫不掩飾大笑的表情，可能帶有『你活該』、『這就是報應』的意味，整體氛圍是損友式的嘲笑。」

使用者：「我昨天打LOL被小學生虐爆還被對面加好友嘲諷」
你的描述：「一個角色表情極度無言、眼神死或充滿了『你沒救了』的鄙視，可能還帶著一絲想安慰但又忍不住想笑的複雜情緒，適合表達對這種慘況的無奈和嘲諷。」

使用者：「我剛剛告白結果對方只回一句『謝謝你』」
你的描述：「一個角色表情像是努力憋住不笑但嘴角已經失守，或者是一種『我就知道會這樣』的瞭然於心，帶有看好戲和假裝安慰的意味，核心是『慘遭拒絕』的經典場面。」

使用者：「老闆叫我週末加班，還說是給我學習的機會。」
你的描述：「一個角色露出極度不屑、翻白眼或暗中比中指的表情，充滿了對這種冠冕堂皇說辭的諷刺和不滿，氛圍是『我信你個鬼』的職場反諷。」

請生成一段簡潔但精準的文字，描述一個最適合用來回應上述使用者輸入的梗圖的**核心視覺特徵、情緒表達、或情境氛圍**。
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
    # 從 meme_info 獲取更豐富的資訊
    core_meaning = meme_info.get('core_meaning_sentiment', '（無核心意義說明）')
    usage_context = meme_info.get('typical_usage_context', '（無典型情境說明）')
    embedding_text = meme_info.get('embedding_text', '（無摘要）') # embedding_text 通常是梗圖內容的濃縮

    # 這個 Prompt 是關鍵，引導 LLM 串連上下文並解釋笑點
    # 修改提示，讓LLM更好地利用新增的梗圖資訊
    prompt = f"""
你是個幽默風趣、反應敏捷的梗圖聊天大師。你的任務是根據使用者的話，以及你精心挑選的梗圖，生成一段既能呼應使用者，又能巧妙運用梗圖精髓的詼諧回覆。就像跟好朋友聊天一樣，自然、有趣，帶點恰到好處的調侃。

使用者說：
\"\"\"
{user_text}
\"\"\"

你找到了一個超搭的梗圖，以下是它的詳細資料：
梗圖檔名：{meme_filename}
梗圖畫面描述：{meme_description}
梗圖核心意義與情緒：{core_meaning}
梗圖典型使用情境：{usage_context}
梗圖內容摘要（用於向量搜尋的文本）：{embedding_text}

現在，請你大展身手，生成一段**回覆文字**：
1.  **緊密結合使用者說的話和梗圖的內涵**。
2.  **發揮創意**：如果梗圖上有文字且可以直接使用，那很好！如果梗圖是圖像為主，或其文字不直接適用，請用你自己的話，巧妙地把梗圖的意境、情緒、或它最精髓的那個「點」給講出來，讓使用者能 get 到為什麼這個梗圖適合這個情境。
3.  **幽默風趣**：語氣要像朋友聊天，可以吐槽、可以調侃、可以表示同情（但方式要好笑）。
4.  **簡潔有力**：不要長篇大論，抓住重點。
5.  **只要輸出最終的回覆文字**，不要包含任何前言、解釋你的思考過程或再次重複梗圖資訊。

範例思考框架：
-   如果使用者在抱怨，梗圖是表達無奈 -> 你可以放大那種無奈感，或者用梗圖的語氣幫使用者發聲。
-   如果使用者在炫耀，梗圖是諷刺 -> 你可以用梗圖的梗來「反諷」一下。
-   如果使用者在難過，梗圖是搞笑安慰 -> 你可以用幽默的方式給予安慰。

範例（請注意，這些只是範例，你要根據實際梗圖資訊靈活應變）：
若使用者說「我今天因為作業寫不完所以不能出去玩」，梗圖核心意義是「我們的肉體是受到禁錮的，用來戲劇性表達壓力」，你可以回覆：
「唉，我懂，「我們的肉體，是受到禁錮的」！作業就是那道無形的牆啊！」

若使用者說「我好喜歡那個女生，但我不敢告白」，梗圖核心意義是「男子激動地問「你心中有愛嗎」，用於諷刺冷血行為或搞笑探問情感」，你可以回覆：
「少年仔，套句梗圖說的，「你心中有愛嗎」？有愛就衝了啊，不要只在這邊內心小劇場爆炸！」

若使用者說「你好笨，還不如我家的狗」，梗圖（派大星說「我無所謂」）核心意義是「表達表面無所謂、實際可能另有情緒的敷衍態度」，你可以回覆：
「哼，我無所謂，我完全無所謂啊～（派大星語氣）」

請開始生成你的回覆文字：
"""
    print("\n=== 正在請 Ollama 生成最終回應文字... ===")
    final_text = ollama_query(prompt, model)
    if final_text:
        print(f"Ollama 最終回應: {final_text}")
    else:
        print("無法從 Ollama 取得最終回應文字。")
        final_text = "呃，我好像詞窮了，不過這個梗圖你看看？" # 提供一個預設回覆
    return final_text


# --- 向量搜尋與資料載入相關函式 (與之前相同，略作調整以符合新邏輯) ---
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
            # 確保 JSON 中的 key (索引 ID) 被正確轉換為整數
            mapping_raw = json.load(f)
            mapping = {int(k): v for k, v in mapping_raw.items()}
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

def search_similar_memes(query_text, index, model, id_to_filename, k=NUM_RESULTS_TO_CONSIDER_FOR_LLM_CHOICE):
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
        distances, indices = index.search(query_vector, k) # distances 是距離，indices 是索引ID
        print("搜尋完成。")

        results = []
        if indices.size > 0:
            found_indices = indices[0] # indices[0] 包含了針對第一個查詢向量(我們只有一個)的k個結果
            found_distances = distances[0]
            for i, idx in enumerate(found_indices):
                # 確保索引 ID 在我們的映射範圍內
                # FAISS 回傳的 idx 可能為 -1，如果找不到足夠的鄰居或索引本身有問題
                if idx != -1 and idx in id_to_filename:
                    filename = id_to_filename[idx]
                    results.append({'filename': filename, 'index_id': int(idx), 'distance': float(found_distances[i])})
                else:
                    print(f"警告：在對應表中找不到索引 ID {idx} 或索引無效。")
        print(f"找到 {len(results)} 個相似結果：{results}")
        return results

    except Exception as e:
        print(f"向量搜尋過程中發生錯誤: {e}")
        return []

# --- 其他輔助函式 ---
def get_meme_path(filename, folder, meme_root_dir=MEME_ROOT_DIR):
    """取得梗圖的完整檔案路徑 (包含 folder)。"""
    # 確保即使在不同作業系統或從不同位置執行腳本時，路徑依然正確
    base_dir = os.path.dirname(os.path.abspath(__file__)) # 獲取目前腳本檔案所在的目錄
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
        if user_input.lower() in ['quit', 'exit', '掰掰', '再見', 'ㄅㄅ']:
            print("掰掰！下次再聊！")
            break
        if not user_input.strip():
            continue

        # 3. 請 Ollama 生成用於 *搜尋* 的描述
        search_description = analyze_response_description(user_input)

        if not search_description:
            print("無法生成搜尋描述，跳過本次回應。")
            continue

        # 4. 進行向量搜尋
        # 目前我們仍然只取最相關的一個梗圖來生成最終回覆
        # 如果未來要讓 LLM 從多個候選中選擇，這裡的 k 值和後續處理邏輯需要調整
        similar_memes_found = search_similar_memes(
            search_description,
            faiss_index,
            embedding_model,
            index_to_filename_map,
            k=1 # 只取回最相關的1個
        )

        # 5. 選擇梗圖並生成最終回應
        if similar_memes_found:
            # 取得最相似的梗圖資訊
            top_meme_info = similar_memes_found[0] # 因為 k=1，所以直接取第一個
            top_meme_filename = top_meme_info['filename']

            # 從完整標註檔中查找該梗圖的詳細資訊
            meme_details = all_meme_annotations.get(top_meme_filename)

            if not meme_details:
                print(f"錯誤：在標註檔中找不到檔案 '{top_meme_filename}' 的詳細資訊。")
                final_text_response = "糟了，我找不到這個梗圖的說明..."
                meme_path_to_show = None
            else:
                # 提取需要的資訊給最終回應生成函式
                # 確保傳遞所有在 generate_final_response 中使用到的欄位
                info_for_final_prompt = {
                    'filename': top_meme_filename,
                    'meme_description': meme_details.get('meme_description', '（無畫面描述）'),
                    'core_meaning_sentiment': meme_details.get('core_meaning_sentiment', '（無核心意義說明）'),
                    'typical_usage_context': meme_details.get('typical_usage_context', '（無典型情境說明）'),
                    'embedding_text': meme_details.get('embedding_text', '（無摘要）'), # 也傳入 embedding_text
                    # 'keywords': meme_details.get('keywords', []) # 關鍵字也可以考慮加入
                }

                # 6. 請 Ollama 生成最終的文字回應
                final_text_response = generate_final_response(user_input, info_for_final_prompt)

                # 7. 取得梗圖路徑以供顯示
                selected_meme_folder = meme_details.get('folder', '')
                if not selected_meme_folder:
                    print(f"警告：梗圖 '{top_meme_filename}' 在標註檔中缺少 'folder' 資訊。")
                    meme_path_to_show = None
                else:
                    meme_path_to_show = get_meme_path(top_meme_filename, selected_meme_folder, MEME_ROOT_DIR)


                # 8. 輸出最終回應文字
                print(f"\n梗圖機器人：{final_text_response}")

                # 9. 顯示圖片
                if meme_path_to_show:
                    try:
                        if not os.path.exists(meme_path_to_show):
                             print(f"錯誤：梗圖檔案不存在於預期路徑 {meme_path_to_show}。請檢查 MEME_ROOT_DIR 設定以及梗圖是否確實放在對應的 folder 子資料夾中。")
                        else:
                            img = Image.open(meme_path_to_show)
                            img.show()
                            print(f"(使用梗圖：{top_meme_filename}，來自資料夾：{selected_meme_folder})")
                    except FileNotFoundError: # 雖然上面檢查過一次，但以防萬一
                        print(f"錯誤：找不到梗圖檔案 {meme_path_to_show}。")
                    except Exception as e:
                        print(f"顯示圖片時發生錯誤: {e}")
                else:
                     print(f"警告：因路徑問題，無法顯示梗圖 '{top_meme_filename}'。")

        else:
            print("\n梗圖機器人：嗯... 這次我沒找到特別適合的梗圖耶。")
            # 在這裡可以選擇讓 LLM 生成一個純文字的幽默回應作為後備
            # fallback_prompt = f"使用者說：\"{user_input}\"\n請你用幽默風趣的風格，純文字回應他，不要使用梗圖。"
            # fallback_response = ollama_query(fallback_prompt, OLLAMA_MODEL)
            # if fallback_response:
            # print(f"\n梗圖機器人 (純文字)：{fallback_response}")
            # else:
            # print("\n梗圖機器人：而且我現在連純文字都想不出來了...")


    print("\n--- 程式執行完畢 ---")
