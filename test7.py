# 第六版，整合 Groq API
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import random
from groq import Groq # 匯入 Groq SDK

# --- 設定 ---
ANNOTATION_FILE = 'meme_annotations_enriched.json'
INDEX_FILE = 'faiss_index.index'
MAPPING_FILE = 'index_to_filename.json'
MEME_ROOT_DIR = 'memes'
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
# 設定 Groq API 金鑰的環境變數名稱 (可選，若已設定則 SDK 會自動讀取)
# GROQ_API_KEY_ENV_VAR = "GROQ_API_KEY" # SDK 會自動查找 GROQ_API_KEY
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"  # 或 "mixtral-8x7b-32768" 等 Groq 提供的模型
NUM_RESULTS_TO_CONSIDER_FOR_LLM_CHOICE = 1
# --- 設定結束 ---

# --- Groq API 相關函式 ---
def query_groq_api(messages, model_name=GROQ_MODEL_NAME, temperature=0.7, max_tokens=10000):
    """
    使用 Groq API 執行查詢並取得輸出。
    messages: 一個列表，包含對話歷史，格式為 [{"role": "system/user/assistant", "content": "..."}]
    """
    try:
        # API 金鑰通常建議設定為環境變數 GROQ_API_KEY，Groq SDK 會自動讀取
        # 如果你想明確傳遞，可以 client = Groq(api_key="YOUR_API_KEY")
        # 但更推薦使用環境變數
        client = Groq() # 如果環境變數 GROQ_API_KEY 已設定，則無需傳入 api_key

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=temperature, # 控制輸出的隨機性，0 表示最確定性
            max_tokens=max_tokens,   # 控制生成文字的最大長度
            # top_p=1, # 可選參數
            # stop=None, # 可選參數，指定停止生成的字元序列
            # stream=False # 設為 True 可進行流式輸出
        )
        response_content = chat_completion.choices[0].message.content
        return response_content.strip()
    except Exception as e:
        print(f"執行 Groq API 查詢時發生錯誤: {e}")
        if "authentication_error" in str(e).lower():
            print("請檢查你的 GROQ_API_KEY 環境變數是否已正確設定，或 API 金鑰是否有效。")
        return None

def analyze_response_description(user_text, model_name_for_groq=GROQ_MODEL_NAME):
    """
    請 Groq API 根據使用者輸入，生成理想梗圖回應的 *語意描述*。
    """
    system_prompt_instruction = f"""
任務：為梗圖向量搜尋生成精準的查詢描述。
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

請生成一段簡潔但精準的文字，描述一個最適合用來回應當前使用者輸入的梗圖的**核心視覺特徵、情緒表達、或情境氛圍**。
請直接輸出這段描述文字，不要包含任何前言或解釋。
"""
    messages_for_groq = [
        {"role": "system", "content": system_prompt_instruction},
        {"role": "user", "content": user_text}
    ]

    print("\n=== 正在請 Groq API 生成搜尋描述... ===")
    response = query_groq_api(messages_for_groq, model_name=model_name_for_groq)
    if response:
        print(f"Groq API 搜尋描述: {response}")
    else:
        print("無法從 Groq API 取得搜尋描述。")
    return response

def generate_final_response(user_text, meme_info, model_name_for_groq=GROQ_MODEL_NAME):
    """
    請 Groq API 結合使用者輸入和選定的梗圖資訊，生成一段有趣且串連上下文的文字回覆。
    """
    meme_filename = meme_info.get('filename', '未知檔案')
    meme_description = meme_info.get('meme_description', '（無描述）')
    core_meaning = meme_info.get('core_meaning_sentiment', '（無核心意義說明）')
    usage_context = meme_info.get('typical_usage_context', '（無典型情境說明）')
    embedding_text = meme_info.get('embedding_text', '（無摘要）')

    system_prompt_instruction = f"""
你是個幽默風趣、反應敏捷的梗圖聊天大師。你的任務是根據使用者的話，以及你精心挑選的梗圖，生成一段既能呼應使用者，又能巧妙運用梗圖精髓的詼諧回覆。就像跟好朋友聊天一樣，自然、有趣，帶點恰到好處的調侃。

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
"""
    messages_for_groq = [
        {"role": "system", "content": system_prompt_instruction},
        {"role": "user", "content": user_text} # 使用者原始輸入作為 user role 的 content
    ]

    print("\n=== 正在請 Groq API 生成最終回應文字... ===")
    final_text = query_groq_api(messages_for_groq, model_name=model_name_for_groq)
    if final_text:
        print(f"Groq API 最終回應: {final_text}")
    else:
        print("無法從 Groq API 取得最終回應文字。")
        final_text = "呃，我好像詞窮了，不過這個梗圖你看看？"
    return final_text


# --- 向量搜尋與資料載入相關函式 (與之前相同) ---
def load_annotations(filepath):
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
    try:
        index = faiss.read_index(index_filepath)
        print(f"成功從 {index_filepath} 載入 FAISS 索引，包含 {index.ntotal} 個向量。")
        return index
    except Exception as e:
        print(f"錯誤：載入 FAISS 索引 {index_filepath} 失敗: {e}")
        return None

def load_index_mapping(mapping_filepath):
    try:
        with open(mapping_filepath, 'r', encoding='utf-8') as f:
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
            found_indices = indices[0]
            found_distances = distances[0]
            for i, idx in enumerate(found_indices):
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, meme_root_dir, folder, filename)

# --- 主程式 ---
if __name__ == "__main__":
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

    # 檢查 Groq API 金鑰是否已設定 (可選，如果未設定 SDK 會在呼叫時報錯)
    if not os.environ.get("GROQ_API_KEY"):
        print("警告：環境變數 GROQ_API_KEY 未設定。Groq SDK 可能無法正常運作。")
        print("請設定 GROQ_API_KEY 環境變數，值為你的 Groq API 金鑰。")


    if not all_meme_annotations or not faiss_index or not index_to_filename_map or not embedding_model:
        print("錯誤：缺少必要的資源（標註檔、索引、對應檔或嵌入模型），無法繼續執行。")
        exit()

    print("--- 資源載入完成 ---")

    while True:
        user_input = input("\n你：")
        if user_input.lower() in ['quit', 'exit', '掰掰', '再見', 'ㄅㄅ']:
            print("掰掰！下次再聊！")
            break
        if not user_input.strip():
            continue

        search_description = analyze_response_description(user_input)

        if not search_description:
            print("無法生成搜尋描述，跳過本次回應。")
            continue

        similar_memes_found = search_similar_memes(
            search_description,
            faiss_index,
            embedding_model,
            index_to_filename_map,
            k=1
        )

        if similar_memes_found:
            top_meme_info = similar_memes_found[0]
            top_meme_filename = top_meme_info['filename']
            meme_details = all_meme_annotations.get(top_meme_filename)

            if not meme_details:
                print(f"錯誤：在標註檔中找不到檔案 '{top_meme_filename}' 的詳細資訊。")
                final_text_response = "糟了，我找不到這個梗圖的說明..."
                meme_path_to_show = None
            else:
                info_for_final_prompt = {
                    'filename': top_meme_filename,
                    'meme_description': meme_details.get('meme_description', '（無畫面描述）'),
                    'core_meaning_sentiment': meme_details.get('core_meaning_sentiment', '（無核心意義說明）'),
                    'typical_usage_context': meme_details.get('typical_usage_context', '（無典型情境說明）'),
                    'embedding_text': meme_details.get('embedding_text', '（無摘要）'),
                }
                final_text_response = generate_final_response(user_input, info_for_final_prompt)
                selected_meme_folder = meme_details.get('folder', '')
                if not selected_meme_folder:
                    print(f"警告：梗圖 '{top_meme_filename}' 在標註檔中缺少 'folder' 資訊。")
                    meme_path_to_show = None
                else:
                    meme_path_to_show = get_meme_path(top_meme_filename, selected_meme_folder, MEME_ROOT_DIR)

                print(f"\n梗圖機器人：{final_text_response}")

                if meme_path_to_show:
                    try:
                        if not os.path.exists(meme_path_to_show):
                             print(f"錯誤：梗圖檔案不存在於預期路徑 {meme_path_to_show}。請檢查 MEME_ROOT_DIR 設定以及梗圖是否確實放在對應的 folder 子資料夾中。")
                        else:
                            img = Image.open(meme_path_to_show)
                            img.show()
                            print(f"(使用梗圖：{top_meme_filename}，來自資料夾：{selected_meme_folder})")
                    except FileNotFoundError:
                        print(f"錯誤：找不到梗圖檔案 {meme_path_to_show}。")
                    except Exception as e:
                        print(f"顯示圖片時發生錯誤: {e}")
                else:
                     print(f"警告：因路徑問題，無法顯示梗圖 '{top_meme_filename}'。")
        else:
            print("\n梗圖機器人：嗯... 這次我沒找到特別適合的梗圖耶。")

    print("\n--- 程式執行完畢 ---")
