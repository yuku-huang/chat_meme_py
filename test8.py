# 向量空間第六版，整合 Groq API，並增加 AI 驗證梗圖選擇與備案機制
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import random
from groq import Groq # 匯入 Groq SDK
import time # 用於可能的重試間隔

# --- 設定 ---
ANNOTATION_FILE = 'meme_annotations_enriched.json'
INDEX_FILE = 'faiss_index.index'
MAPPING_FILE = 'index_to_filename.json'
MEME_ROOT_DIR = 'memes'
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
GROQ_MODEL_NAME = "llama3-8b-8192"  # 或 "mixtral-8x7b-32768" 等 Groq 提供的模型
# GROQ_MODEL_FOR_VALIDATION = "llama3-8b-8192" # 可以為驗證步驟選擇不同模型或相同模型
NUM_INITIAL_SEARCH_RESULTS = 3 # 向量搜尋時初步取回 N 個結果，讓驗證模型有選擇或比較空間
MAX_REFINEMENT_ATTEMPTS = 1 # 如果第一個梗圖不好，嘗試重新生成搜尋描述的次數
# --- 設定結束 ---

# --- Groq API 相關函式 ---
def query_groq_api(messages, model_name=GROQ_MODEL_NAME, temperature=0.7, max_tokens=1024, is_json_output=False):
    """
    使用 Groq API 執行查詢並取得輸出。
    messages: 一個列表，包含對話歷史，格式為 [{"role": "system/user/assistant", "content": "..."}]
    is_json_output: 如果期望 Groq 回傳的是 JSON 字串，設為 True
    """
    try:
        client = Groq()
        completion_params = {
            "messages": messages,
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if is_json_output:
            # 某些模型支援 JSON mode，可以提高輸出 JSON 的可靠性
            # 檢查你使用的模型是否支援，例如 Llama 3 的 instruct 模型通常支援
            # Groq 的 API 可能需要特定的方式來啟用 JSON mode，請參考其文件
            # 這裡我們先假設模型能良好地依照指示輸出 JSON 字串
             completion_params["response_format"] = {"type": "json_object"}


        chat_completion = client.chat.completions.create(**completion_params)
        response_content = chat_completion.choices[0].message.content.strip()

        if is_json_output:
            try:
                return json.loads(response_content)
            except json.JSONDecodeError as e:
                print(f"Groq API 錯誤：期望得到 JSON，但解析失敗。回應內容：{response_content} \n錯誤：{e}")
                # 嘗試從可能包含前後文的字串中提取 JSON
                try:
                    # 簡單提取，可能需要更複雜的正規表達式
                    json_part = response_content[response_content.find('{'):response_content.rfind('}')+1]
                    return json.loads(json_part)
                except Exception:
                    print("JSON 提取失敗。")
                    return {"error": "Failed to parse JSON response", "raw_response": response_content}
        return response_content
    except Exception as e:
        print(f"執行 Groq API 查詢時發生錯誤: {e}")
        if "authentication_error" in str(e).lower():
            print("請檢查你的 GROQ_API_KEY 環境變數是否已正確設定，或 API 金鑰是否有效。")
        if is_json_output:
            return {"error": str(e)}
        return None

def analyze_response_description(user_text, model_name_for_groq=GROQ_MODEL_NAME, previous_attempts=None):
    """
    請 Groq API 根據使用者輸入，生成理想梗圖回應的 *語意描述*。
    previous_attempts: (可選) 一個列表，包含先前嘗試過的搜尋描述和為什麼它們不理想的說明。
    """
    system_prompt_instruction = f"""
任務：為梗圖向量搜尋生成**精準且具多樣性**的查詢描述。
你是一位幽默且富有創意的梗圖分析師。你的目標是理解使用者的意圖和情緒，然後描述一個或多個（如果適用）最能對此做出幽默回應的梗圖所應具備的**核心特徵、畫面元素、人物情緒、或它所傳達的獨特氛圍和諷刺點**。
**重要：請盡量避免生成過於常見或通用的描述，除非它與使用者輸入的契合度極高。嘗試思考是否有更獨特、新穎或出人意料的切入點來選擇梗圖，同時保持幽默和相關性。**
如果使用者輸入本身很普通，你可以建議一個調侃這種普通感的梗圖。
"""
    user_prompt_content = f"使用者輸入：\n\"\"\"\n{user_text}\n\"\"\""

    if previous_attempts:
        user_prompt_content += "\n\n先前嘗試的問題與改進方向：\n"
        for attempt in previous_attempts:
            user_prompt_content += f"- 上次描述: \"{attempt['description']}\", 問題: \"{attempt['reasoning']}\"\n"
        user_prompt_content += "\n請基於上述反饋，生成一個**全新且更好**的搜尋描述。"
    else:
        user_prompt_content += "\n請生成一段簡潔但精準的文字，描述一個最適合用來回應上述使用者輸入的梗圖的**核心視覺特徵、情緒表達、或情境氛圍**。\n請直接輸出這段描述文字，不要包含任何前言或解釋。"


    messages_for_groq = [
        {"role": "system", "content": system_prompt_instruction},
        {"role": "user", "content": user_prompt_content}
    ]

    print(f"\n=== 正在請 Groq API 生成搜尋描述 (考量先前嘗試: {bool(previous_attempts)})... ===")
    response = query_groq_api(messages_for_groq, model_name=model_name_for_groq)
    if response and isinstance(response, str): #確保 response 是字串
        print(f"Groq API 搜尋描述: {response}")
    elif not response:
        print("無法從 Groq API 取得搜尋描述。")
    return response if isinstance(response, str) else None


def validate_meme_choice(user_text, meme_info, model_name_for_groq=GROQ_MODEL_NAME):
    """
    請 Groq API (作為評審) 評估選擇的梗圖是否適合。
    返回一個包含評估結果的字典。
    """
    system_prompt_instruction = """
你是梗圖品質管制專家與幽默感評鑑師。你的任務是嚴格評估一個被檢索到的梗圖是否真正優秀、幽默，並且極度適合用來回應使用者的陳述。請務必批判性且誠實地進行評估。

你的輸出必須是 JSON 格式，包含以下鍵：
- "relevance_score": 數字，1-5分，代表梗圖與使用者陳述的相關性 (5分最高)。
- "humor_fit_score": 數字，1-5分，代表梗圖在此特定情境下的幽默契合度 (5分最高)。
- "is_suitable": 字串，"Yes" 或 "No"，代表此梗圖是否適合發送。
- "justification": 字串，簡要說明你的判斷理由。如果 "is_suitable" 為 "No"，請說明為什麼不適合，並**明確建議一個更佳的梗圖概念或搜尋方向**（例如：「這個太溫和了，試著找一個更誇張諷刺的」或「這個主題不對，應該找關於拖延症的」）。如果 "is_suitable" 為 "Yes"，可以給一句簡短的正面評價。
- "alternative_search_description": 字串，僅在 "is_suitable" 為 "No" 且你能提供一個**具體的、用於重新搜尋的梗圖描述**時提供此鍵。如果無法提供具體描述，則此鍵可省略或為空字串。
"""

    user_prompt_content = f"""
請評估以下梗圖是否適合回應使用者的陳述：

使用者的陳述：
\"\"\"
{user_text}
\"\"\"

已檢索到的梗圖資訊：
檔名：{meme_info.get('filename', '未知檔案')}
梗圖畫面描述：{meme_info.get('meme_description', '（無描述）')}
梗圖核心意義與情緒：{meme_info.get('core_meaning_sentiment', '（無核心意義說明）')}
梗圖典型使用情境：{meme_info.get('typical_usage_context', '（無典型情境說明）')}
梗圖內容摘要：{meme_info.get('embedding_text', '（無摘要）')}

請根據以上所有資訊，輸出你的 JSON 格式評估報告。
"""
    messages_for_groq = [
        {"role": "system", "content": system_prompt_instruction},
        {"role": "user", "content": user_prompt_content}
    ]
    print(f"\n=== 正在請 Groq API 驗證梗圖選擇：{meme_info.get('filename')} ===")
    validation_result = query_groq_api(messages_for_groq, model_name=model_name_for_groq, temperature=0.3, is_json_output=True)

    if validation_result and not validation_result.get("error"):
        print(f"Groq API 驗證結果: {validation_result}")
        # 基本的檢查，確保必要欄位存在
        if all(k in validation_result for k in ["relevance_score", "humor_fit_score", "is_suitable", "justification"]):
            return validation_result
        else:
            print(f"驗證結果 JSON 格式不完整: {validation_result}")
            return {"is_suitable": "No", "justification": "AI評審回傳格式錯誤", "relevance_score": 0, "humor_fit_score": 0}
    else:
        print(f"無法從 Groq API 取得有效的驗證結果。原始回應: {validation_result}")
        return {"is_suitable": "No", "justification": "AI評審無回應或出錯", "relevance_score": 0, "humor_fit_score": 0}


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
"""
    messages_for_groq = [
        {"role": "system", "content": system_prompt_instruction},
        {"role": "user", "content": user_text}
    ]

    print("\n=== 正在請 Groq API 生成最終回應文字... ===")
    final_text = query_groq_api(messages_for_groq, model_name=model_name_for_groq)
    if final_text:
        print(f"Groq API 最終回應: {final_text}")
    else:
        print("無法從 Groq API 取得最終回應文字。")
        final_text = "呃，我好像詞窮了，不過這個梗圖你看看？"
    return final_text

def generate_text_only_fallback_response(user_text, reason_for_no_meme="這次沒找到絕配的梗圖", model_name_for_groq=GROQ_MODEL_NAME):
    """
    如果找不到合適的梗圖，請 Groq API 生成純文字的幽默回應。
    """
    system_prompt_instruction = f"""
你是個幽默風趣、反應敏捷的聊天大師。雖然這次沒有找到適合的梗圖來搭配，但你的任務依然是用純文字給出一個能讓使用者會心一笑的回應。
請針對使用者的話，給出一個簡短、有趣、像朋友聊天的回覆。
你可以稍微提及一下為什麼這次沒有梗圖（例如："{reason_for_no_meme}"），但重點還是放在幽默地回應使用者。
"""
    messages_for_groq = [
        {"role": "system", "content": system_prompt_instruction},
        {"role": "user", "content": user_text}
    ]
    print(f"\n=== 正在請 Groq API 生成純文字備案回應 (原因: {reason_for_no_meme})... ===")
    fallback_text = query_groq_api(messages_for_groq, model_name=model_name_for_groq)
    if fallback_text:
        print(f"Groq API 純文字備案回應: {fallback_text}")
    else:
        print("無法從 Groq API 取得純文字備案回應。")
        fallback_text = "嗯... 我今天好像不太幽默，連梗都想不出來了。"
    return fallback_text


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
    # ... (其他 error handling 保持不變)
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
    # ... (其他 error handling 保持不變)
    except json.JSONDecodeError:
        print(f"錯誤：無法解析 JSON 檔案 {mapping_filepath}")
        return None
    except Exception as e:
        print(f"載入 ID 對應檔時發生錯誤: {e}")
        return None

def search_similar_memes(query_text, index, model, id_to_filename, k=NUM_INITIAL_SEARCH_RESULTS):
    if not query_text or index is None or model is None or id_to_filename is None:
        print("搜尋前缺少必要元素（查詢文字、索引、模型或對應表）。")
        return []
    try:
        print(f"正在生成查詢向量 for: \"{query_text[:50]}...\"") # 印出部分查詢文字
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
        print(f"找到 {len(results)} 個相似結果：{[r['filename'] for r in results]}")
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

    if not os.environ.get("GROQ_API_KEY"):
        print("警告：環境變數 GROQ_API_KEY 未設定。Groq SDK 可能無法正常運作。")

    if not all_meme_annotations or not faiss_index or not index_to_filename_map or not embedding_model:
        print("錯誤：缺少必要的資源，無法繼續執行。")
        exit()
    print("--- 資源載入完成 ---")

    previous_search_attempts = [] # 用於追蹤並改進搜尋描述

    while True:
        user_input = input("\n你：")
        if user_input.lower() in ['quit', 'exit', '掰掰', '再見', 'ㄅㄅ']:
            print("掰掰！下次再聊！")
            break
        if not user_input.strip():
            continue

        chosen_meme_info = None
        final_text_response = None
        meme_path_to_show = None
        attempts_left = MAX_REFINEMENT_ATTEMPTS + 1 # 初始嘗試 + 重試次數

        current_search_description = analyze_response_description(user_input, previous_attempts=previous_search_attempts)
        previous_search_attempts = [] # 每次新的使用者輸入都重置

        if not current_search_description:
            print("無法生成初始搜尋描述，嘗試純文字回應。")
            final_text_response = generate_text_only_fallback_response(user_input, "我想不太到梗圖的點子耶")
            print(f"\n梗圖機器人：{final_text_response}")
            continue

        while attempts_left > 0:
            print(f"\n--- 第 {MAX_REFINEMENT_ATTEMPTS + 2 - attempts_left} 次嘗試搜尋梗圖 ---")
            attempts_left -= 1

            similar_memes_found = search_similar_memes(
                current_search_description,
                faiss_index,
                embedding_model,
                index_to_filename_map,
                k=NUM_INITIAL_SEARCH_RESULTS # 搜尋多個以供選擇
            )

            if not similar_memes_found:
                print("向量搜尋未找到任何梗圖。")
                if attempts_left == 0: # 如果是最後一次嘗試
                    final_text_response = generate_text_only_fallback_response(user_input, "資料庫裡好像沒有完全符合的梗圖耶")
                else:
                    # 這裡可以讓 LLM 嘗試生成一個全新的搜尋描述，而不是基於上一個的改進
                    print("嘗試生成一個全新的搜尋描述...")
                    current_search_description = analyze_response_description(user_input, previous_attempts=[{"description": current_search_description, "reasoning":"上次搜尋沒有結果"}])
                    if not current_search_description:
                        final_text_response = generate_text_only_fallback_response(user_input, "我想不太到梗圖的點子耶")
                        break # 跳出重試迴圈
                continue # 繼續下一次重試或結束

            # --- AI 驗證選擇的梗圖 ---
            # 簡單起見，我們先驗證第一個找到的梗圖
            # 更進階的作法是讓 LLM 從 similar_memes_found (N個) 中挑選一個最好的
            best_candidate_from_search = similar_memes_found[0] # 以第一個為例
            meme_details_for_validation = all_meme_annotations.get(best_candidate_from_search['filename'])

            if not meme_details_for_validation:
                print(f"錯誤：在標註檔中找不到檔案 '{best_candidate_from_search['filename']}' 的詳細資訊。")
                # 可以嘗試驗證下一個 similar_memes_found 中的梗圖，或直接放棄
                if attempts_left == 0:
                    final_text_response = generate_text_only_fallback_response(user_input, "找到的梗圖資料不完整")
                else:
                    current_search_description = analyze_response_description(user_input, previous_attempts=[{"description": current_search_description, "reasoning":f"檔案 {best_candidate_from_search['filename']} 資料不完整"}])
                    if not current_search_description:
                        final_text_response = generate_text_only_fallback_response(user_input, "我想不太到梗圖的點子耶")
                        break
                continue

            # 補全梗圖資訊以供驗證函式使用
            full_meme_info_for_validation = {
                'filename': best_candidate_from_search['filename'],
                'meme_description': meme_details_for_validation.get('meme_description', '（無畫面描述）'),
                'core_meaning_sentiment': meme_details_for_validation.get('core_meaning_sentiment', '（無核心意義說明）'),
                'typical_usage_context': meme_details_for_validation.get('typical_usage_context', '（無典型情境說明）'),
                'embedding_text': meme_details_for_validation.get('embedding_text', '（無摘要）'),
            }

            validation = validate_meme_choice(user_input, full_meme_info_for_validation)

            if validation.get("is_suitable") == "Yes" and validation.get("relevance_score", 0) >= 3 and validation.get("humor_fit_score", 0) >= 3:
                print(f"AI 評審通過梗圖：{best_candidate_from_search['filename']}")
                chosen_meme_info = full_meme_info_for_validation
                final_text_response = generate_final_response(user_input, chosen_meme_info)
                selected_meme_folder = meme_details_for_validation.get('folder', '')
                if selected_meme_folder:
                    meme_path_to_show = get_meme_path(chosen_meme_info['filename'], selected_meme_folder)
                else:
                    print(f"警告：梗圖 '{chosen_meme_info['filename']}' 在標註檔中缺少 'folder' 資訊。")
                break # 找到合適的，跳出重試迴圈
            else:
                print(f"AI 評審否決梗圖：{best_candidate_from_search['filename']}。理由：{validation.get('justification')}")
                previous_search_attempts.append({
                    "description": current_search_description,
                    "reasoning": validation.get('justification', '未提供具體理由')
                })
                alternative_desc = validation.get("alternative_search_description")
                if alternative_desc and attempts_left > 0: # 如果有建議且還有重試次數
                    print(f"使用 AI 建議的新搜尋描述：{alternative_desc}")
                    current_search_description = alternative_desc
                elif attempts_left > 0: # 沒有具體建議，但仍可重試
                     print("AI 未提供具體新描述，嘗試基於先前失敗理由重新生成...")
                     current_search_description = analyze_response_description(user_input, previous_attempts=previous_search_attempts)
                     if not current_search_description:
                        final_text_response = generate_text_only_fallback_response(user_input, "我想不太到梗圖的點子耶")
                        break # 跳出重試迴圈
                else: # 沒有重試次數了
                    final_text_response = generate_text_only_fallback_response(user_input, validation.get('justification', "AI評審覺得不夠好"))
                    break # 跳出重試迴圈
        # --- 重試迴圈結束 ---

        # 輸出最終結果
        if final_text_response:
            print(f"\n梗圖機器人：{final_text_response}")
            if chosen_meme_info and meme_path_to_show: # 只有在確定選擇了梗圖且路徑有效時才顯示
                try:
                    if not os.path.exists(meme_path_to_show):
                         print(f"錯誤：梗圖檔案不存在於預期路徑 {meme_path_to_show}。")
                    else:
                        img = Image.open(meme_path_to_show)
                        img.show()
                        print(f"(使用梗圖：{chosen_meme_info['filename']})")
                except Exception as e:
                    print(f"顯示圖片時發生錯誤: {e}")
            elif chosen_meme_info and not meme_path_to_show:
                print(f"(備註：選擇了梗圖 '{chosen_meme_info['filename']}'，但無法顯示圖片路徑)")
        else:
            # 理論上應該總是有 final_text_response (至少是純文字備案)
            print("\n梗圖機器人：今天好像不太對勁，我想休息一下...")


    print("\n--- 程式執行完畢 ---")
