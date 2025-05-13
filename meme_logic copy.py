# meme_logic.py
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import logging # 使用 logging 模組

# --- 基本設定 ---
# 建議將這些路徑設為絕對路徑或相對於此檔案的路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTATION_FILE = os.path.join(BASE_DIR, 'meme_annotations_enriched.json')
INDEX_FILE = os.path.join(BASE_DIR, 'faiss_index.index')
MAPPING_FILE = os.path.join(BASE_DIR, 'index_to_filename.json')
MEME_ROOT_DIR = os.path.join(BASE_DIR, 'memes') # 梗圖圖片的根目錄

EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_FOR_BOT", "llama3-8b-8192") # 從環境變數讀取模型名稱

NUM_INITIAL_SEARCH_RESULTS = 3
MAX_REFINEMENT_ATTEMPTS = 1

# --- 初始化 Logger ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- 全域變數，用於快取載入的資源 ---
all_meme_annotations_cache = None
faiss_index_cache = None
index_to_filename_map_cache = None
embedding_model_cache = None
groq_client_cache = None

def get_groq_client():
    """取得 Groq Client 實例 (帶快取)"""
    global groq_client_cache
    if groq_client_cache is None:
        try:
            groq_api_key = os.environ.get("GROQ_API_KEY")
            if not groq_api_key:
                logger.error("環境變數 GROQ_API_KEY 未設定！")
                raise ValueError("GROQ_API_KEY is not set.")
            groq_client_cache = Groq(api_key=groq_api_key)
            logger.info("Groq Client 初始化成功。")
        except Exception as e:
            logger.error(f"初始化 Groq Client 失敗: {e}")
            raise
    return groq_client_cache

def load_all_resources():
    """載入所有必要的資源 (標註、索引、模型等) 並快取"""
    global all_meme_annotations_cache, faiss_index_cache, index_to_filename_map_cache, embedding_model_cache

    if all_meme_annotations_cache and faiss_index_cache and index_to_filename_map_cache and embedding_model_cache:
        logger.info("所有資源已從快取載入。")
        return True

    logger.info("--- 開始載入資源 ---")
    try:
        with open(ANNOTATION_FILE, 'r', encoding='utf-8') as f:
            all_meme_annotations_cache = json.load(f)
        logger.info(f"成功從 {ANNOTATION_FILE} 載入 {len(all_meme_annotations_cache)} 筆完整標註。")

        faiss_index_cache = faiss.read_index(INDEX_FILE)
        logger.info(f"成功從 {INDEX_FILE} 載入 FAISS 索引，包含 {faiss_index_cache.ntotal} 個向量。")

        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            mapping_raw = json.load(f)
            index_to_filename_map_cache = {int(k): v for k, v in mapping_raw.items()}
        logger.info(f"成功從 {MAPPING_FILE} 載入 ID 對應關係。")

        logger.info(f"正在載入嵌入模型: {EMBEDDING_MODEL_NAME}...")
        embedding_model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("嵌入模型載入完成。")

        get_groq_client() # 初始化 Groq client

        logger.info("--- 所有資源載入完成 ---")
        return True
    except FileNotFoundError as e:
        logger.error(f"資源檔案未找到: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析錯誤: {e}")
    except Exception as e:
        logger.error(f"載入資源時發生未預期錯誤: {e}")
    return False


def query_groq_api(messages, model_name=GROQ_MODEL_NAME, temperature=0.7, max_tokens=1024, is_json_output=False):
    """使用 Groq API 執行查詢並取得輸出"""
    client = get_groq_client()
    if not client:
        return {"error": "Groq client not initialized"} if is_json_output else None

    try:
        completion_params = {
            "messages": messages,
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if is_json_output:
            completion_params["response_format"] = {"type": "json_object"}

        chat_completion = client.chat.completions.create(**completion_params)
        response_content = chat_completion.choices[0].message.content.strip()

        if is_json_output:
            try:
                return json.loads(response_content)
            except json.JSONDecodeError as e:
                logger.error(f"Groq API 錯誤：期望得到 JSON，但解析失敗。回應內容：{response_content} \n錯誤：{e}")
                try:
                    json_part = response_content[response_content.find('{'):response_content.rfind('}')+1]
                    return json.loads(json_part)
                except Exception:
                    logger.error("JSON 提取失敗。")
                    return {"error": "Failed to parse JSON response", "raw_response": response_content}
        return response_content
    except Exception as e:
        logger.error(f"執行 Groq API 查詢時發生錯誤: {e}")
        if "authentication_error" in str(e).lower():
            logger.error("請檢查你的 GROQ_API_KEY 環境變數是否已正確設定，或 API 金鑰是否有效。")
        if is_json_output:
            return {"error": str(e)}
        return None

def analyze_response_description(user_text, previous_attempts=None):
    """請 Groq API 根據使用者輸入，生成理想梗圖回應的 *語意描述*"""
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
    logger.info(f"請求 Groq 生成搜尋描述 (考量先前嘗試: {bool(previous_attempts)})")
    response = query_groq_api(messages_for_groq, model_name=GROQ_MODEL_NAME)
    if response and isinstance(response, str):
        logger.info(f"Groq API 搜尋描述: {response}")
    elif not response:
        logger.warning("無法從 Groq API 取得搜尋描述。")
    return response if isinstance(response, str) else None


def validate_meme_choice(user_text, meme_info):
    """請 Groq API (作為評審) 評估選擇的梗圖是否適合"""
    system_prompt_instruction = """
你是梗圖品質管制專家與幽默感評鑑師。你的任務是嚴格評估一個被檢索到的梗圖是否真正優秀、幽默，並且極度適合用來回應使用者的陳述。請務必批判性且誠實地進行評估。

你的輸出必須是 JSON 格式，包含以下鍵：
- "relevance_score": 數字，1-5分，代表梗圖與使用者陳述的相關性 (5分最高)。
- "humor_fit_score": 數字，1-5分，代表梗圖在此特定情境下的幽默契合度 (5分最高)。
- "is_suitable": 字串，"Yes" 或 "No"，代表此梗圖是否適合發送。
- "justification": 字串，簡要說明你的判斷理由。如果 "is_suitable" 為 "No"，請說明為什麼不適合，並**明確建議一個更佳的梗圖概念或搜尋方向**。如果 "is_suitable" 為 "Yes"，可以給一句簡短的正面評價。
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
    logger.info(f"請求 Groq 驗證梗圖選擇：{meme_info.get('filename')}")
    validation_result = query_groq_api(messages_for_groq, model_name=GROQ_MODEL_NAME, temperature=0.3, is_json_output=True)

    if validation_result and not validation_result.get("error"):
        logger.info(f"Groq API 驗證結果: {validation_result}")
        if all(k in validation_result for k in ["relevance_score", "humor_fit_score", "is_suitable", "justification"]):
            return validation_result
        else:
            logger.warning(f"驗證結果 JSON 格式不完整: {validation_result}")
            return {"is_suitable": "No", "justification": "AI評審回傳格式錯誤", "relevance_score": 0, "humor_fit_score": 0, "alternative_search_description": ""}
    else:
        logger.error(f"無法從 Groq API 取得有效的驗證結果。原始回應: {validation_result}")
        return {"is_suitable": "No", "justification": "AI評審無回應或出錯", "relevance_score": 0, "humor_fit_score": 0, "alternative_search_description": ""}


def generate_final_response_text(user_text, meme_info):
    """請 Groq API 結合使用者輸入和選定的梗圖資訊，生成一段有趣且串連上下文的文字回覆"""
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
    logger.info("請求 Groq 生成最終回應文字")
    final_text = query_groq_api(messages_for_groq, model_name=GROQ_MODEL_NAME)
    if final_text:
        logger.info(f"Groq API 最終回應: {final_text}")
    else:
        logger.warning("無法從 Groq API 取得最終回應文字。")
        final_text = "呃，我好像詞窮了，不過這個梗圖你看看？" # 預設回覆
    return final_text

def generate_text_only_fallback_response(user_text, reason_for_no_meme="這次沒找到絕配的梗圖"):
    """如果找不到合適的梗圖，請 Groq API 生成純文字的幽默回應"""
    system_prompt_instruction = f"""
你是個幽默風趣、反應敏捷的聊天大師。雖然這次沒有找到適合的梗圖來搭配，但你的任務依然是用純文字給出一個能讓使用者會心一笑的回應。
請針對使用者的話，給出一個簡短、有趣、像朋友聊天的回覆。
你可以稍微提及一下為什麼這次沒有梗圖（例如："{reason_for_no_meme}"），但重點還是放在幽默地回應使用者。
"""
    messages_for_groq = [
        {"role": "system", "content": system_prompt_instruction},
        {"role": "user", "content": user_text}
    ]
    logger.info(f"請求 Groq 生成純文字備案回應 (原因: {reason_for_no_meme})")
    fallback_text = query_groq_api(messages_for_groq, model_name=GROQ_MODEL_NAME)
    if fallback_text:
        logger.info(f"Groq API 純文字備案回應: {fallback_text}")
    else:
        logger.warning("無法從 Groq API 取得純文字備案回應。")
        fallback_text = "嗯... 我今天好像不太幽默，連梗都想不出來了。"
    return fallback_text

def search_similar_memes_faiss(query_text, k=NUM_INITIAL_SEARCH_RESULTS):
    """在 FAISS 索引中搜尋最相似的 k 個結果"""
    if not query_text or faiss_index_cache is None or embedding_model_cache is None or index_to_filename_map_cache is None:
        logger.error("FAISS 搜尋前缺少必要元素（查詢文字、索引、模型或對應表）。")
        return []
    try:
        logger.info(f"FAISS 搜尋：正在生成查詢向量 for: \"{query_text[:50]}...\"")
        query_vector = embedding_model_cache.encode([query_text], convert_to_numpy=True)
        query_vector = query_vector.astype('float32')
        logger.info("FAISS 搜尋：查詢向量生成完畢。")

        logger.info(f"FAISS 搜尋：正在搜尋前 {k} 個最相似的梗圖...")
        distances, indices = faiss_index_cache.search(query_vector, k)
        logger.info("FAISS 搜尋：搜尋完成。")

        results = []
        if indices.size > 0:
            found_indices = indices[0]
            # found_distances = distances[0] # 距離暫時未使用
            for i, idx in enumerate(found_indices):
                if idx != -1 and idx in index_to_filename_map_cache:
                    filename = index_to_filename_map_cache[idx]
                    results.append({'filename': filename, 'index_id': int(idx)}) # 'distance': float(found_distances[i])
                else:
                    logger.warning(f"FAISS 搜尋：在對應表中找不到索引 ID {idx} 或索引無效。")
        logger.info(f"FAISS 搜尋：找到 {len(results)} 個相似結果：{[r['filename'] for r in results]}")
        return results
    except Exception as e:
        logger.error(f"FAISS 向量搜尋過程中發生錯誤: {e}")
        return []

def get_meme_details(filename):
    """從快取中取得梗圖的詳細資訊"""
    if all_meme_annotations_cache:
        return all_meme_annotations_cache.get(filename)
    return None

def get_meme_image_path(filename, folder):
    """取得梗圖的本地檔案路徑"""
    return os.path.join(MEME_ROOT_DIR, folder, filename)


def get_meme_reply(user_input_text):
    """
    處理使用者輸入，回傳梗圖回覆結果。
    回傳格式: {"text": "回覆文字", "image_path": "本地圖片路徑或None", "meme_filename": "梗圖檔名或None"}
    """
    if not all_meme_annotations_cache: # 確保資源已載入
        if not load_all_resources():
            return {"text": "抱歉，我內部出了一點小問題，暫時無法提供梗圖服務。", "image_path": None, "meme_filename": None}

    chosen_meme_info = None
    final_text_response = None
    meme_local_path = None
    selected_meme_filename = None
    attempts_left = MAX_REFINEMENT_ATTEMPTS + 1
    previous_search_attempts = []

    current_search_description = analyze_response_description(user_input_text, previous_attempts=None)

    if not current_search_description:
        logger.warning("無法生成初始搜尋描述，嘗試純文字回應。")
        final_text_response = generate_text_only_fallback_response(user_input_text, "我想不太到梗圖的點子耶")
        return {"text": final_text_response, "image_path": None, "meme_filename": None}

    while attempts_left > 0:
        logger.info(f"--- 第 {MAX_REFINEMENT_ATTEMPTS + 2 - attempts_left} 次嘗試搜尋梗圖 ---")
        attempts_left -= 1

        similar_memes_found = search_similar_memes_faiss(current_search_description, k=NUM_INITIAL_SEARCH_RESULTS)

        if not similar_memes_found:
            logger.info("向量搜尋未找到任何梗圖。")
            if attempts_left == 0:
                final_text_response = generate_text_only_fallback_response(user_input_text, "資料庫裡好像沒有完全符合的梗圖耶")
                break
            else:
                logger.info("嘗試生成一個全新的搜尋描述...")
                current_search_description = analyze_response_description(user_input_text, previous_attempts=[{"description": current_search_description, "reasoning":"上次搜尋沒有結果"}])
                if not current_search_description:
                    final_text_response = generate_text_only_fallback_response(user_input_text, "我想不太到梗圖的點子耶")
                    break
                previous_search_attempts.append({"description": "N/A", "reasoning": "上次搜尋沒有結果，產生新描述"}) # 記錄一下
                continue

        # 簡單起見，我們先驗證第一個找到的梗圖
        best_candidate_from_search = similar_memes_found[0]
        meme_details_for_validation = get_meme_details(best_candidate_from_search['filename'])

        if not meme_details_for_validation:
            logger.error(f"在標註檔中找不到檔案 '{best_candidate_from_search['filename']}' 的詳細資訊。")
            if attempts_left == 0:
                final_text_response = generate_text_only_fallback_response(user_input_text, "找到的梗圖資料不完整")
                break
            else:
                # 嘗試下一個候選，或重新生成描述
                # 這裡簡化為直接基於錯誤重新生成描述
                current_search_description = analyze_response_description(user_input_text, previous_attempts=[{"description": current_search_description, "reasoning":f"檔案 {best_candidate_from_search['filename']} 資料不完整"}])
                if not current_search_description:
                    final_text_response = generate_text_only_fallback_response(user_input_text, "我想不太到梗圖的點子耶")
                    break
                previous_search_attempts.append({"description": current_search_description, "reasoning": f"檔案 {best_candidate_from_search['filename']} 資料不完整"})
                continue

        full_meme_info_for_validation = {
            'filename': best_candidate_from_search['filename'],
            'meme_description': meme_details_for_validation.get('meme_description', ''),
            'core_meaning_sentiment': meme_details_for_validation.get('core_meaning_sentiment', ''),
            'typical_usage_context': meme_details_for_validation.get('typical_usage_context', ''),
            'embedding_text': meme_details_for_validation.get('embedding_text', ''),
        }

        validation = validate_meme_choice(user_input_text, full_meme_info_for_validation)

        if validation.get("is_suitable") == "Yes" and validation.get("relevance_score", 0) >= 3 and validation.get("humor_fit_score", 0) >= 3:
            logger.info(f"AI 評審通過梗圖：{best_candidate_from_search['filename']}")
            chosen_meme_info = full_meme_info_for_validation
            final_text_response = generate_final_response_text(user_input_text, chosen_meme_info)
            selected_meme_folder = meme_details_for_validation.get('folder', '')
            selected_meme_filename = chosen_meme_info['filename']
            if selected_meme_folder:
                meme_local_path = get_meme_image_path(selected_meme_filename, selected_meme_folder)
            else:
                logger.warning(f"梗圖 '{selected_meme_filename}' 在標註檔中缺少 'folder' 資訊。")
            break
        else:
            logger.info(f"AI 評審否決梗圖：{best_candidate_from_search['filename']}。理由：{validation.get('justification')}")
            previous_search_attempts.append({
                "description": current_search_description, # 記錄被否決時的搜尋描述
                "reasoning": validation.get('justification', '未提供具體理由')
            })
            alternative_desc = validation.get("alternative_search_description")

            if alternative_desc and attempts_left > 0:
                logger.info(f"使用 AI 建議的新搜尋描述：{alternative_desc}")
                current_search_description = alternative_desc
            elif attempts_left > 0:
                logger.info("AI 未提供具體新描述，嘗試基於先前失敗理由重新生成...")
                current_search_description = analyze_response_description(user_input_text, previous_attempts=previous_search_attempts)
                if not current_search_description:
                    final_text_response = generate_text_only_fallback_response(user_input_text, "我想不太到梗圖的點子耶")
                    break
            else:
                final_text_response = generate_text_only_fallback_response(user_input_text, validation.get('justification', "AI評審覺得不夠好"))
                break

    if not final_text_response: # 兜底，理論上應該在迴圈結束時有值
        final_text_response = "今天好像不太對勁，我想休息一下..."

    return {"text": final_text_response, "image_path": meme_local_path, "meme_filename": selected_meme_filename}

# 主動載入一次資源，如果此模組被匯入
if __name__ != "__main__": # 當作模組匯入時執行
    if not load_all_resources():
        logger.critical("Meme Logic 模組初始化失敗，無法載入必要資源！")

# 可用於直接測試此模組
if __name__ == "__main__":
    if load_all_resources():
        print("資源載入成功，可以開始測試 get_meme_reply。")
        test_input = input("請輸入測試文字：")
        if test_input:
            reply = get_meme_reply(test_input)
            print("\n--- 測試回覆 ---")
            print(f"文字: {reply['text']}")
            if reply['image_path']:
                print(f"梗圖檔案: {reply['meme_filename']}")
                print(f"梗圖路徑: {reply['image_path']}")
                # 在這裡可以嘗試用 Pillow 顯示圖片 (如果在本機環境)
                try:
                    if os.path.exists(reply['image_path']):
                        img = Image.open(reply['image_path'])
                        img.show()
                    else:
                        print(f"測試警告：圖片路徑不存在 {reply['image_path']}")
                except Exception as e:
                    print(f"測試顯示圖片錯誤: {e}")
            else:
                print("無梗圖回覆。")
    else:
        print("資源載入失敗，無法執行測試。")

