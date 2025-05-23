# meme_logic.py
import json
import os
import logging
import re
import time
import random
from typing import Dict, Optional, List
import requests # 確保 requests 已在 requirements.txt

# --- 初始化 Logger ---
logger = logging.getLogger(__name__)
# 注意：在 Vercel 環境中，logging 的基本設定可能由平台處理，
# 但為了本地測試和明確性，可以保留或調整。
if not logger.hasHandlers(): # 避免重複設定 handler
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- API 金鑰管理 (Groq) ---
import threading

class APIKeyManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(APIKeyManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.api_keys: list = []
        self.current_index: int = 0
        self._index_lock = threading.Lock()
        self.load_api_keys()
        if self.api_keys:
            self.current_index = random.randint(0, len(self.api_keys) - 1)
        self._initialized = True

    def load_api_keys(self):
        i = 1
        while True:
            key = os.environ.get(f'GROQ_API_KEY_{i}')
            if not key:
                break
            self.api_keys.append(key)
            i += 1
        if not self.api_keys:
            single_key = os.environ.get('GROQ_API_KEY')
            if single_key:
                self.api_keys.append(single_key)
        if not self.api_keys:
            logger.error("未找到任何 Groq API 金鑰！請設定 GROQ_API_KEY 或 GROQ_API_KEY_1 等環境變數。")

    def get_next_key(self, task_type: str = 'default') -> Optional[str]:
        if not self.api_keys:
            return None
        with self._index_lock:
            key = self.api_keys[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.api_keys)
        return key

api_key_manager = APIKeyManager()

# --- 基本設定 ---
# BASE_DIR 和 ANNOTATION_FILE 的定義方式將改變，因為我們要從 URL 載入
# ANNOTATION_FILE = os.path.join(BASE_DIR, 'meme_annotations_enriched.json') # 不再這樣使用

GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_FOR_BOT", "llama3-8b-8192") # 更新為常見模型

# --- 新增：梗圖註釋檔案的 URL ---
# 你需要將 'YOUR_ANNOTATIONS_JSON_URL' 替換成你 meme_annotations_enriched.json 檔案的實際公開 URL
ANNOTATIONS_JSON_URL = os.environ.get('ANNOTATIONS_JSON_URL', 'YOUR_ANNOTATIONS_JSON_URL')

MEME_SEARCH_API_URL = os.environ.get('MEME_SEARCH_API_URL', 'YOUR_MEME_SEARCH_API_ENDPOINT/search')
MEME_DETAILS_API_URL = os.environ.get('MEME_DETAILS_API_URL', 'YOUR_MEME_SEARCH_API_ENDPOINT/details')

# --- 常數 ---
NUM_CANDIDATE_REPLIES = 3
NUM_MEMES_PER_REPLY_SEARCH = 3
MAX_OUTER_LOOP_ATTEMPTS = 2
MIN_RELEVANCE_SCORE_FOR_ACCEPTANCE = 3
MIN_HUMOR_FIT_SCORE_FOR_ACCEPTANCE = 3

# --- 全域變數 ---
all_meme_annotations_cache = None
groq_clients_cache: Dict[str, 'Groq'] = {}

Groq = None

def ensure_groq_imported():
    global Groq
    if Groq is None:
        try:
            from groq import Groq as GroqClient
            Groq = GroqClient
            logger.info("Groq SDK 成功匯入。")
        except ImportError:
            logger.error("Groq SDK 未安裝。請執行 'pip install groq'")
            raise # 這裡拋出異常，因為 Groq 是核心依賴

def get_groq_client(task_type: str = 'default') -> Optional['Groq']:
    global groq_clients_cache
    ensure_groq_imported() # 確保 Groq 已匯入
    if Groq is None:
        logger.error("Groq SDK 無法使用。")
        return None

    # ... (get_groq_client 的其餘部分與之前相同)
    if task_type in groq_clients_cache:
        return groq_clients_cache[task_type]

    try:
        api_key = api_key_manager.get_next_key(task_type)
        if not api_key:
            logger.error(f"無法為任務類型 '{task_type}' 獲取 API 金鑰")
            return None
        client = Groq(api_key=api_key) # 使用 Groq() 而不是 GroqClient()
        groq_clients_cache[task_type] = client
        logger.info(f"為任務類型 '{task_type}' 初始化新的 Groq Client")
        return client
    except Exception as e:
        logger.error(f"初始化 Groq Client 失敗: {e}")
        return None


def load_all_resources():
    global all_meme_annotations_cache

    if all_meme_annotations_cache:
        logger.info("梗圖註釋已從快取載入。")
        return True

    logger.info("--- 開始載入資源 (從 URL 載入註釋檔案) ---")
    
    if not ANNOTATIONS_JSON_URL or ANNOTATIONS_JSON_URL == 'YOUR_ANNOTATIONS_JSON_URL':
        logger.error("ANNOTATIONS_JSON_URL 未設定或仍為預設值。無法載入梗圖註釋。")
        return False
        
    try:
        logger.info(f"正在從 URL 下載註釋檔案: {ANNOTATIONS_JSON_URL}")
        response = requests.get(ANNOTATIONS_JSON_URL, timeout=15) # 設定超時
        response.raise_for_status() # 檢查 HTTP 錯誤
        all_meme_annotations_cache = response.json()
        logger.info(f"成功從 URL 載入 {len(all_meme_annotations_cache)} 筆完整標註。")

        # 初始化 Groq client (如果尚未初始化)
        # 確保 get_groq_client 在這裡被呼叫，以處理 API 金鑰和 client 初始化
        if not get_groq_client(): # 嘗試初始化預設的 client
             logger.error("預設的 Groq Client 初始化失敗。")
             # 根據你的應用邏輯，這裡可能需要決定是否要讓應用程式啟動失敗
             # return False # 如果 Groq Client 是絕對必要的

        logger.info("--- 資源載入完成 ---")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"從 URL 下載註釋檔案時發生網路錯誤: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"解析從 URL 下載的 JSON 註釋時發生錯誤: {e}")
    except Exception as e:
        logger.error(f"載入資源時發生未預期錯誤: {e}")
    
    all_meme_annotations_cache = None # 確保出錯時快取是 None
    return False

# ... (query_groq_api, generate_multiple_candidate_replies, validate_meme_choice, 
#      generate_final_response_text, generate_text_only_fallback_response, 
#      search_memes_via_api, get_meme_details_via_api, get_meme_reply 這些函式保持與上一版相同)
# 你需要確保這些函式中的 logger.xxx 呼叫是正確的。

# 以下是為了讓程式碼片段完整而複製過來的函式，內容與上一版相同
# 請檢查並確保它們與你最新的版本一致

def query_groq_api(messages, model_name=GROQ_MODEL_NAME, temperature=0.7, max_tokens=1024, is_json_output=False, task_type: str = 'default'):
    client = get_groq_client(task_type)
    if not client:
        return {"error": "Groq client not initialized"} if is_json_output else None

    max_retries = 3
    retry_count = 0
    last_error = None

    while retry_count < max_retries:
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
                    logger.error(f"Groq API 錯誤：期望得到 JSON，但解析失敗。回應內容：{response_content} 錯誤：{e}")
                    try:
                        json_part_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                        if json_part_match:
                            json_part = json_part_match.group(0)
                            logger.info(f"提取到的 JSON 部分: {json_part}")
                            return json.loads(json_part)
                        else:
                            raise ValueError("在回應中找不到有效的 JSON 物件。")
                    except Exception as extract_e:
                        logger.error(f"JSON 提取失敗: {extract_e}")
                        return {"error": "Failed to parse or extract JSON response", "raw_response": response_content}
            return response_content

        except Exception as e:
            last_error = e
            if "rate_limit_exceeded" in str(e).lower() or "authentication_error" in str(e).lower() or "api_key" in str(e).lower():
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Groq API 錯誤 ({type(e).__name__})，嘗試切換 API 金鑰並重試 ({retry_count}/{max_retries})...")
                    client = get_groq_client(task_type) # 這會輪換金鑰
                    if not client:
                        logger.error("無法獲取新的 Groq Client 進行重試。")
                        break 
                    time.sleep(2 ** retry_count) 
                    continue
            logger.error(f"執行 Groq API 查詢時發生非預期的錯誤: {e}")
            break 

    logger.error(f"Groq API 查詢在 {retry_count} 次嘗試後失敗。最後錯誤: {last_error}")
    if is_json_output:
        return {"error": f"Max retries or critical error: {str(last_error)}"}
    return None


def generate_multiple_candidate_replies(user_text, num_replies=NUM_CANDIDATE_REPLIES):
    system_prompt_instruction = f"""
你是一個反應快又幽默的聊天夥伴。針對使用者的輸入，請生成 {num_replies} 個不同的、簡短且風趣的文字回覆。
每個回覆都應該是針對使用者輸入的一個潛在回應。
請以 JSON 格式輸出，包含一個名為 "replies" 的鍵，其值為一個包含這些回覆字串的列表。
例如: {{"replies": ["回覆1", "回覆2", "回覆3"]}}
"""
    user_prompt_content = f"""
使用者輸入：
{user_text}

請生成 {num_replies} 個不同的幽默回覆。
"""
    messages_for_groq = [
        {"role": "system", "content": system_prompt_instruction},
        {"role": "user", "content": user_prompt_content}
    ]
    logger.info(f"請求 Groq 生成 {num_replies} 個候選回覆 for: {user_text[:50]}...")
    response = query_groq_api(messages_for_groq, model_name=GROQ_MODEL_NAME, temperature=0.8, is_json_output=True, task_type='generate_replies')

    if response and not response.get("error") and "replies" in response and isinstance(response["replies"], list):
        logger.info(f"Groq API 成功生成 {len(response['replies'])} 個候選回覆: {response['replies']}")
        return response["replies"]
    else:
        logger.warning(f"無法從 Groq API 取得有效的候選回覆列表。回應: {response}")
        return []

def validate_meme_choice(user_text, meme_info):
    system_prompt_instruction = """
You are a meme quality control expert and humor evaluator. Your task is to critically and honestly assess whether a retrieved meme is truly excellent, humorous, and highly suitable for responding to the user's statement.
Pay special attention to whether the meme's concept aligns with the context of the user's conversation, ensuring the dialogue is coherent and not disjointed.

Your output must be in JSON format, including the following keys:
- "relevance_score": A number from 1 to 5 representing the relevance of the meme to the user's statement (5 being the highest).
- "humor_fit_score": A number from 1 to 5 representing the humor fit of the meme in this specific context (5 being the highest).
- "is_suitable": A string, "Yes" or "No", indicating whether the meme is suitable to be sent.
- "justification": A string briefly explaining your reasoning. If "is_suitable" is "No", explain why it is not suitable and clearly suggest a better meme concept or search direction. If "is_suitable" is "Yes", you can provide a short positive comment.
- "alternative_search_description": A string provided only if "is_suitable" is "No" and you can offer a specific meme description for re-searching. If no specific description can be provided, this key can be omitted or left empty.
"""
    user_prompt_content = f"""
Please evaluate whether the following meme is suitable for responding to the user's statement:

User's statement:
{user_text}

Retrieved meme information:
Filename: {meme_info.get('filename', 'Unknown file')}
Meme description: {meme_info.get('meme_description', '(No description)')}
Core meaning and sentiment: {meme_info.get('core_meaning_sentiment', '(No core meaning description)')}
Typical usage context: {meme_info.get('typical_usage_context', '(No typical context description)')}
Meme content summary: {meme_info.get('embedding_text', '(No summary)')}

Based on all the above information, please output your evaluation report in JSON format.
"""
    messages_for_groq = [
        {"role": "system", "content": system_prompt_instruction},
        {"role": "user", "content": user_prompt_content}
    ]
    logger.info(f"Requesting Groq to evaluate meme suitability: {meme_info.get('filename', 'Unknown file')}")
    validation_result = query_groq_api(messages_for_groq, model_name=GROQ_MODEL_NAME, temperature=0.3, is_json_output=True, task_type='validate_meme')

    if validation_result and not validation_result.get("error"):
        logger.info(f"Groq API evaluation result: {validation_result}")
        if all(k in validation_result for k in ["relevance_score", "humor_fit_score", "is_suitable", "justification"]):
            return validation_result
        else:
            logger.warning(f"驗證結果 JSON 格式不完整: {validation_result}")
            return {
                "relevance_score": 1, "humor_fit_score": 1, 
                "is_suitable": "No", "justification": "AI validation response format error.",
                "alternative_search_description": ""
            }
    else:
        logger.warning(f"Failed to obtain valid evaluation result from Groq API. Response: {validation_result}")
        return {"is_suitable": "No", "relevance_score": 1, "humor_fit_score": 1, "justification": "Failed to obtain valid evaluation result from AI."}


def generate_final_response_text(user_text, meme_info):
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

現在，請你大展身手，生成一段回覆文字：
1.  緊密結合使用者說的話和梗圖的內涵。
2.  發揮創意：如果梗圖上有文字且可以直接使用，那很好！如果梗圖是圖像為主，或其文字不直接適用，請用你自己的話，巧妙地把梗圖的意境、情緒、或它最精髓的那個「點」給講出來，讓使用者能 get 到為什麼這個梗圖適合這個情境。
3.  幽默風趣：語氣要像朋友聊天，可以吐槽、可以調侃、可以表示同情（但方式要好笑）。
4.  簡潔有力：不要長篇大論，抓住重點。
5.  只要輸出最終的回覆文字，不要包含任何前言、解釋你的思考過程或再次重複梗圖資訊。
"""
    messages_for_groq = [
        {"role": "system", "content": system_prompt_instruction},
        {"role": "user", "content": user_text}
    ]
    logger.info("請求 Groq 生成最終回應文字")
    final_text = query_groq_api(messages_for_groq, model_name=GROQ_MODEL_NAME, task_type='generate_response')
    if final_text and isinstance(final_text, str): 
        logger.info(f"Groq API 最終回應: {final_text}")
    else:
        logger.warning(f"無法從 Groq API 取得最終回應文字，或回傳非字串。回應: {final_text}")
        final_text = "呃，我好像詞窮了，不過這個梗圖你看看？" 
    return final_text

def generate_text_only_fallback_response(user_text, reason_for_no_meme="這次沒找到絕配的梗圖"):
    system_prompt_instruction = f"""
你是個幽默風趣、反應敏捷的聊天大師。雖然這次沒有找到適合的梗圖來搭配，但你的任務依然是用純文字給出一個能讓使用者會心一笑的回應。
請針對使用者的話，給出一個簡短、有趣、像朋友聊天的回覆。
你可以稍微提及一下為什麼這次沒有梗圖（例如："{reason_for_no_meme}"），但重點還是放在幽默地回應使用者，但是回應不要太長，2句話以內，最好1句話。
"""
    messages_for_groq = [
        {"role": "system", "content": system_prompt_instruction},
        {"role": "user", "content": user_text}
    ]
    logger.info(f"請求 Groq 生成純文字備案回應 (原因: {reason_for_no_meme})")
    fallback_text = query_groq_api(messages_for_groq, model_name=GROQ_MODEL_NAME, task_type='generate_replies') 
    if fallback_text and isinstance(fallback_text, str):
        logger.info(f"Groq API 純文字備案回應: {fallback_text}")
    else:
        logger.warning(f"無法從 Groq API 取得純文字備案回應，或回傳非字串。回應: {fallback_text}")
        fallback_text = "嗯... 我今天好像不太幽默，連梗都想不出來了。"
    return fallback_text

def search_memes_via_api(query_text: str, k: int = NUM_MEMES_PER_REPLY_SEARCH) -> List[Dict]:
    if not MEME_SEARCH_API_URL or MEME_SEARCH_API_URL == 'YOUR_MEME_SEARCH_API_ENDPOINT/search':
        logger.error("MEME_SEARCH_API_URL 未設定或仍為預設值。無法執行外部梗圖搜尋。")
        return []

    payload = {"query_text": query_text, "k": k}
    logger.info(f"正在呼叫外部梗圖搜尋 API: {MEME_SEARCH_API_URL}，查詢: {query_text[:50]}..., k={k}")

    try:
        response = requests.post(MEME_SEARCH_API_URL, json=payload, timeout=10) 
        response.raise_for_status()  
        api_results = response.json() 
        
        if "results" in api_results and isinstance(api_results["results"], list):
            logger.info(f"外部 API 成功回傳 {len(api_results['results'])} 個搜尋結果。")
            return api_results["results"] 
        else:
            logger.warning(f"外部 API 回應格式不符預期。回應: {api_results}")
            return []
    except requests.exceptions.RequestException as e:
        logger.error(f"呼叫外部梗圖搜尋 API 時發生錯誤: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"解析外部梗圖搜尋 API 回應時發生 JSON 錯誤: {e}")
        return []

def get_meme_details_via_api(filename: str) -> Optional[Dict]:
    global all_meme_annotations_cache
    if all_meme_annotations_cache and filename in all_meme_annotations_cache:
        return all_meme_annotations_cache[filename]

    if not MEME_DETAILS_API_URL or MEME_DETAILS_API_URL == 'YOUR_MEME_SEARCH_API_ENDPOINT/details':
        logger.warning(f"MEME_DETAILS_API_URL 未設定或仍為預設值，且快取中無 '{filename}' 的資訊。")
        # 如果 ANNOTATIONS_JSON_URL 設定了，但 all_meme_annotations_cache 是 None (表示初次載入失敗)
        # 這裡不應該再嘗試從本地檔案讀取，因為我們的目標是完全移除本地檔案依賴
        if ANNOTATIONS_JSON_URL and ANNOTATIONS_JSON_URL != 'YOUR_ANNOTATIONS_JSON_URL' and all_meme_annotations_cache is None:
             logger.warning(f"註釋快取為空，且無法從 MEME_DETAILS_API_URL 獲取 '{filename}'。")
        return None # 直接回傳 None

    params = {"filename": filename}
    logger.info(f"正在呼叫外部梗圖詳細資訊 API: {MEME_DETAILS_API_URL}，檔案名: {filename}")
    try:
        response = requests.get(MEME_DETAILS_API_URL, params=params, timeout=5)
        response.raise_for_status()
        meme_details = response.json()

        if meme_details and isinstance(meme_details, dict): 
            logger.info(f"外部 API 成功回傳梗圖 '{filename}' 的詳細資訊。")
            # 更新快取
            if all_meme_annotations_cache is not None: # 只有當快取成功初始化過才更新
                 all_meme_annotations_cache[filename] = meme_details
            elif ANNOTATIONS_JSON_URL and ANNOTATIONS_JSON_URL != 'YOUR_ANNOTATIONS_JSON_URL':
                 # 如果快取從未成功載入 (例如 ANNOTATIONS_JSON_URL 首次載入失敗)
                 # 這裡我們只拿到了單個梗圖的資訊，不適合直接覆蓋 all_meme_annotations_cache
                 logger.info(f"註釋快取為空，已透過 API 取得 '{filename}' 的單獨資訊。")
            return meme_details
        else:
            logger.warning(f"外部 API 為 '{filename}' 回傳的詳細資訊格式不符預期。回應: {meme_details}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"呼叫外部梗圖詳細資訊 API 時發生錯誤 for '{filename}': {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"解析外部梗圖詳細資訊 API 回應時發生 JSON 錯誤 for '{filename}': {e}")
        return None


def get_meme_reply(user_input_text: str) -> Dict:
    if not all_meme_annotations_cache: 
        if not load_all_resources():
            return {"text": "抱歉，我內部出了一點小問題，暫時無法提供梗圖服務。", "image_path": None, "meme_filename": None, "meme_folder": None}

    chosen_meme_info = None
    final_text_response = None
    selected_meme_filename = None
    selected_meme_folder = None # 初始化
    
    outer_attempts_left = MAX_OUTER_LOOP_ATTEMPTS

    while outer_attempts_left > 0:
        logger.info(f"--- 開始第 {MAX_OUTER_LOOP_ATTEMPTS - outer_attempts_left + 1} 輪梗圖搜尋與評估 ---")
        outer_attempts_left -= 1

        candidate_reply_texts = generate_multiple_candidate_replies(user_input_text, NUM_CANDIDATE_REPLIES)

        if not candidate_reply_texts:
            logger.warning("無法生成候選回覆文字。")
            if outer_attempts_left == 0: break
            continue

        all_potential_memes_with_details = []
        for reply_text in candidate_reply_texts:
            logger.info(f"以候選回覆透過 API 搜尋梗圖: {reply_text[:50]}...")
            similar_memes_found = search_memes_via_api(reply_text, k=NUM_MEMES_PER_REPLY_SEARCH)
            
            for meme_summary in similar_memes_found: 
                filename = meme_summary.get('filename')
                if not filename:
                    logger.warning(f"API 搜尋結果缺少 'filename': {meme_summary}")
                    continue

                meme_details = get_meme_details_via_api(filename) # 會先查快取
                if meme_details:
                    full_meme_info = {
                        'filename': filename,
                        'meme_description': meme_details.get('meme_description', ''),
                        'core_meaning_sentiment': meme_details.get('core_meaning_sentiment', ''),
                        'typical_usage_context': meme_details.get('typical_usage_context', ''),
                        'embedding_text': meme_details.get('embedding_text', ''),
                        'folder': meme_details.get('folder', '') 
                    }
                    all_potential_memes_with_details.append({
                        'meme_info': full_meme_info,
                        'source_reply_text': reply_text
                    })
                else:
                    logger.warning(f"無法取得檔案 '{filename}' 的詳細資訊 (來自快取或 API)。")

        if not all_potential_memes_with_details:
            logger.info("本輪未找到任何潛在梗圖。")
            if outer_attempts_left == 0: break
            continue

        validated_candidates = []
        for potential in all_potential_memes_with_details:
            validation = validate_meme_choice(user_input_text, potential['meme_info'])
            validated_candidates.append({
                'validation_result': validation,
                'meme_info': potential['meme_info'],
                'source_reply_text': potential['source_reply_text']
            })
        
        validated_candidates.sort(
            key=lambda vc: (
                vc['validation_result'].get('is_suitable', 'No') != 'Yes',
                -(vc['validation_result'].get('relevance_score', 0) + vc['validation_result'].get('humor_fit_score', 0))
            )
        )

        best_choice = None
        for vc in validated_candidates:
            val_res = vc['validation_result']
            if (val_res.get("is_suitable") == "Yes" and
                val_res.get("relevance_score", 0) >= MIN_RELEVANCE_SCORE_FOR_ACCEPTANCE and
                val_res.get("humor_fit_score", 0) >= MIN_HUMOR_FIT_SCORE_FOR_ACCEPTANCE):
                best_choice = vc
                break
        
        if best_choice:
            logger.info(f"AI 評審通過梗圖：{best_choice['meme_info']['filename']} (來自候選回覆: {best_choice['source_reply_text'][:30]}...)")
            chosen_meme_info = best_choice['meme_info']
            final_text_response = generate_final_response_text(user_input_text, chosen_meme_info)
            selected_meme_filename = chosen_meme_info['filename']
            selected_meme_folder = chosen_meme_info.get('folder') # 從 chosen_meme_info 獲取 folder
            
            return {"text": final_text_response, "image_path": None, "meme_filename": selected_meme_filename, "meme_folder": selected_meme_folder}
        else:
            logger.info("本輪所有候選梗圖經驗證後均不適用。")
            if outer_attempts_left == 0: break

    logger.info("所有嘗試結束後，仍未找到合適的梗圖。")
    final_text_response = generate_text_only_fallback_response(user_input_text, "試了幾種不同的點子，但好像都沒找到最搭的梗圖耶！")
    return {"text": final_text_response, "image_path": None, "meme_filename": None, "meme_folder": None}


# 主動載入一次資源，如果此模組被匯入
if __name__ != "__main__":
    if not load_all_resources(): # 這裡會嘗試從 URL 載入
        logger.critical("Meme Logic 模組初始化失敗，無法載入必要資源！後續功能可能無法正常運作。")

# 可用於直接測試此模組
if __name__ == "__main__":
    logger.info("開始本地測試 meme_logic.py")
    # 確保在本地測試時，相關環境變數已設定，特別是 ANNOTATIONS_JSON_URL
    # 例如： export ANNOTATIONS_JSON_URL="你的JSON檔案的公開網址"
    #       export MEME_SEARCH_API_URL="你的搜尋服務API網址"
    #       export MEME_DETAILS_API_URL="你的詳細資訊服務API網址"
    #       export GROQ_API_KEY="你的Groq API金鑰"
    #       export CLOUD_MEME_BASE_URL="你的雲端圖片庫基礎網址" (雖然這裡不直接用，但 line_bot 會用)


    if load_all_resources():
        print("資源載入成功，可以開始測試 get_meme_reply。")
        print(f"註釋檔案將從: {ANNOTATIONS_JSON_URL} 載入")
        print(f"梗圖搜尋將嘗試呼叫 API: {MEME_SEARCH_API_URL}")
        print(f"梗圖詳細資訊將嘗試從快取或 API: {MEME_DETAILS_API_URL} 獲取")
        
        test_input = input("請輸入測試文字：")
        if test_input:
            reply = get_meme_reply(test_input)
            print("\n--- 測試回覆 ---")
            print(f"文字: {reply['text']}")
            if reply['meme_filename']:
                print(f"梗圖檔案: {reply['meme_filename']}")
                print(f"梗圖資料夾: {reply.get('meme_folder')}")
                print(f"提示：如需顯示圖片，請自行使用檔名 '{reply['meme_filename']}' 和資料夾 '{reply.get('meme_folder')}' 配合雲端儲存路徑。")
            else:
                print("無梗圖回覆。")
    else:
        print("資源載入失敗，無法執行測試。請檢查 ANNOTATIONS_JSON_URL 環境變數是否已設定並可公開存取。")

