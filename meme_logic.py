# meme_logic.py
import json
import os
import logging
import re
import time
import random
from typing import Dict, Optional, List
import requests # 確保 requests 已在 requirements.txt
from sentence_transformers import SentenceTransformer # ADD THIS LINE

# --- 初始化 Logger ---
logger = logging.getLogger(__name__)
# 避免在 Vercel 等環境中重複設定 handler，導致日誌重複輸出
if not logger.hasHandlers():
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
        else:
            logger.warning("未從環境變數中載入任何 Groq API 金鑰。Groq API 呼叫將會失敗。")
        self._initialized = True

    def load_api_keys(self):
        i = 1
        loaded_keys_count = 0
        while True:
            key = os.environ.get(f'GROQ_API_KEY_{i}')
            if not key:
                # 嘗試載入單一金鑰 GROQ_API_KEY (如果 GROQ_API_KEY_1 等不存在)
                if i == 1:
                    single_key = os.environ.get('GROQ_API_KEY')
                    if single_key:
                        self.api_keys.append(single_key)
                        loaded_keys_count += 1
                        logger.info(f"成功載入環境變數 GROQ_API_KEY。")
                break # 如果 GROQ_API_KEY_i 不存在，且 i > 1，或 i=1 但單一金鑰也不存在，則停止
            self.api_keys.append(key)
            logger.info(f"成功載入環境變數 GROQ_API_KEY_{i}。")
            loaded_keys_count += 1
            i += 1
        if loaded_keys_count == 0:
            logger.error("重大錯誤：未找到任何 Groq API 金鑰！請設定 GROQ_API_KEY 或 GROQ_API_KEY_1 等環境變數。")


    def get_next_key(self, task_type: str = 'default') -> Optional[str]:
        if not self.api_keys:
            logger.error(f"任務 '{task_type}' 無可用的 Groq API 金鑰。")
            return None
        with self._index_lock:
            key_to_use = self.api_keys[self.current_index]
            logger.info(f"為任務 '{task_type}' 選用 API 金鑰索引 {self.current_index+1}/{len(self.api_keys)}。")
            self.current_index = (self.current_index + 1) % len(self.api_keys)
        return key_to_use

api_key_manager = APIKeyManager()

# --- 基本設定 ---
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_FOR_BOT", "meta-llama/llama-4-scout-17b-16e-instruct")
ANNOTATIONS_JSON_URL = os.environ.get('ANNOTATIONS_JSON_URL') # 強制從環境變數讀取
MEME_SEARCH_API_URL = os.environ.get('MEME_SEARCH_API_URL')
MEME_DETAILS_API_URL = os.environ.get('MEME_DETAILS_API_URL')

# --- 常數 ---
NUM_CANDIDATE_REPLIES = 3
NUM_MEMES_PER_REPLY_SEARCH = 3
MAX_OUTER_LOOP_ATTEMPTS = 2
MIN_RELEVANCE_SCORE_FOR_ACCEPTANCE = 3
MIN_HUMOR_FIT_SCORE_FOR_ACCEPTANCE = 3

# --- 全域變數 ---
all_meme_annotations_cache = None
groq_clients_cache: Dict[str, 'Groq'] = {}
Groq = None # Groq SDK 的型別提示
embedding_model_for_search_cache = None # ADD THIS LINE
SEARCH_EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2' # ADD THIS LINE

def ensure_groq_imported_and_configured():
    global Groq
    if Groq is None:
        try:
            from groq import Groq as GroqClient
            Groq = GroqClient
            logger.info("Groq SDK 成功匯入。")
        except ImportError:
            logger.error("重大錯誤：Groq SDK 未安裝。請將 'groq' 加入 requirements.txt。")
            return False # 表示匯入失敗

    if not api_key_manager.api_keys: # 檢查是否有載入任何 API 金鑰
        logger.error("重大錯誤：Groq API 金鑰未設定或未載入。無法初始化 Groq Client。")
        return False # 表示金鑰設定失敗
    return True

def get_groq_client(task_type: str = 'default') -> Optional['Groq']:
    global groq_clients_cache
    if not ensure_groq_imported_and_configured(): # 先確保 SDK 和金鑰都就緒
        return None
    # Groq 應該已經被 ensure_groq_imported_and_configured 設定
    if Groq is None: # 再次檢查，理論上不應該發生
        logger.error("Groq SDK 在 ensure_groq_imported_and_configured 後仍為 None。")
        return None

    if task_type in groq_clients_cache and groq_clients_cache[task_type] is not None:
        return groq_clients_cache[task_type]

    try:
        api_key = api_key_manager.get_next_key(task_type)
        if not api_key: # 雖然 ensure_groq_imported_and_configured 已經檢查過，但多一層保護
            logger.error(f"任務 '{task_type}' 無法獲取 API 金鑰 (在 get_groq_client 中)。")
            return None
        
        client = Groq(api_key=api_key)
        groq_clients_cache[task_type] = client
        logger.info(f"為任務類型 '{task_type}' 成功初始化新的 Groq Client。")
        return client
    except Exception as e:
        logger.error(f"初始化 Groq Client 時發生錯誤 (任務 '{task_type}'): {e}", exc_info=True)
        return None

def ensure_embedding_model_loaded():
    """確保用於搜尋的 SentenceTransformer 模型已載入。"""
    global embedding_model_for_search_cache
    if embedding_model_for_search_cache is None:
        try:
            logger.info(f"正在為搜尋服務客戶端載入嵌入模型: {SEARCH_EMBEDDING_MODEL_NAME}...")
            embedding_model_for_search_cache = SentenceTransformer(SEARCH_EMBEDDING_MODEL_NAME)
            logger.info("搜尋服務客戶端的嵌入模型載入完成。")
            return True
        except Exception as e:
            logger.error(f"載入搜尋服務客戶端的嵌入模型 {SEARCH_EMBEDDING_MODEL_NAME} 時發生錯誤: {e}", exc_info=True)
            embedding_model_for_search_cache = None # 確保出錯時快取是 None
            return False
    return True

def load_all_resources():
    global all_meme_annotations_cache
    logger.info("--- 開始載入資源 (meme_logic.py) ---")

    if all_meme_annotations_cache is not None:
        logger.info("梗圖註釋已從快取載入。")
    else:
        if not ANNOTATIONS_JSON_URL: # 檢查環境變數是否設定
            logger.error("重大錯誤：ANNOTATIONS_JSON_URL 環境變數未設定。無法載入梗圖註釋。")
            return False
        
        logger.info(f"正在從 URL 下載註釋檔案: {ANNOTATIONS_JSON_URL}")
        try:
            response = requests.get(ANNOTATIONS_JSON_URL, timeout=20) # 增加超時時間
            response.raise_for_status() 
            all_meme_annotations_cache = response.json()
            logger.info(f"成功從 URL 載入 {len(all_meme_annotations_cache)} 筆完整標註。")
        except requests.exceptions.RequestException as e:
            logger.error(f"從 URL 下載註釋檔案時發生網路錯誤: {e}", exc_info=True)
            all_meme_annotations_cache = None # 確保出錯時快取是 None
            return False
        except json.JSONDecodeError as e:
            logger.error(f"解析從 URL 下載的 JSON 註釋時發生錯誤: {e}", exc_info=True)
            all_meme_annotations_cache = None
            return False
        except Exception as e: # 捕獲其他可能的錯誤
            logger.error(f"載入註釋檔案時發生未預期錯誤: {e}", exc_info=True)
            all_meme_annotations_cache = None
            return False

    # 檢查並初始化 Groq Client
    # 即使之前已經呼叫過 ensure_groq_imported_and_configured，這裡再次獲取 client 以確認
    if get_groq_client('initial_load_check') is None: # 使用特定的任務類型名稱
        logger.error("重大錯誤：Groq Client 在資源載入過程中初始化失敗。可能是 API 金鑰問題。")
        # 即使註釋載入成功，如果 Groq Client 失敗，也應視為整體資源載入失敗
        return False

    logger.info("--- 所有核心資源 (註釋和 Groq Client) 載入/檢查完成 (meme_logic.py) ---")
    return True

# --- 以下是你的核心邏輯函式，保持與之前版本一致，但確保它們都先檢查資源是否已載入 ---

def query_groq_api(messages, model_name=GROQ_MODEL_NAME, temperature=0.7, max_tokens=1024, is_json_output=False, task_type: str = 'default'):
    client = get_groq_client(task_type) # 每次呼叫時獲取 client，它會處理金鑰輪換
    if not client:
        logger.error(f"任務 '{task_type}' 無法獲取 Groq Client。")
        return {"error": "Groq client not available"} if is_json_output else None

    max_retries = 2 # 減少重試次數，因為金鑰輪換應該處理大部分問題
    current_retry = 0
    last_error = None

    while current_retry <= max_retries:
        try:
            logger.info(f"Groq API 請求 (任務: {task_type}, 模型: {model_name}, 第 {current_retry+1} 次嘗試)")
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
            logger.info(f"Groq API 成功回應 (任務: {task_type})")

            if is_json_output:
                try:
                    return json.loads(response_content)
                except json.JSONDecodeError as e:
                    logger.error(f"Groq API (任務: {task_type}) JSON 解析失敗。回應: {response_content[:200]}... 錯誤: {e}")
                    # 嘗試更穩健的 JSON 提取
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL)
                    if not json_match:
                        json_match = re.search(r'(\{.*?\})', response_content, re.DOTALL) # 更寬鬆的匹配

                    if json_match:
                        try:
                            extracted_json_str = json_match.group(1)
                            logger.info(f"從 Groq 回應中提取的 JSON 字串: {extracted_json_str[:200]}...")
                            return json.loads(extracted_json_str)
                        except json.JSONDecodeError as inner_e:
                            logger.error(f"提取的 JSON 字串解析失敗: {inner_e}")
                            return {"error": "Failed to parse extracted JSON response", "raw_response": response_content}
                    return {"error": "Failed to parse JSON response and no valid JSON block found", "raw_response": response_content}
            return response_content
        except Exception as e:
            last_error = e
            logger.warning(f"Groq API 查詢 (任務: {task_type}) 第 {current_retry+1} 次嘗試失敗: {type(e).__name__} - {str(e)}")
            
            # 如果是金鑰相關錯誤或速率限制，get_groq_client 在下次呼叫時會嘗試輪換
            # 對於可重試的錯誤，進行退避
            # 檢查錯誤類型，例如：
            # from groq.types.chat import RateLimitError # 假設的錯誤類型，請查閱 Groq SDK
            # if isinstance(e, RateLimitError) or "authentication_error" in str(e).lower():
            if "rate_limit" in str(e).lower() or "authentication" in str(e).lower() or "key" in str(e).lower():
                 if current_retry < max_retries:
                    client = get_groq_client(task_type) # 嘗試獲取下一個金鑰的 client
                    if not client: # 如果沒有更多金鑰或 client 無法建立
                        logger.error("Groq Client 獲取失敗，終止重試。")
                        break
                    wait_time = (2 ** current_retry) + random.uniform(0, 1)
                    logger.info(f"等待 {wait_time:.2f} 秒後重試...")
                    time.sleep(wait_time)
                 else:
                    logger.error("達到最大重試次數。")
                    break # 不再重試
            else: # 對於其他類型的錯誤，可能不應該重試
                logger.error(f"發生不可重試的 Groq API 錯誤 (任務: {task_type}): {e}", exc_info=True)
                break
            current_retry += 1

    logger.error(f"Groq API 查詢 (任務: {task_type}) 在 {max_retries+1} 次嘗試後最終失敗。最後錯誤: {last_error}", exc_info=True if last_error else False)
    if is_json_output:
        return {"error": f"Groq API request failed after multiple retries: {str(last_error)}"}
    return None


def generate_multiple_candidate_replies(user_text, num_replies=NUM_CANDIDATE_REPLIES):
    # (此函式邏輯與上一版相同，確保它呼叫更新後的 query_groq_api)
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
    # (此函式邏輯與上一版相同，確保它呼叫更新後的 query_groq_api)
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
    logger.info(f"請求 Groq 驗證梗圖選擇：{meme_info.get('filename', 'Unknown file')}")
    validation_result = query_groq_api(messages_for_groq, model_name=GROQ_MODEL_NAME, temperature=0.3, is_json_output=True, task_type='validate_meme')

    if validation_result and not validation_result.get("error"):
        logger.info(f"Groq API 驗證結果: {validation_result}")
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
        logger.warning(f"無法從 Groq API 取得有效的驗證結果。原始回應: {validation_result}")
        return {"is_suitable": "No", "relevance_score": 1, "humor_fit_score": 1, "justification": "Failed to obtain valid evaluation result from AI."}


def generate_final_response_text(user_text, meme_info):
    # (此函式邏輯與上一版相同，確保它呼叫更新後的 query_groq_api)
    meme_filename = meme_info.get('filename', '未知檔案')
    # ... (其餘部分與之前相同)
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
    # (此函式邏輯與上一版相同，確保它呼叫更新後的 query_groq_api)
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
    # (此函式邏輯與上一版相同，但加入對環境變數的檢查)
    if not MEME_SEARCH_API_URL:
        logger.error("重大錯誤：MEME_SEARCH_API_URL 環境變數未設定。無法執行外部梗圖搜尋。")
        return []

    # NEW: Ensure embedding model is loaded and encode query_text
    if not ensure_embedding_model_loaded() or embedding_model_for_search_cache is None:
        logger.error("搜尋嵌入模型未載入，無法執行梗圖搜尋。")
        return []
    
    try:
        query_vector_np = embedding_model_for_search_cache.encode([query_text], convert_to_numpy=True)
        query_vector_list = query_vector_np[0].tolist() # Convert to list of floats
    except Exception as e:
        logger.error(f"使用文字 '{query_text[:50]}...' 生成查詢向量時發生錯誤: {e}", exc_info=True)
        return []

    # payload = {"query_text": query_text, "k": k} # OLD PAYLOAD
    payload = {"query_vector": query_vector_list, "k": k} # NEW PAYLOAD
    logger.info(f"正在呼叫外部梗圖搜尋 API: {MEME_SEARCH_API_URL}，查詢向量長度: {len(query_vector_list)}, k={k}") # UPDATED LOG

    try:
        response = requests.post(MEME_SEARCH_API_URL, json=payload, timeout=15) # 增加超時
        response.raise_for_status()  
        api_results = response.json() 
        
        if "results" in api_results and isinstance(api_results["results"], list):
            logger.info(f"外部 API 成功回傳 {len(api_results['results'])} 個搜尋結果。")
            return api_results["results"] 
        else:
            logger.warning(f"外部 API 回應格式不符預期。回應: {api_results}")
            return []
    except requests.exceptions.Timeout:
        logger.error(f"呼叫外部梗圖搜尋 API 超時 ({MEME_SEARCH_API_URL})。")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"呼叫外部梗圖搜尋 API 時發生錯誤: {e}", exc_info=True)
        return []
    except json.JSONDecodeError as e: # 通常 response.json() 內部會處理，但以防萬一
        logger.error(f"解析外部梗圖搜尋 API 回應時發生 JSON 錯誤: {e}", exc_info=True)
        return []


def get_meme_details_via_api(filename: str) -> Optional[Dict]:
    # (此函式邏輯與上一版相同，但加入對環境變數的檢查)
    global all_meme_annotations_cache
    if all_meme_annotations_cache and filename in all_meme_annotations_cache:
        return all_meme_annotations_cache[filename]

    if not MEME_DETAILS_API_URL:
        logger.warning(f"MEME_DETAILS_API_URL 環境變數未設定，且快取中無 '{filename}' 的資訊。")
        return None

    params = {"filename": filename}
    logger.info(f"正在呼叫外部梗圖詳細資訊 API: {MEME_DETAILS_API_URL}，檔案名: {filename}")
    try:
        response = requests.get(MEME_DETAILS_API_URL, params=params, timeout=10) # 增加超時
        response.raise_for_status()
        meme_details = response.json()

        if meme_details and isinstance(meme_details, dict): 
            logger.info(f"外部 API 成功回傳梗圖 '{filename}' 的詳細資訊。")
            if all_meme_annotations_cache is not None:
                 all_meme_annotations_cache[filename] = meme_details
            return meme_details
        else:
            logger.warning(f"外部 API 為 '{filename}' 回傳的詳細資訊格式不符預期。回應: {meme_details}")
            return None
    except requests.exceptions.Timeout:
        logger.error(f"呼叫外部梗圖詳細資訊 API 超時 ({MEME_DETAILS_API_URL} for {filename})。")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"呼叫外部梗圖詳細資訊 API 時發生錯誤 for '{filename}': {e}", exc_info=True)
        return None
    except json.JSONDecodeError as e:
        logger.error(f"解析外部梗圖詳細資訊 API 回應時發生 JSON 錯誤 for '{filename}': {e}", exc_info=True)
        return None

def get_meme_reply(user_input_text: str) -> Dict:
    # (此函式邏輯與上一版相同，但依賴上面函式的改進)
    # 確保在呼叫此函式前，load_all_resources 已被成功執行
    if all_meme_annotations_cache is None: # 嚴格檢查快取是否已載入
        logger.error("get_meme_reply 被呼叫，但梗圖註釋快取未載入。可能是 load_all_resources 失敗。")
        # 嘗試再次載入，如果應用程式允許這種延遲載入
        # if not load_all_resources():
        return {"text": "抱歉，我內部初始化出錯了，暫時無法服務。", "image_path": None, "meme_filename": None, "meme_folder": None}

    # ... (get_meme_reply 的其餘邏輯與上一版相同)
    chosen_meme_info = None
    final_text_response = None
    selected_meme_filename = None
    selected_meme_folder = None 
    
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

                meme_details = get_meme_details_via_api(filename) 
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
            selected_meme_folder = chosen_meme_info.get('folder') 
            
            return {"text": final_text_response, "image_path": None, "meme_filename": selected_meme_filename, "meme_folder": selected_meme_folder}
        else:
            logger.info("本輪所有候選梗圖經驗證後均不適用。")
            if outer_attempts_left == 0: break

    logger.info("所有嘗試結束後，仍未找到合適的梗圖。")
    final_text_response = generate_text_only_fallback_response(user_input_text, "試了幾種不同的點子，但好像都沒找到最搭的梗圖耶！")
    return {"text": final_text_response, "image_path": None, "meme_filename": None, "meme_folder": None}


# 主動載入一次資源的呼叫應該在 line_bot.py 中進行，以控制應用程式的啟動流程
# if __name__ != "__main__":
#     if not load_all_resources(): 
#         logger.critical("Meme Logic 模組初始化失敗，無法載入必要資源！後續功能可能無法正常運作。")

if __name__ == "__main__":
    # 本地測試時，需要手動設定環境變數
    # 例如： export ANNOTATIONS_JSON_URL="https://..."
    #       export GROQ_API_KEY_1="gsk_..."
    #       export MEME_SEARCH_API_URL="http://localhost:8080/search" (如果你本地也運行 search_service.py)
    logger.info("開始本地測試 meme_logic.py")
    
    # 確保環境變數已設定
    if not os.environ.get('ANNOTATIONS_JSON_URL'):
        print("錯誤：本地測試前，請設定 ANNOTATIONS_JSON_URL 環境變數。")
        exit()
    if not (os.environ.get('GROQ_API_KEY') or os.environ.get('GROQ_API_KEY_1')):
        print("錯誤：本地測試前，請設定 GROQ_API_KEY 或 GROQ_API_KEY_1 環境變數。")
        exit()
    if not os.environ.get('MEME_SEARCH_API_URL'):
        print("警告：本地測試時，MEME_SEARCH_API_URL 未設定。梗圖搜尋將失敗。")
    
    if load_all_resources():
        print("資源載入成功，可以開始測試 get_meme_reply。")
        test_input = input("請輸入測試文字：")
        if test_input:
            reply = get_meme_reply(test_input)
            print("\n--- 測試回覆 ---")
            print(f"文字: {reply['text']}")
            if reply['meme_filename']:
                print(f"梗圖檔案: {reply['meme_filename']}")
                print(f"梗圖資料夾: {reply.get('meme_folder')}")
            else:
                print("無梗圖回覆。")
    else:
        print("資源載入失敗，無法執行測試。請檢查環境變數和日誌。")

