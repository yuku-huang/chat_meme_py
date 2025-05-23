# meme_logic.py
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import logging # 使用 logging 模組
import re
import time
from typing import Dict, Optional
import random

# --- 初始化 Logger ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- API 金鑰管理 ---
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
        """從環境變數或設定檔載入 API 金鑰"""
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
        """輪詢取得下一個可用的 API 金鑰（所有任務共用）"""
        if not self.api_keys:
            return None
        with self._index_lock:
            key = self.api_keys[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.api_keys)
        return key

# 初始化 API 金鑰管理器
api_key_manager = APIKeyManager()

# --- 基本設定 ---
# 建議將這些路徑設為絕對路徑或相對於此檔案的路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTATION_FILE = os.path.join(BASE_DIR, 'meme_annotations_enriched.json')
INDEX_FILE = os.path.join(BASE_DIR, 'faiss_index.index')
MAPPING_FILE = os.path.join(BASE_DIR, 'index_to_filename.json')
# MEME_ROOT_DIR = os.path.join(BASE_DIR, 'memes') # 梗圖圖片的根目錄 - 不再由此處提供給 LINE

EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_FOR_BOT", "meta-llama/llama-4-scout-17b-16e-instruct") # 從環境變數讀取模型名稱

# --- 新增/修改的常數 ---
NUM_CANDIDATE_REPLIES = 3  # 要生成多少個候選文字回覆
NUM_MEMES_PER_REPLY_SEARCH = 3 # 每個候選回覆要搜尋多少張梗圖
MAX_OUTER_LOOP_ATTEMPTS = 2    # 如果第一輪找不到滿意的梗圖，總共要嘗試幾輪
MIN_RELEVANCE_SCORE_FOR_ACCEPTANCE = 3 # AI評審認為梗圖相關的最低分數
MIN_HUMOR_FIT_SCORE_FOR_ACCEPTANCE = 3 # AI評審認為梗圖幽默契合的最低分數
# NUM_INITIAL_SEARCH_RESULTS = 3 # 這個常數被 NUM_MEMES_PER_REPLY_SEARCH 取代
# MAX_REFINEMENT_ATTEMPTS = 1 # 這個常數被 MAX_OUTER_LOOP_ATTEMPTS 的新邏輯取代

# --- 全域變數，用於快取載入的資源 ---
all_meme_annotations_cache = None
faiss_index_cache = None
index_to_filename_map_cache = None
embedding_model_cache = None
groq_clients_cache: Dict[str, Groq] = {}

def get_groq_client(task_type: str = 'default') -> Optional[Groq]:
    """取得 Groq Client 實例 (帶快取)"""
    global groq_clients_cache
    
    # 檢查快取中是否已有此任務類型的客戶端
    if task_type in groq_clients_cache:
        return groq_clients_cache[task_type]

    try:
        api_key = api_key_manager.get_next_key(task_type)
        if not api_key:
            logger.error(f"無法為任務類型 '{task_type}' 獲取 API 金鑰")
            return None

        client = Groq(api_key=api_key)
        groq_clients_cache[task_type] = client
        logger.info(f"為任務類型 '{task_type}' 初始化新的 Groq Client")
        return client
    except Exception as e:
        logger.error(f"初始化 Groq Client 失敗: {e}")
        return None

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


def query_groq_api(messages, model_name=GROQ_MODEL_NAME, temperature=0.7, max_tokens=1024, is_json_output=False, task_type: str = 'default'):
    """使用 Groq API 執行查詢並取得輸出"""
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
            if "rate_limit_exceeded" in str(e).lower() or "authentication_error" in str(e).lower():
                retry_count += 1
                if retry_count < max_retries:
                    # 當遇到速率限制或認證錯誤時，切換到下一個 API 金鑰
                    client = get_groq_client(task_type)
                    if not client:
                        logger.error("無法獲取新的 API 金鑰")
                        break
                    
                    wait_time = 2 ** retry_count
                    logger.warning(f"達到速率限制或認證錯誤，等待 {wait_time} 秒後使用新的 API 金鑰重試...")
                    time.sleep(wait_time)
                    continue
            logger.error(f"執行 Groq API 查詢時發生錯誤: {e}")
            if is_json_output:
                return {"error": str(e)}
            return None

    logger.error(f"在 {max_retries} 次重試後仍然失敗。最後錯誤: {last_error}")
    if is_json_output:
        return {"error": f"Max retries exceeded: {str(last_error)}"}
    return None

def generate_multiple_candidate_replies(user_text, num_replies=NUM_CANDIDATE_REPLIES):
    """請 Groq API 生成多個候選的幽默回覆文字"""
    system_prompt_instruction = f"""
你是個反應快又幽默的聊天夥伴。針對使用者的輸入，請生成 {num_replies} 個不同的、簡短且風趣的文字回覆。
每個回覆都應該是針對使用者輸入的一個潛在回應。
請以 JSON 格式輸出，包含一個名為 replies 的鍵，其值為一個包含這些回覆字串的列表。
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


def analyze_response_description(user_text, previous_attempts=None):
    """請 Groq API 分析使用者輸入，生成梗圖搜尋描述"""
    system_prompt_instruction = """
你是梗圖搜尋專家。你的任務是分析使用者的輸入，並生成一個簡短但精確的描述，用於搜尋最適合的梗圖。
請以 JSON 格式輸出，包含以下鍵：
- search_description: 字串，用於搜尋梗圖的簡短描述。
- reasoning: 字串，簡要說明為什麼這個描述適合用來搜尋梗圖。
"""
    previous_attempts_str = ""
    if previous_attempts:
        previous_attempts_str = f"之前的搜尋嘗試：\n{json.dumps(previous_attempts, ensure_ascii=False, indent=2)}\n"

    user_prompt_content = (
        f"請分析以下使用者輸入，並生成一個適合用來搜尋梗圖的描述：\n\n"
        f"使用者輸入：\n{user_text}\n\n"
        f"{previous_attempts_str}"
        f"請生成一個簡短但精確的描述，用於搜尋最適合的梗圖。\n"
    )
    messages_for_groq = [
        {"role": "system", "content": system_prompt_instruction},
        {"role": "user", "content": user_prompt_content}
    ]
    logger.info(f"請求 Groq 分析使用者輸入並生成搜尋描述: {user_text[:50]}...")
    analysis_result = query_groq_api(messages_for_groq, model_name=GROQ_MODEL_NAME, temperature=0.5, is_json_output=True, task_type='analyze_response')

    if analysis_result and not analysis_result.get("error") and "search_description" in analysis_result:
        logger.info(f"Groq API 分析結果: {analysis_result}")
        return analysis_result["search_description"]
    else:
        logger.warning(f"無法從 Groq API 取得有效的分析結果。回應: {analysis_result}")
        return None

def validate_meme_choice(user_text, meme_info):
    """Ask Groq API to evaluate whether the selected meme is suitable"""
    system_prompt_instruction = """
You are a meme quality control expert and humor evaluator. Your task is to critically and honestly assess whether a retrieved meme is truly excellent, humorous, and highly suitable for responding to the user's statement.
Pay special attention to whether the meme's concept aligns with the context of the user's conversation, ensuring the dialogue is coherent and not disjointed.

Your output must be in JSON format, including the following keys:
- relevance_score: A number from 1 to 5 representing the relevance of the meme to the user's statement (5 being the highest).
- humor_fit_score: A number from 1 to 5 representing the humor fit of the meme in this specific context (5 being the highest).
- is_suitable: A string, Yes or No, indicating whether the meme is suitable to be sent.
- justification: A string briefly explaining your reasoning. If is_suitable is No, explain why it is not suitable and clearly suggest a better meme concept or search direction. If is_suitable is Yes, you can provide a short positive comment.
- alternative_search_description: A string provided only if is_suitable is No and you can offer a specific meme description for re-searching. If no specific description can be provided, this key can be omitted or left empty.
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
        return validation_result
    else:
        logger.warning(f"Failed to obtain valid evaluation result from Groq API. Response: {validation_result}")
        return {"is_suitable": "No", "relevance_score": 1, "humor_fit_score": 1, "justification": "Failed to obtain valid evaluation result."}

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

現在，請你大展身手，生成一段回覆文字：
1.  緊密結合使用者說的話和梗圖的內涵。
2.  發揮創意：如果梗圖上有文字且可以直接使用，那很好！如果梗圖是圖像為主，或其文字不直接適用，請用你自己的話，巧妙地把梗圖的意境、情緒、或它最精髓的那個點給講出來，讓使用者能 get 到為什麼這個梗圖適合這個情境。
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
你可以稍微提及一下為什麼這次沒有梗圖（例如：{reason_for_no_meme}），但重點還是放在幽默地回應使用者，但是回應不要太長，2句話以內，最好1句話。
"""
    messages_for_groq = [
        {"role": "system", "content": system_prompt_instruction},
        {"role": "user", "content": user_text}
    ]
    logger.info(f"請求 Groq 生成純文字備案回應 (原因: {reason_for_no_meme})")
    fallback_text = query_groq_api(messages_for_groq, model_name=GROQ_MODEL_NAME, task_type='generate_replies')
    if fallback_text:
        logger.info(f"Groq API 純文字備案回應: {fallback_text}")
    else:
        logger.warning("無法從 Groq API 取得純文字備案回應。")
        fallback_text = "嗯... 我今天好像不太幽默，連梗都想不出來了。"
    return fallback_text

def search_similar_memes_faiss(query_text, k=NUM_MEMES_PER_REPLY_SEARCH):
    """在 FAISS 索引中搜尋最相似的 k 個結果"""
    if not query_text or faiss_index_cache is None or embedding_model_cache is None or index_to_filename_map_cache is None:
        logger.error("FAISS 搜尋前缺少必要元素（查詢文字、索引、模型或對應表）。")
        return []
    try:
        logger.info(f"FAISS 搜尋：正在生成查詢向量 for: {query_text[:50]}...")
        query_vector = embedding_model_cache.encode([query_text], convert_to_numpy=True)
        query_vector = query_vector.astype('float32')
        logger.info("FAISS 搜尋：查詢向量生成完畢。")

        logger.info(f"FAISS 搜尋：正在搜尋前 {k} 個最相似的梗圖...")
        distances, indices = faiss_index_cache.search(query_vector, k)
        logger.info("FAISS 搜尋：搜尋完成。")

        results = []
        if indices.size > 0:
            found_indices = indices[0]
            for i, idx in enumerate(found_indices):
                if idx != -1 and idx in index_to_filename_map_cache:
                    filename = index_to_filename_map_cache[idx]
                    results.append({'filename': filename, 'index_id': int(idx)})
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
    return os.path.join(BASE_DIR, 'memes', folder, filename)


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
    
    outer_attempts_left = MAX_OUTER_LOOP_ATTEMPTS

    while outer_attempts_left > 0:
        logger.info(f"--- 開始第 {MAX_OUTER_LOOP_ATTEMPTS - outer_attempts_left + 1} 輪梗圖搜尋與評估 ---")
        outer_attempts_left -= 1

        candidate_reply_texts = generate_multiple_candidate_replies(user_input_text, NUM_CANDIDATE_REPLIES)

        if not candidate_reply_texts:
            logger.warning("無法生成候選回覆文字。")
            if outer_attempts_left == 0: # 如果是最後一輪嘗試，則跳出
                break
            continue # 嘗試下一輪

        all_potential_memes_with_details = []
        for reply_text in candidate_reply_texts:
            logger.info(f"以候選回覆搜尋梗圖: {reply_text[:50]}...")
            similar_memes_found = search_similar_memes_faiss(reply_text, k=NUM_MEMES_PER_REPLY_SEARCH)
            
            for meme_summary in similar_memes_found:
                meme_details = get_meme_details(meme_summary['filename'])
                if meme_details:
                    # 建立完整的梗圖資訊供驗證函式使用
                    full_meme_info = {
                        'filename': meme_summary['filename'],
                        'meme_description': meme_details.get('meme_description', ''),
                        'core_meaning_sentiment': meme_details.get('core_meaning_sentiment', ''),
                        'typical_usage_context': meme_details.get('typical_usage_context', ''),
                        'embedding_text': meme_details.get('embedding_text', ''),
                        # 保留原始資料夾資訊以便後續取得路徑
                        'folder': meme_details.get('folder', '') 
                    }
                    all_potential_memes_with_details.append({
                        'meme_info': full_meme_info,
                        'source_reply_text': reply_text # 記錄這個梗圖是從哪個候選回覆來的
                    })
                else:
                    logger.warning(f"在標註檔中找不到檔案 '{meme_summary['filename']}' 的詳細資訊。")

        if not all_potential_memes_with_details:
            logger.info("本輪未找到任何潛在梗圖。")
            if outer_attempts_left == 0:
                break
            continue

        validated_candidates = []
        for potential in all_potential_memes_with_details:
            validation = validate_meme_choice(user_input_text, potential['meme_info']) # 用原始使用者輸入驗證
            validated_candidates.append({
                'validation_result': validation,
                'meme_info': potential['meme_info'],
                'source_reply_text': potential['source_reply_text']
            })
        
        # 排序，優先選擇 is_suitable="Yes"，然後是分數高的
        validated_candidates.sort(
            key=lambda vc: (
                vc['validation_result'].get('is_suitable', 'No') != 'Yes', # "No" or other will be True (sorts later)
                -(vc['validation_result'].get('relevance_score', 0) + vc['validation_result'].get('humor_fit_score', 0)) # Desc score
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
            selected_meme_folder = chosen_meme_info.get('folder') # 從 meme_info 中取得 folder

            if selected_meme_folder:
                # meme_local_path 不再需要給 line_bot.py, 但如果本地測試需要可以保留計算
                # meme_local_path = get_meme_image_path(selected_meme_filename, selected_meme_folder)
                pass # 不再設定 meme_local_path 給 line_bot.py
            else:
                logger.warning(f"梗圖 '{selected_meme_filename}' 在標註檔中缺少 'folder' 資訊，無法取得圖片路徑。")
                # meme_local_path = None # 確保 image_path 為 None
            
            return {"text": final_text_response, "image_path": None, "meme_filename": selected_meme_filename}
        else:
            logger.info("本輪所有候選梗圖經驗證後均不適用。")
            if outer_attempts_left == 0: # 如果是最後一輪嘗試，則跳出
                break
            # 否則，外層 while 迴圈會繼續下一輪

    # 如果所有輪次都沒有找到合適的梗圖
    logger.info("所有嘗試結束後，仍未找到合適的梗圖。")
    final_text_response = generate_text_only_fallback_response(user_input_text, "試了幾種不同的點子，但好像都沒找到最搭的梗圖耶！")
    return {"text": final_text_response, "image_path": None, "meme_filename": None} # 確保這裡也回傳 None


# 主動載入一次資源，如果此模組被匯入
if __name__ != "__main__": # 當作模組匯入時執行
    if not load_all_resources():
        logger.critical("Meme Logic 模組初始化失敗，無法載入必要資源！")

# 可用於直接測試此模組
if __name__ == "__main__":
    # 為了讓 query_groq_api 中的 JSON 提取正常運作，如果它使用了 re 模組
    import re 
    # 為了測試時顯示圖片
    try:
        from PIL import Image
    except ImportError:
        print("Pillow 函式庫未安裝，無法在測試中顯示圖片。請執行 pip install Pillow")
        Image = None


    if load_all_resources():
        print("資源載入成功，可以開始測試 get_meme_reply。")
        
        # 測試 generate_multiple_candidate_replies
        # print("\n--- 測試 generate_multiple_candidate_replies ---")
        # test_replies = generate_multiple_candidate_replies("今天天氣真好")
        # if test_replies:
        #     for i, r_text in enumerate(test_replies):
        #         print(f"候選回覆 {i+1}: {r_text}")
        # else:
        #     print("無法生成候選回覆。")

        test_input = input("請輸入測試文字：")
        if test_input:
            reply = get_meme_reply(test_input)
            print("\n--- 測試回覆 ---")
            print(f"文字: {reply['text']}")
            if reply['image_path']:
                print(f"梗圖檔案: {reply['meme_filename']}")
                print(f"梗圖路徑: {reply['image_path']}")
                if Image:
                    try:
                        if reply['image_path'] and os.path.exists(reply['image_path']):
                            img = Image.open(reply['image_path'])
                            img.show()
                        elif reply['meme_filename']:
                            print(f"本地測試提示：雖然 image_path 為 None，但梗圖檔名為 {reply['meme_filename']}。如需本地顯示，請自行處理路徑。")
                        else:
                            print(f"測試警告：圖片路徑不存在 {reply['image_path']}")
                    except Exception as e:
                        print(f"測試顯示圖片錯誤: {e}")
            else:
                print("無梗圖回覆。")
    else:
        print("資源載入失敗，無法執行測試。")
