# search_service.py
import os
import json
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify

# --- 基本設定 ---
# 假設這些檔案與 search_service.py 在 Render 部署時位於同一目錄
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTATION_FILE = os.path.join(BASE_DIR, 'meme_annotations_enriched.json')
INDEX_FILE = os.path.join(BASE_DIR, 'faiss_index.index')
MAPPING_FILE = os.path.join(BASE_DIR, 'index_to_filename.json')
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2' # 與建立索引時相同

# --- 初始化 Flask 應用 ---
app = Flask(__name__)

# --- 設定 Logger ---
# Render.com 通常會處理日誌輸出，但本地測試時可以設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 全域變數，用於快取載入的資源 ---
embedding_model_cache = None
faiss_index_cache = None
index_to_filename_map_cache = None
all_meme_annotations_cache = None

def load_all_search_resources():
    """在應用程式啟動時載入所有必要的搜尋資源。"""
    global embedding_model_cache, faiss_index_cache, index_to_filename_map_cache, all_meme_annotations_cache
    
    if embedding_model_cache and faiss_index_cache and index_to_filename_map_cache and all_meme_annotations_cache:
        logger.info("所有搜尋資源已從快取載入。")
        return True

    logger.info("--- 開始載入搜尋服務資源 ---")
    try:
        logger.info(f"正在載入嵌入模型: {EMBEDDING_MODEL_NAME}...")
        embedding_model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("嵌入模型載入完成。")

        logger.info(f"正在載入 FAISS 索引從: {INDEX_FILE}...")
        if not os.path.exists(INDEX_FILE):
            logger.error(f"FAISS 索引檔案 {INDEX_FILE} 未找到！")
            return False
        faiss_index_cache = faiss.read_index(INDEX_FILE)
        logger.info(f"FAISS 索引載入完成，包含 {faiss_index_cache.ntotal} 個向量。")

        logger.info(f"正在載入索引 ID 對應檔從: {MAPPING_FILE}...")
        if not os.path.exists(MAPPING_FILE):
            logger.error(f"索引 ID 對應檔案 {MAPPING_FILE} 未找到！")
            return False
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            mapping_raw = json.load(f)
            index_to_filename_map_cache = {int(k): v for k, v in mapping_raw.items()}
        logger.info("索引 ID 對應檔載入完成。")

        logger.info(f"正在載入完整標註檔從: {ANNOTATION_FILE}...")
        if not os.path.exists(ANNOTATION_FILE):
            logger.error(f"完整標註檔案 {ANNOTATION_FILE} 未找到！")
            return False
        with open(ANNOTATION_FILE, 'r', encoding='utf-8') as f:
            all_meme_annotations_cache = json.load(f)
        logger.info(f"成功從 {ANNOTATION_FILE} 載入 {len(all_meme_annotations_cache)} 筆完整標註。")
        
        logger.info("--- 搜尋服務資源載入完成 ---")
        return True
        
    except Exception as e:
        logger.error(f"載入搜尋資源時發生錯誤: {e}", exc_info=True)
        return False

# 在 Flask 應用程式啟動時執行一次資源載入
# 使用 app.before_first_request (舊版 Flask) 或其他方式確保只執行一次
# 對於 Gunicorn，這通常在 worker 啟動時執行
resources_loaded_successfully = load_all_search_resources()
if not resources_loaded_successfully:
    logger.critical("搜尋服務核心資源載入失敗，API 可能無法正常運作！")

@app.route("/")
def home():
    """健康檢查端點或基本資訊頁面。"""
    if resources_loaded_successfully:
        return jsonify({
            "status": "ok",
            "message": "Meme Search Service is running.",
            "resources_status": "loaded",
            "index_size": faiss_index_cache.ntotal if faiss_index_cache else "N/A",
            "annotations_count": len(all_meme_annotations_cache) if all_meme_annotations_cache else "N/A"
        }), 200
    else:
        return jsonify({
            "status": "error",
            "message": "Meme Search Service is NOT running properly due to resource loading failure."
        }), 500


@app.route("/search", methods=['POST'])
def search_memes_api():
    """
    API 端點，接收查詢文字和數量 k，回傳相似的梗圖檔案名稱。
    請求格式: {"query_text": "some query", "k": 5}
    回應格式: {"results": [{"filename": "meme1.jpg", "distance": 0.xx}, ...]}
    """
    if not resources_loaded_successfully:
        return jsonify({"error": "Service resources not loaded. Cannot perform search."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    query_text = data.get("query_text")
    k = data.get("k", 3) # 預設回傳 3 個

    if not query_text:
        return jsonify({"error": "Missing 'query_text' in request body."}), 400
    if not isinstance(k, int) or k <= 0:
        return jsonify({"error": "'k' must be a positive integer."}), 400

    logger.info(f"收到搜尋請求: query='{query_text[:50]}...', k={k}")

    try:
        if embedding_model_cache is None or faiss_index_cache is None or index_to_filename_map_cache is None:
            logger.error("搜尋資源未正確初始化。")
            return jsonify({"error": "Search resources not initialized."}), 500

        query_vector = embedding_model_cache.encode([query_text], convert_to_numpy=True)
        query_vector = query_vector.astype('float32')
        
        distances, indices = faiss_index_cache.search(query_vector, k)
        
        results = []
        if indices.size > 0:
            found_indices = indices[0]
            found_distances = distances[0]
            for i, idx in enumerate(found_indices):
                if idx != -1 and idx in index_to_filename_map_cache:
                    filename = index_to_filename_map_cache[idx]
                    # 你可以選擇是否回傳 distance
                    results.append({'filename': filename, 'distance': float(found_distances[i])}) 
                else:
                    logger.warning(f"在對應表中找不到索引 ID {idx} 或索引無效。")
        
        logger.info(f"搜尋完成，找到 {len(results)} 個結果。")
        return jsonify({"results": results}), 200

    except Exception as e:
        logger.error(f"執行梗圖搜尋時發生錯誤: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during search."}), 500

@app.route("/details", methods=['GET'])
def get_meme_details_api():
    """
    API 端點，接收梗圖檔案名稱，回傳其詳細註釋。
    請求格式: /details?filename=meme1.jpg
    回應格式: (meme_annotations_enriched.json 中對應檔案的內容) 或 {"error": "..."}
    """
    if not resources_loaded_successfully or all_meme_annotations_cache is None:
        return jsonify({"error": "Service resources not loaded. Cannot fetch details."}), 503

    filename = request.args.get("filename")
    if not filename:
        return jsonify({"error": "Missing 'filename' query parameter."}), 400

    logger.info(f"收到詳細資訊請求 for: {filename}")
    
    details = all_meme_annotations_cache.get(filename)
    
    if details:
        return jsonify(details), 200
    else:
        logger.warning(f"找不到檔案 '{filename}' 的詳細資訊。")
        return jsonify({"error": f"Details not found for filename: {filename}"}), 404

if __name__ == "__main__":
    # Render.com 會使用 Gunicorn，所以這部分主要用於本地開發測試
    # 注意：在 Render 上，PORT 環境變數會由平台設定
    port = int(os.environ.get("PORT", 8080)) 
    # debug=True 僅用於開發，部署時應為 False 或由 Gunicorn 控制
    app.run(host="0.0.0.0", port=port, debug=False)
