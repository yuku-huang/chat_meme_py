import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

# --- 設定 ---
ANNOTATION_FILE = 'meme_annotations_enriched.json' # 包含 embedding_text 的標註檔
INDEX_FILE = 'faiss_index.index' # 要儲存的 FAISS 索引檔名
MAPPING_FILE = 'index_to_filename.json' # 索引 ID 到檔案名稱的對應檔
# 選擇一個嵌入模型，'paraphrase-multilingual-MiniLM-L12-v2' 是個不錯的多語言選項
# 你也可以選擇其他支援中文的模型，例如 Hugging Face 上的 'uer/sbert-base-chinese-nli'
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
# --- 設定結束 ---

def load_annotations(filepath):
    """載入 JSON 標註檔"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功從 {filepath} 載入 {len(data)} 筆標註。")
        return data
    except FileNotFoundError:
        print(f"錯誤：找不到標註檔 {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"錯誤：無法解析 JSON 檔案 {filepath}")
        return None

def generate_embeddings(texts, model_name):
    """使用指定的 Sentence Transformer 模型生成文字嵌入"""
    print(f"正在載入嵌入模型: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        print("模型載入完成。開始生成嵌入...")
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        print(f"成功生成 {len(embeddings)} 筆嵌入。")
        return embeddings
    except Exception as e:
        print(f"生成嵌入時發生錯誤: {e}")
        return None

def build_faiss_index(embeddings):
    """使用 FAISS 建立向量索引"""
    if embeddings is None or len(embeddings) == 0:
        print("錯誤：沒有有效的嵌入來建立索引。")
        return None

    dimension = embeddings.shape[1] # 取得向量維度
    print(f"向量維度: {dimension}")

    # 建立一個基本的 IndexFlatL2 索引 (使用 L2 距離)
    # 對於大量資料，可以考慮更高效的索引類型，如 IndexIVFFlat
    index = faiss.IndexFlatL2(dimension)

    # 將向量加入索引
    index.add(embeddings.astype('float32')) # FAISS 需要 float32

    print(f"成功建立 FAISS 索引，包含 {index.ntotal} 個向量。")
    return index

def save_index_and_mapping(index, index_to_filename, index_filepath, mapping_filepath):
    """儲存 FAISS 索引和 ID 對應檔"""
    try:
        faiss.write_index(index, index_filepath)
        print(f"FAISS 索引已儲存至: {index_filepath}")
    except Exception as e:
        print(f"儲存 FAISS 索引時發生錯誤: {e}")

    try:
        with open(mapping_filepath, 'w', encoding='utf-8') as f:
            json.dump(index_to_filename, f, ensure_ascii=False, indent=4)
        print(f"索引 ID 對應檔已儲存至: {mapping_filepath}")
    except Exception as e:
        print(f"儲存 ID 對應檔時發生錯誤: {e}")

if __name__ == "__main__":
    print("--- 開始建立梗圖向量索引 ---")

    # 1. 載入標註
    meme_data = load_annotations(ANNOTATION_FILE)
    if not meme_data:
        exit()

    # 2. 準備文字和對應的檔案名稱
    texts_to_embed = []
    filenames = []
    for fname, info in meme_data.items():
        embedding_text = info.get('embedding_text')
        if embedding_text and isinstance(embedding_text, str) and embedding_text.strip():
            texts_to_embed.append(embedding_text.strip())
            filenames.append(fname)
        else:
            print(f"警告：跳過檔案 '{fname}'，因為缺少有效 'embedding_text'。")

    if not texts_to_embed:
        print("錯誤：沒有找到任何有效的 'embedding_text' 來生成嵌入。")
        exit()

    print(f"準備為 {len(texts_to_embed)} 個梗圖生成嵌入。")

    # 3. 生成嵌入向量
    embeddings = generate_embeddings(texts_to_embed, EMBEDDING_MODEL_NAME)
    if embeddings is None:
        exit()

    # 4. 建立 FAISS 索引
    index = build_faiss_index(embeddings)
    if index is None:
        exit()

    # 5. 建立索引 ID 到檔案名稱的對應關係
    # FAISS 索引中的 ID 是從 0 開始的連續整數，對應加入時的順序
    index_to_filename_map = {i: fname for i, fname in enumerate(filenames)}

    # 6. 儲存索引和對應檔
    save_index_and_mapping(index, index_to_filename_map, INDEX_FILE, MAPPING_FILE)

    print("--- 梗圖向量索引建立完成 ---")
