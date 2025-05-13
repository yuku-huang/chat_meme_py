import json
import subprocess
import os
from sentence_transformers import SentenceTransformer, util

# 讀取標注資料
def load_meme_annotations(path='meme_annotations.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Ollama 呼叫函式，產生吐槽句子
def generate_initial_reply(user_text, model="gemmapro"):
    prompt = f"""
你是一個幽默的吐槽助手，請根據使用者輸入產生一段吐槽句子。

使用者輸入："{user_text}"

請直接輸出吐槽句子，不要其他多餘說明。
"""
    return ollama_query(prompt, model)

# Ollama 執行指令
def ollama_query(prompt, model="gemmapro"):
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        if result.returncode != 0:
            print("Error:", result.stderr)
            return None
        return result.stdout.strip()
    except Exception as e:
        print("Exception:", e)
        return None

# 用 sentence-transformers 做向量搜尋
def semantic_search(meme_data, query, top_k=3):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # 建立 corpus：把 title+keywords 合併成描述
    corpus = []
    filenames = []
    for fname, info in meme_data.items():
        desc = info.get('title', '') + ' ' + ' '.join(info.get('keywords', []))
        corpus.append(desc)
        filenames.append(fname)

    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)

    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = cos_scores.topk(k=top_k)

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append({
            'filename': filenames[idx],
            'description': corpus[idx],
            'score': score.item()
        })
    return results

# 取得完整路徑
def get_meme_path(filename, meme_dir='memes'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, meme_dir, filename)

# 主流程
if __name__ == "__main__":
    meme_data = load_meme_annotations('meme_annotations.json')

    user_input = input("請輸入文字：")

    # 1. LLM 產生吐槽句子
    initial_reply = generate_initial_reply(user_input)
    print("初步吐槽句子：", initial_reply)

    if not initial_reply:
        print("沒產生吐槽句子，結束。")
        exit()

    # 2. 用吐槽句子做向量搜尋找梗圖
    matched_memes = semantic_search(meme_data, initial_reply, top_k=3)
    print(f"找到 {len(matched_memes)} 張相似梗圖：")
    for meme in matched_memes:
        print(f"- {meme['filename']} ({meme['score']:.4f}) 描述：{meme['description']}")

    # 3. 選擇相似度最高的梗圖（可改成隨機或其他策略）
    best_meme = matched_memes[0]
    meme_path = get_meme_path(best_meme['filename'])

    print("\n=== 最終回覆 ===")
    print(f"吐槽句子：{initial_reply}")
    print(f"選擇梗圖：{best_meme['filename']}")
    print(f"梗圖路徑：{meme_path}")