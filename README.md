# MemeBot - 使用 Flask 和 LINE Messaging API 的梗圖回覆機器人

## 專案概述
這是我python通識課的期末專案。
MemeBot 是一個使用 Flask 框架與 LINE Messaging API 所構建的聊天機器人，能夠結合 Groq API 進行自然語言生成和評估，並透過 FAISS 進行向量搜尋，推薦與使用者對話最契合的梗圖。

主要功能：
- 根據使用者輸入自動生成多個候選回覆文字
- 使用 FAISS 向量搜尋在本地梗圖資料庫中找到最相似的梗圖
- 調用 Groq API 為每張候選梗圖進行適合度評分，挑選最合適的梗圖
- 支援本地圖片回傳或純文字備案回應

## 目前進度

### 已完成

1. 安裝並設定 ngrok，將本地服務暴露給 LINE Webhook，並取得公開 URL。
2. 設定環境變數：
   - `LINE_CHANNEL_SECRET`
   - `LINE_CHANNEL_ACCESS_TOKEN`
   - `APP_BASE_URL`（Ngrok URL）
3. 最佳化機器人程式碼以支持本地與 static 圖片回覆。
4. 對圖片 URL 中的空格與特殊字元使用 `urllib.parse.quote` 進行編碼處理。
5. 錯誤處理：
   - 401 Unauthorized（重新產生 Channel Access Token 並更新程式）
   - 400 Bad Request（忽略重送訊息 `isRedelivery`，確認 Webhook URL 設定）
6. Groq API 速率限制（429 Rate Limit）處理：
   - 實作重試機制（最多重試 3 次、指數退避）
   - 多金鑰管理：
     - 初始版本根據任務類型分配固定金鑰
     - 升級為全域輪詢，並隨機初始化起始索引
     - 最終改為遇到金鑰失效或速率限制即切換至下一把金鑰
7. 修正 `meme_logic.py` 中 f-string 語法錯誤（反斜線、三引號）並清理 Prompt 格式。
8. 翻譯所有中文 Prompt 為英文，並將 `validate_meme_choice` 中的 Prompt 更新為英文版。

## 技術棧

- 程式語言：Python 3.x
- Web 框架：Flask
- LINE SDK：line-bot-sdk v3
- Ngrok：本地端公開 URL
- 向量搜尋：FAISS
- 嵌入模型：Sentence-Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- AI 生成與評估：Groq Python SDK
- URL 編碼：`urllib.parse`
- 日誌：`logging`
- 圖片處理（選用）：Pillow
- 其他：`json`, `os`, `re`, `time`, `random`

## 檔案結構

```
/ (專案根目錄)
├─ app.py               # Flask 服務啟動程式
├─ meme_logic.py        # 梗圖生成與驗證核心邏輯
├─ requirements.txt     # 相依套件清單
├─ meme_annotations_enriched.json  # 梗圖註釋資料
├─ faiss_index.index    # FAISS 向量索引檔
├─ index_to_filename.json  # 索引 ID 對應檔案映射
├─ memes/               # 梗圖圖片資料夾
│  └─ ...
└─ README.md            # 專案說明文件
```

## 安裝與設定

1. 建立虛擬環境並安裝套件

   ```bash
   python -m venv venv
   # Linux/macOS
   source venv/bin/activate
   # Windows PowerShell
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. 設定環境變數

   ```bash
   export LINE_CHANNEL_SECRET="你的 Channel Secret"
   export LINE_CHANNEL_ACCESS_TOKEN="你的 Channel Access Token"
   export APP_BASE_URL="https://你的Ngrok URL"
   export GROQ_API_KEY_1="你的 Groq API Key 1"
   export GROQ_API_KEY_2="你的 Groq API Key 2"  # （如有多把金鑰）
   export GROQ_MODEL_FOR_BOT="meta-llama/llama-4-scout-17b-16e-instruct"
   ```

   Windows PowerShell：

   ```powershell
   setx LINE_CHANNEL_SECRET "你的 Channel Secret"
   setx LINE_CHANNEL_ACCESS_TOKEN "你的 Channel Access Token"
   setx APP_BASE_URL "https://你的Ngrok URL"
   setx GROQ_API_KEY_1 "你的 Groq API Key 1"
   setx GROQ_API_KEY_2 "你的 Groq API Key 2"
   setx GROQ_MODEL_FOR_BOT "meta-llama/llama-4-scout-17b-16e-instruct"
   ```

3. 啟動 Ngrok

   ```bash
   ngrok http 5000
   ```

4. 將 Ngrok 公開 URL 加上 `/callback` 設定為 LINE Webhook URL

5. 啟動 Flask 應用

   ```bash
   python app.py
   ```

## 使用方式

在 LINE 聊天介面向機器人發送文字訊息，MemeBot 會根據對話內容推薦並回傳梗圖，若找不到合適梗圖則提供純文字幽默回應。

## 更新里程碑

- 第一版：完成最基礎的聊天回覆功能，機器人能接收並回應使用者的文字訊息
- 第二版：加入本地梗圖回傳功能，可以把 static 資料夾裡的圖片傳給使用者
- 第三版：修正圖片 URL 內空格和特殊字元編碼問題，確保圖片能正確顯示
- 第四版：新增錯誤處理，遇到 401/400 錯誤會自動更新 Token 並忽略重複訊息
- 第五版：實作 Groq API 多金鑰輪詢和速率限制重試機制，提高請求穩定度
- 第六版：整合 FAISS 向量搜尋與資源快取，讓梗圖搜尋更快速準確
- 第七版：加入 AI 驗證機制，並把所有 Prompt 調整成中英文雙語版，優化推薦品質

## 未來計劃

- 擴充梗圖庫，加入更多不同主題和風格的圖片
- 部署機器人到雲端，確保長期穩定運行
- 建置後台管理介面與儀表板，方便即時查看系統狀態和推薦效果

## 聯絡方式

如有問題或建議，請聯絡：kelly58516776@gmail.com 