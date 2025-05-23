# from flask import Flask, request, abort
# import logging

# app = Flask(__name__)
# app.logger.setLevel(logging.INFO)

# @app.route("/")
# def hello():
#     app.logger.info("Root path was called.")
#     return "Hello from Flask!"

# @app.route("/callback", methods=['POST'])
# def callback():
#     app.logger.info("Callback received.")
#     # 暫時不處理 LINE 的複雜邏輯
#     return 'OK_SIMPLE'

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5001))
#     app.run(host="0.0.0.0", port=port, debug=False) # 正式部署時 debug=False
# line_bot_app.py
import os
import logging
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage, ImageMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from urllib.parse import quote
import json

# 直接設定環境變數（請替換成您的實際值）
# os.environ['APP_BASE_URL'] = 'https://chat-meme-py.vercel.app'  # 例如：https://xxxx-xxx-xxx-xxx-xxx.ngrok.io

# 從環境變數讀取 LINE Bot 的憑證
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
# 你部署此應用程式的公開網址，用於產生圖片 URL
# 例如：https://your-app-name.onrender.com 或 ngrok 的 URL
# **重要**：結尾不要加斜線 (/)
APP_BASE_URL = os.environ.get('APP_BASE_URL')

# *** 新增：雲端梗圖圖片的基礎 URL ***
# 例如：https://your-cloud-storage-domain.com/memes/
CLOUD_MEME_BASE_URL = os.environ.get('CLOUD_MEME_BASE_URL')


if not LINE_CHANNEL_SECRET or not LINE_CHANNEL_ACCESS_TOKEN:
    logging.error("錯誤：LINE_CHANNEL_SECRET 或 LINE_CHANNEL_ACCESS_TOKEN 未設定！")
    exit()
if not APP_BASE_URL:
    logging.warning("警告：APP_BASE_URL 未設定。圖片可能無法正確顯示。請設定為你應用程式的公開網址。")
if not CLOUD_MEME_BASE_URL: # 修改判斷條件
    logging.warning("警告：CLOUD_MEME_BASE_URL 未設定。圖片可能無法正確顯示。請設定為你雲端儲存的梗圖基礎網址。")


# 初始化 Flask 應用
app = Flask(__name__)

# 設定 LINE Bot
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# 引入你的梗圖邏輯
import meme_logic

# --- 設定 Logger ---
# Flask 已經有自己的 logger，這裡可以調整層級或格式
app.logger.setLevel(logging.INFO)
# 可以將 meme_logic 的 logger 也整合進來或分開處理
meme_logic_logger = logging.getLogger('meme_logic') # 與 meme_logic.py 中定義的 logger 名稱一致
meme_logic_logger.setLevel(logging.INFO)


# 載入梗圖邏輯所需的資源 (應用程式啟動時執行一次)
# 確保在 meme_logic 中有相應的初始化函式，例如 load_all_resources()
if not meme_logic.load_all_resources():
    app.logger.critical("梗圖邏輯資源載入失敗，應用程式可能無法正常運作！")

# Webhook 路徑，LINE Platform 會把事件發送到這裡
@app.route("/callback", methods=['POST'])
def callback():
    # 取得 X-Line-Signature header 值，用於驗證請求
    signature = request.headers['X-Line-Signature']

    # 取得請求主體 (request body) 的文字內容
    body = request.get_data(as_text=True)
    app.logger.info(f"Request body: {body}")

    # 解析 JSON 以檢查是否為重複訊息
    try:
        data = json.loads(body)
        if 'events' in data and len(data['events']) > 0:
            event = data['events'][0]
            # 檢查是否為重複訊息
            if event.get('deliveryContext', {}).get('isRedelivery', False):
                app.logger.info("收到重複訊息，忽略處理")
                return 'OK'
    except json.JSONDecodeError:
        app.logger.error("無法解析請求內容為 JSON")
        abort(400)

    # 處理 webhook 事件
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.error("簽名驗證失敗。請檢查你的 Channel Secret 是否正確。")
        abort(400)
    except Exception as e:
        app.logger.error(f"處理 Webhook 時發生錯誤: {e}")
        abort(500)
    return 'OK'

# 處理文字訊息事件
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
    user_id = event.source.user_id
    text = event.message.text
    reply_token = event.reply_token
    app.logger.info(f"收到來自 {user_id} 的訊息: {text}")

    # 呼叫梗圖邏輯
    reply_data = meme_logic.get_meme_reply(text)
    # reply_data 預期格式: {"text": "回覆文字", "image_path": "本地圖片路徑或None", "meme_filename": "梗圖檔名或None"}

    messages_to_reply = []

    # 準備文字回覆
    if reply_data.get("text"):
        messages_to_reply.append(TextMessage(text=reply_data["text"]))
    else: # 兜底，以防萬一
        messages_to_reply.append(TextMessage(text="我好像有點卡住了，稍等一下喔！"))

    # *** 修改圖片回覆邏輯：從雲端 URL 獲取圖片 ***
    if reply_data.get("meme_filename"): # 只要有檔名，就嘗試構建 URL
        # 從 meme_details 獲取 folder
        meme_details = meme_logic.get_meme_details(reply_data["meme_filename"])
        if meme_details and meme_details.get("folder"):
            meme_folder = meme_details.get("folder")
            # 構建雲端圖片 URL
            # 確保 CLOUD_MEME_BASE_URL 結尾沒有斜線，並且拼接後面的路徑
            # 例如：https://your-cloud-storage.com/memes/SpongeBob/640.jpg
            if CLOUD_MEME_BASE_URL:
                # 確保 CLOUD_MEME_BASE_URL 結尾沒有斜線，而路徑開頭有斜線
                image_url = f"{CLOUD_MEME_BASE_URL.rstrip('/')}/{quote(meme_folder)}/{quote(reply_data['meme_filename'])}"
                app.logger.info(f"準備發送圖片，雲端 URL: {image_url}")
                messages_to_reply.append(ImageMessage(
                    original_content_url=image_url,
                    preview_image_url=image_url
                ))
            else:
                app.logger.warning("CLOUD_MEME_BASE_URL 未設定，無法產生圖片的公開 URL，將只回覆文字。")
        else:
            app.logger.warning(f"無法取得梗圖 {reply_data['meme_filename']} 的資料夾資訊，無法產生圖片 URL。")

    # 使用 Messaging API 回覆訊息
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        try:
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=reply_token,
                    messages=messages_to_reply
                )
            )
            app.logger.info(f"成功回覆訊息給 {user_id}")
        except Exception as e:
            app.logger.error(f"回覆訊息時發生錯誤: {e}")
            # 如果回覆失敗，不要重試，因為回覆令牌可能已經過期
            # 記錄錯誤並繼續處理下一個請求
            return

# 提供靜態檔案服務 (用於讓 LINE 可以存取梗圖圖片)
# 假設你的梗圖放在專案根目錄下的 'memes_hosted_for_line' 資料夾
# 並且該資料夾結構與 MEME_ROOT_DIR 內部一致
# 例如： memes_hosted_for_line/SpongeBob/640.jpg
# 注意：這裡的 'static' 是 URL 的一部分，而 'directory' 是實際的檔案系統路徑
# 我們讓 Flask 從 'MEME_ROOT_DIR' (例如 ./memes) 提供服務，URL 路徑是 /static/memes/
# @app.route('/static/memes/<path:folder_and_filename>')
# def serve_meme_image(folder_and_filename):
    # folder_and_filename 會是像 "SpongeBob/640.jpg" 這樣的形式
    # meme_logic.MEME_ROOT_DIR 是梗圖的實際根目錄，例如 "/path/to/your_project/memes"
    # send_from_directory 需要 (目錄, 檔案名稱)
    # 我們需要將 folder_and_filename 分割成目錄部分和檔案名稱部分
    # 但由於 folder_and_filename 本身就代表了相對於 MEME_ROOT_DIR 的路徑，
    # 所以可以直接將 MEME_ROOT_DIR 作為 directory，folder_and_filename 作為 path
    # app.logger.info(f"嘗試提供靜態檔案：從 {meme_logic.MEME_ROOT_DIR} 提供 {folder_and_filename}")
    # try:
        # return send_from_directory(meme_logic.MEME_ROOT_DIR, folder_and_filename)
    # except FileNotFoundError:
        # app.logger.error(f"請求的靜態檔案未找到: {folder_and_filename} 在 {meme_logic.MEME_ROOT_DIR} 中")
        # abort(404)


if __name__ == "__main__":
    # 從環境變數取得埠號，預設為 5000 (Render.com 等平台會自動設定 PORT 環境變數)
    port = int(os.environ.get("PORT", 5001)) # 改用 5001 避免與其他常用服務衝突
    # 啟動 Flask 應用程式
    # host='0.0.0.0' 讓它可以從外部網路存取 (在容器或 VM 中很重要)
    app.run(host="0.0.0.0", port=port, debug=True) # debug=True 在開發時使用，正式部署時應設為 False
