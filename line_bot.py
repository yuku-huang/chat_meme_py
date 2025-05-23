# line_bot.py (極簡測試版 - 再次確認這個版本能否正常運作)
from flask import Flask
import logging
import os

app = Flask(__name__)

# 基本的日誌設定 (與你上次提供的版本相同)
if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    if gunicorn_logger.handlers: # 檢查是否有 handlers
        app.logger.handlers = gunicorn_logger.handlers
        app.logger.setLevel(gunicorn_logger.level)
    else: # 如果 gunicorn logger 沒有 handlers (例如本地執行 flask run)
        logging.basicConfig(level=logging.INFO)
        app.logger.setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)
    app.logger.setLevel(logging.INFO)

app.logger.info("極簡版 Flask app 已初始化 (line_bot.py)。")

@app.route("/")
def hello():
    app.logger.info("極簡版根路徑 '/' 被訪問。")
    return "Hello from Minimal Vercel Flask App! Stage 2 Test.", 200

@app.route("/callback", methods=['POST'])
def callback_stub():
    app.logger.info("極簡版 Webhook callback '/callback' 收到 POST 請求。")
    # 在這個極簡版本中，我們不驗證簽名，也不處理 body
    # 只是為了確認 Vercel 能否正確路由並執行這個函式
    return 'OK_CALLBACK_STUB', 200
# # line_bot.py
# import os
# import logging
# from flask import Flask, request, abort
# from linebot.v3 import WebhookHandler
# from linebot.v3.exceptions import InvalidSignatureError
# from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage, ImageMessage
# from linebot.v3.webhooks import MessageEvent, TextMessageContent
# from urllib.parse import quote
# import json

# # --- 初始化 Flask 應用 ---
# # 確保 app 是全域的，Vercel (Gunicorn) 會尋找名為 'app' 的 WSGI callable
# app = Flask(__name__)

# # --- 設定 Logger ---
# # Flask 已經有自己的 logger (app.logger)，我們可以使用它
# # 或者，如果你想用 root logger，可以像 meme_logic.py 那樣設定
# # 這裡我們使用 app.logger，並確保在 Vercel 上也能看到日誌
# if __name__ != '__main__': # 當透過 Gunicorn 執行時 (Vercel 環境)
#     gunicorn_logger = logging.getLogger('gunicorn.error')
#     app.logger.handlers = gunicorn_logger.handlers
#     app.logger.setLevel(gunicorn_logger.level)
# else: # 本地執行時
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# app.logger.info("Flask 應用程式開始初始化 (line_bot.py)")

# # --- 環境變數讀取 ---
# LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
# LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
# APP_BASE_URL = os.environ.get('APP_BASE_URL') # Vercel 通常會自動設定這個為你的部署 URL
# CLOUD_MEME_BASE_URL = os.environ.get('CLOUD_MEME_BASE_URL')

# # 檢查必要的環境變數
# missing_vars = []
# if not LINE_CHANNEL_SECRET: missing_vars.append("LINE_CHANNEL_SECRET")
# if not LINE_CHANNEL_ACCESS_TOKEN: missing_vars.append("LINE_CHANNEL_ACCESS_TOKEN")
# # APP_BASE_URL 和 CLOUD_MEME_BASE_URL 在圖片回覆時才重要，但最好也檢查
# if not APP_BASE_URL: app.logger.warning("警告：APP_BASE_URL 環境變數未設定。")
# if not CLOUD_MEME_BASE_URL: app.logger.warning("警告：CLOUD_MEME_BASE_URL 環境變數未設定。圖片可能無法正確顯示。")

# if missing_vars:
#     error_message = f"重大錯誤：必要的環境變數未設定：{', '.join(missing_vars)}。應用程式無法啟動。"
#     app.logger.critical(error_message)
#     # 在 Vercel 環境中，直接拋出異常可能會導致部署失敗或服務無法啟動，
#     # 這比靜默失敗更好，因為它能讓你意識到問題。
#     # 但要注意，如果是在 import 時就拋出，可能會導致 issubclass 錯誤。
#     # 更好的方式是在應用程式的某個早期檢查點進行。
#     # 目前，我們先記錄錯誤。如果 Vercel 仍然報 issubclass 錯誤，可能需要調整這裡的處理。
#     # raise RuntimeError(error_message) # 暫時不拋出，以觀察 issubclass 錯誤是否解決


# # --- 設定 LINE Bot ---
# try:
#     if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
#         handler = WebhookHandler(LINE_CHANNEL_SECRET)
#         configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
#         app.logger.info("LINE Bot WebhookHandler 和 Configuration 初始化成功。")
#     else:
#         handler = None
#         configuration = None
#         app.logger.error("LINE Bot 初始化失敗，因為缺少 Channel Secret 或 Access Token。")
# except Exception as e:
#     handler = None
#     configuration = None
#     app.logger.critical(f"LINE Bot 初始化過程中發生嚴重錯誤: {e}", exc_info=True)


# # --- 引入並初始化梗圖邏輯 ---
# import meme_logic
# resources_loaded = False
# try:
#     app.logger.info("準備呼叫 meme_logic.load_all_resources()...")
#     resources_loaded = meme_logic.load_all_resources() # 呼叫資源載入
#     if resources_loaded:
#         app.logger.info("meme_logic.load_all_resources() 成功完成。")
#     else:
#         app.logger.critical("meme_logic.load_all_resources() 失敗。梗圖邏輯可能無法正常運作。")
# except Exception as e:
#     app.logger.critical(f"呼叫 meme_logic.load_all_resources() 時發生未預期錯誤: {e}", exc_info=True)
#     resources_loaded = False # 確保標記為失敗

# # --- Flask 路由 ---
# @app.route("/")
# def home():
#     app.logger.info("根路徑 '/' 被訪問。")
#     if not handler or not configuration:
#         return "LINE Bot 配置錯誤，請檢查環境變數。", 500
#     if not resources_loaded:
#         return "梗圖服務資源載入失敗，請檢查日誌。", 500
#     return "LINE Bot (meme_logic externalized) is running!", 200

# @app.route("/callback", methods=['POST'])
# def callback():
#     if not handler:
#         app.logger.error("Webhook callback 被呼叫，但 LINE Handler 未初始化。")
#         abort(500) # 內部伺服器錯誤

#     app.logger.info("Webhook callback 收到請求。")
#     signature = request.headers.get('X-Line-Signature')
#     body = request.get_data(as_text=True)
#     app.logger.debug(f"請求主體: {body[:200]}...") # 只記錄部分 body

#     try:
#         # 檢查是否為重複訊息 (如果 LINE SDK v3 有此機制)
#         # 注意：LINE SDK v3 的 MessageEvent 物件本身沒有 isRedelivery 屬性。
#         # 重複訊息的處理通常在更底層或由 LINE Platform 控制。
#         # 如果你需要明確處理，可能需要檢查請求頭中的 'X-Line-Retry-Key' 或其他標識。
#         # 為了簡化，這裡暫不加入明確的 isRedelivery 檢查。
#         # data = json.loads(body)
#         # if 'events' in data and len(data['events']) > 0 and data['events'][0].get('deliveryContext', {}).get('isRedelivery', False):
#         #     app.logger.info("收到重複訊息 (isRedelivery=true)，忽略處理。")
#         #     return 'OK'
#         pass
#     except json.JSONDecodeError:
#         app.logger.warning("無法解析請求主體為 JSON (用於 isRedelivery 檢查)。繼續處理...")
#         # 即使無法解析，也應該嘗試讓 handler 處理，因為 handler 可能能處理非 JSON 格式或有自己的解析
#         pass


#     try:
#         handler.handle(body, signature)
#     except InvalidSignatureError:
#         app.logger.error("簽名驗證失敗。請檢查你的 Channel Secret 是否正確。")
#         abort(400)
#     except Exception as e:
#         app.logger.error(f"處理 Webhook 事件時發生錯誤: {e}", exc_info=True)
#         abort(500)
#     return 'OK'

# @handler.add(MessageEvent, message=TextMessageContent)
# def handle_text_message(event: MessageEvent):
#     user_id = event.source.user_id if event.source else "未知使用者"
#     text = event.message.text
#     reply_token = event.reply_token
#     app.logger.info(f"收到來自 {user_id} 的文字訊息: {text}")

#     if not resources_loaded:
#         app.logger.error(f"由於資源載入失敗，無法為 {user_id} 處理訊息 '{text}'。")
#         # 可以選擇回覆一個錯誤訊息給使用者
#         # error_reply = TextMessage(text="抱歉，我目前遇到一些技術問題，暫時無法服務。")
#         # line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=[error_reply]))
#         return # 或者直接不回應

#     try:
#         reply_data = meme_logic.get_meme_reply(text)
#         app.logger.info(f"從 meme_logic 獲取的回覆資料: {reply_data}")

#         messages_to_reply = []
#         if reply_data and reply_data.get("text"):
#             messages_to_reply.append(TextMessage(text=reply_data["text"]))
#         else:
#             app.logger.warning(f"meme_logic 未回傳有效的文字回覆給使用者輸入 '{text}'。使用預設回覆。")
#             messages_to_reply.append(TextMessage(text="嗯...我好像要想一下。"))

#         if reply_data and reply_data.get("meme_filename") and reply_data.get("meme_folder"):
#             if CLOUD_MEME_BASE_URL:
#                 # 確保 CLOUD_MEME_BASE_URL 結尾有斜線，或在這裡處理
#                 base_url = CLOUD_MEME_BASE_URL.rstrip('/')
#                 image_url = f"{base_url}/{quote(reply_data['meme_folder'])}/{quote(reply_data['meme_filename'])}"
#                 app.logger.info(f"準備發送圖片，雲端 URL: {image_url}")
#                 messages_to_reply.append(ImageMessage(
#                     original_content_url=image_url,
#                     preview_image_url=image_url # 通常與 original_content_url 相同
#                 ))
#             else:
#                 app.logger.warning("CLOUD_MEME_BASE_URL 未設定，無法產生圖片的公開 URL，將只回覆文字。")
#         elif reply_data and reply_data.get("meme_filename") and not reply_data.get("meme_folder"):
#              app.logger.warning(f"找到梗圖檔名 '{reply_data.get('meme_filename')}' 但缺少資料夾資訊，無法產生圖片 URL。")


#         if not messages_to_reply: # 再次檢查，以防萬一
#             app.logger.error("沒有任何訊息可以回覆。這不應該發生。")
#             return

#         if configuration: # 確保 configuration 已初始化
#             with ApiClient(configuration) as api_client:
#                 line_bot_api = MessagingApi(api_client)
#                 line_bot_api.reply_message(
#                     ReplyMessageRequest(
#                         reply_token=reply_token,
#                         messages=messages_to_reply
#                     )
#                 )
#             app.logger.info(f"成功回覆訊息給 {user_id}。")
#         else:
#             app.logger.error("LINE MessagingApi Configuration 未初始化，無法回覆訊息。")

#     except Exception as e:
#         app.logger.error(f"處理文字訊息或回覆時發生錯誤: {e}", exc_info=True)
#         # 這裡可以考慮是否要嘗試回覆一個通用的錯誤訊息給使用者
#         # 但要注意 reply_token 可能已經失效

# # 為了讓 Vercel (或 Gunicorn) 能找到 app 物件，它必須在模組的頂層。
# # if __name__ == "__main__": 這部分僅用於本地開發執行。
# # if __name__ == "__main__":
# #     # 確保本地測試時，必要的環境變數已設定
# #     # 例如： export LINE_CHANNEL_SECRET="..."
# #     #       export LINE_CHANNEL_ACCESS_TOKEN="..."
# #     #       export ANNOTATIONS_JSON_URL="..."
# #     #       export GROQ_API_KEY_1="..."
# #     #       export MEME_SEARCH_API_URL="..."
# #     #       export CLOUD_MEME_BASE_URL="..."
    
# #     app.logger.info("應用程式以本地開發模式啟動。")
# #     if not resources_loaded:
# #         app.logger.warning("警告：資源未完全載入，本地測試功能可能受限。")
# #     if not handler or not configuration:
# #         app.logger.warning("警告：LINE Bot 未完全配置，本地測試功能可能受限。")

# #     port = int(os.environ.get("PORT", 5001))
# #     app.run(host="0.0.0.0", port=port, debug=True) # 本地開發時 debug=True
