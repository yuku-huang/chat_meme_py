# line_bot.py  -- 修正版
import os
import logging
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, TextMessage, ImageMessage
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from urllib.parse import quote
import json
import meme_logic

# ────────────────────────────────
# Flask app (全域 WSGI callable)
# ────────────────────────────────
app = Flask(__name__)

# ────────────────────────────────
# Logging
# ────────────────────────────────
if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# ────────────────────────────────
# 環境變數
# ────────────────────────────────
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
APP_BASE_URL = os.environ.get('APP_BASE_URL')
CLOUD_MEME_BASE_URL = os.environ.get('CLOUD_MEME_BASE_URL')

missing_vars = [
    v for v in ('LINE_CHANNEL_SECRET', 'LINE_CHANNEL_ACCESS_TOKEN')
    if globals()[v] is None
]
if missing_vars:
    app.logger.critical(f"缺少必要環境變數: {', '.join(missing_vars)}")

# ────────────────────────────────
# LINE SDK 初始化
# ────────────────────────────────
try:
    if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
        line_handler = WebhookHandler(LINE_CHANNEL_SECRET)        # ← 改名
        configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
        app.logger.info("LINE WebhookHandler 初始化成功")
    else:
        line_handler = None
        configuration = None
except Exception as e:
    line_handler = None
    configuration = None
    app.logger.critical(f"LINE SDK 初始化失敗: {e}", exc_info=True)

# ────────────────────────────────
# 梗圖資源載入
# ────────────────────────────────
app.logger.info("載入 meme_logic 資源...")
resources_loaded = meme_logic.load_all_resources()

# ────────────────────────────────
# 路由
# ────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    if not (line_handler and configuration and resources_loaded):
        return "Service unavailable.", 500
    return "LINE Bot is running!", 200


@app.route("/callback", methods=["POST"])
def callback():
    if not line_handler:
        app.logger.error("LINE handler 未初始化")
        abort(500)

    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)

    try:
        line_handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.error("簽名錯誤")
        abort(400)
    except Exception as e:
        app.logger.error(f"處理 webhook 時發生錯誤: {e}", exc_info=True)
        abort(500)
    return "OK"


@line_handler.add(MessageEvent, message=TextMessageContent)       # ← 改裝飾器
def handle_text_message(event: MessageEvent):
    user_id = getattr(event.source, "user_id", "unknown")
    text = event.message.text
    reply_token = event.reply_token

    if not resources_loaded:
        return

    try:
        reply_data = meme_logic.get_meme_reply(text)
        msgs = [TextMessage(text=reply_data.get("text", "嗯...我想一下。"))]

        if reply_data.get("meme_filename") and reply_data.get("meme_folder"):
            if CLOUD_MEME_BASE_URL:
                base = CLOUD_MEME_BASE_URL.rstrip("/")
                url = f"{base}/{quote(reply_data['meme_folder'])}/{quote(reply_data['meme_filename'])}"
                msgs.append(ImageMessage(original_content_url=url, preview_image_url=url))

        with ApiClient(configuration) as api_client:
            MessagingApi(api_client).reply_message(
                ReplyMessageRequest(reply_token=reply_token, messages=msgs)
            )
    except Exception as e:
        app.logger.error(f"回覆訊息失敗: {e}", exc_info=True)
