import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

# 任務資料
# 每個任務包含：任務名稱, 開始月份 (從 1 開始), 持續月數
tasks = [
    ('Controller', 1, 1),
    ('Data Verify', 1, 2),
    ('FPGA Verify', 2, 1),
    ('Gate Level Optimization', 3, 1),
    ('Circuit Level Optimization', 3, 2),
    ('Layout', 3, 2),
    ('Paper', 4, 1),
    ('Demo', 4, 1)
]

# 設定圖表
fig, ax = plt.subplots(figsize=(10, 6))

# 繪製甘特圖的長條
# 我們需要將月份轉換為日期，這裡假設每個月從第一天開始
for i, (task, start_month, duration) in enumerate(tasks):
    # 計算開始日期 (假設從某個基準年開始，這裡用 2023 年)
    start_date = datetime.date(2023, start_month, 1)
    # 計算結束日期
    end_date = datetime.date(2023, start_month + duration, 1) - datetime.timedelta(days=1) # 減一天確保在當月結束

    # 繪製長條，y軸位置根據任務索引，寬度為持續時間
    ax.barh(i, duration, left=mdates.date2num(start_date), height=0.5, label=task)

# 設定 y 軸標籤為任務名稱
ax.set_yticks(range(len(tasks)))
ax.set_yticklabels([task[0] for task in tasks])

# 設定 x 軸為日期格式，並顯示月份
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%B')) # 顯示月份名稱

# 設定 x 軸範圍，從第一個任務開始前一點到最後一個任務結束後一點
start_date_min = datetime.date(2023, min(task[1] for task in tasks), 1)
end_date_max = datetime.date(2023, max(task[1] + task[2] for task in tasks), 1)
ax.set_xlim(mdates.date2num(start_date_min - datetime.timedelta(days=5)),
            mdates.date2num(end_date_max + datetime.timedelta(days=5)))


# 顯示網格線
ax.grid(axis='x', linestyle='--')

# 設定圖表標題
ax.set_title('專案甘特圖')
ax.set_xlabel('月份')

# 反轉 y 軸，讓第一個任務在最上面
ax.invert_yaxis()

# 調整佈局，防止標籤重疊
plt.tight_layout()

# 顯示圖表
plt.show()
