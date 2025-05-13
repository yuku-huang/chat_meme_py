import time
import urllib.parse
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By

def get_meme_links_by_keyword(keyword, limit=10):
    print(f"📥 搜尋關鍵字：{keyword}")
    encoded_keyword = urllib.parse.quote(keyword)
    search_url = f"https://memes.tw/wtf?q={encoded_keyword}"

    options = uc.ChromeOptions()
    options.headless = False  # 記得先打開觀察網頁行為
    driver = uc.Chrome(options=options)

    driver.get(search_url)
    time.sleep(3)

    # 滾動頁面模擬載入
    for i in range(5):
        print(f"🔄 第 {i+1} 次滾動...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

    # ✅ 用 Selenium 找 <a class="wtf-item"> 的元素
    meme_links = []
    a_tags = driver.find_elements(By.CSS_SELECTOR, 'a.wtf-item')
    print(f"🔗 抓到 {len(a_tags)} 筆連結")

    for a in a_tags:
        href = a.get_attribute("href")
        if href and "/meme/" in href:
            meme_links.append(href)
            print(f"✔ 梗圖連結：{href}")
        if len(meme_links) >= limit:
            break

    driver.quit()
    return meme_links

if __name__ == "__main__":
    keyword = input("請輸入搜尋關鍵字：")
    links = get_meme_links_by_keyword(keyword, limit=10)

    print("\n🔥 搜尋結果梗圖連結：")
    for idx, link in enumerate(links, 1):
        print(f"{idx}. {link}")
