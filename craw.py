from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.support.ui import Select
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Headers configuration
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
    'cookie': 'CLIENT%5FID=20250605145540875%5F162%2E120%2E184%2E42; IS_TOUCH_DEVICE=F; SCREEN_SIZE=WIDTH=1920&HEIGHT=1080; _ga=GA1.1.840122806.1749106555; _cc_id=3224308362bae4a31824fce979f1b90f; panoramaId_expiry=1749711358245; panoramaId=430135968e9de6287a8b15d792c116d5393864e8d6ec6add45711ea9094e79a6; panoramaIdType=panoIndiv; __gads=ID=b3c9f5b080086ab3:T=1749106558:RT=1749106558:S=ALNI_MZDG-XzpcBCg1pu3vxzuRN_75OIMw; __eoi=ID=a2e8879dbb5ae235:T=1749106558:RT=1749106558:S=AA-AfjayyUqjRnie2fyWCffSNo3I; TW_STOCK_BROWSE_LIST=2330; jiyakeji_uuid=2bd4aec0-41da-11f0-8f43-a1801de691f1; truvid_protected={"val":"c","level":1,"geo":"TW","timestamp":1749106600}; cto_bidid=FLclkl9oQWwlMkZRbXQlMkI5bEI2VDVJNExtUVRNQzBjN3laTFU5M2RaVDF4ZXJhbXloVUxwcHJpeWRFdzFBazhkaFFPbmU1bDVVQ1Nhd0hUM1Z0R0dyYVl3WVpBRTlZQzVsTWFTeCUyRlJYM3RJZmpwdWVhTSUzRA; FCNEC=%5B%5B%22AKsRol8hs26We8zzQR1IjnDCsX1h2OjqGvJoDth1e5e_mrJuLbUl2LkxBj-kS-KD1WoDHlXM1UqmN7zVZAdWv74b8FuqP4ebxx7yTeEJ7w8yCRSmf-mB3SbSwwolFqKM5wHGt4M-MrZ6uSJXzHuKqert39qJLPz9zw%3D%3D%22%5D%5D; _ga_0LP5MLQS7E=GS2.1.s1749106555$o1$g1$t1749106697$j59$l0$h0; cto_bundle=y1aSmF9tZGp6dW5CMDdYYkUxN2F4VDhobU5KNXVVNmE2MVk5Q2FmVDVTQkY2YllhU1l1bGRLZVVleHlDUmhkcDZDZEY3VCUyQmxzVXlvNjVUeUZDNTBtOEVQZklFcTlxQmRueDZjb0hQODVlTyUyRmFzNVIwSVJOTmM5VUE5SWF4ckdHMU1wc1hRdmJKRUhhQ1dPNnpSQ1JNRzZiV09vZFBibTFUUnMlMkZTVlg4bGpQVFJUeE4lMkZmJTJGS2FxNTN4OG9kTDliVFhVVGxGSFlSRTlhYTFjWUpvb3J3Y2h0MVNZUSUzRCUzRA'
}

# Stock list
stock_list = [
    ("2303", "聯電"),
    ("2002", "中鋼"),
    ("1303", "南亞"),
    ("2357", "華碩"),
    ("1301", "台塑"),
    ("2311", "日月光"),
    ("2371", "大同"),
    ("1326", "台化"),
    ("1216", "統一"),
    ("2317", "鴻海"),
    ("2342", "茂矽"),
    ("2356", "英業達"),
    ("2344", "華邦電"),
    ("2201", "裕隆"),
    ("2337", "旺宏"),
    ("1101", "台泥"),
    ("1602", "太電"),
    ("2603", "長榮"),
    ("2610", "華航"),
    ("1102", "亞泥"),
    ("2308", "台達電"),
    ("2313", "華通"),
    ("2501", "國建"),
    ("2204", "中華"),
    ("8712", "國產車"),
    ("2325", "矽品"),
    ("2327", "國巨"),
    ("1504", "東元"),
    ("2505", "國揚"),
    ("2206", "三陽"),
    ("1802", "台玻"),
    ("1314", "中石化"),
    ("2912", "統一超"),
    ("2323", "中環"),
    ("2609", "陽明"),
    ("1462", "東雲"),
    ("1201", "味全"),
    ("1434", "福懋"),
    ("2352", "佳世達"),
    ("2339", "合泰"),
    ("2515", "中工"),
    ("2105", "正新"),
    ("2506", "太設"),
    ("9917", "中保"),
    ("1440", "南紡"),
    ("1903", "士紙"),
    ("2504", "國產"),
    ("2326", "亞瑟"),
    ("5605", "遠航"),
    ("2348", "力廣"),
    ("2347", "聯強"),
    ("2349", "錸德"),
    ("2322", "致福"),
    ("2518", "長億"),
    ("2373", "震旦行"),
    ("2362", "藍天"),
    ("2329", "華泰"),
    ("9907", "統一實"),
    ("2338", "光罩"),
    ("2350", "環電"),
    ("2101", "南港"),
    ("2526", "大陸"),
    ("2315", "神達"),
    ("1503", "士電"),
    ("2301", "光寶科"),
    ("2608", "大榮"),
    ("9801", "力霸"),
    ("1409", "新纖"),
    ("1907", "永豐餘"),
    ("2534", "宏盛"),
    ("1304", "台聚"),
    ("2401", "凌陽"),
    ("1604", "聲寶"),
    ("2334", "國豐"),
    ("1717", "長興"),
    ("2411", "飛瑞"),
    ("2533", "昱成"),
    ("2905", "三商行"),
    ("2310", "旭麗"),
    ("2705", "六福"),
    ("2364", "倫飛"),
    ("2520", "冠德"),
    ("2006", "東鋼"),
    ("2328", "廣宇"),
    ("1718", "中纖"),
    ("2103", "台橡"),
    ("8382", "美式"),
    ("8295", "中強"),
    ("1419", "新紡"),
    ("1207", "嘉食化"),
    ("2604", "立榮"),
    ("1710", "東聯"),
    ("1103", "嘉泥"),
    ("2523", "德寶"),
    ("2525", "寶祥"),
    ("1701", "中化"),
    ("3258", "誠洲"),
    ("1109", "信大"),
    ("2028", "威致"),
    ("1904", "正隆"),
    ("1810", "和成"),
    ("9933", "中鼎"),
    ("1520", "D 復盛"),
    ("1229", "聯華"),
    ("2107", "厚生"),
    ("2606", "裕民"),
    ("2340", "光磊"),
    ("9925", "新保"),
    ("2316", "楠梓電"),
    ("1227", "佳格"),
    ("2023", "燁輝"),
    ("2015", "豐興"),
    ("1313", "聯成"),
    ("9910", "豐泰"),
    ("1460", "宏遠"),
    ("2336", "D 致伸"),
    ("1513", "中興電"),
    ("2332", "友訊"),
    ("2363", "矽統"),
    ("8716", "尖美"),
    ("9902", "台火"),
    ("4506", "崇友"),
    ("1608", "華榮"),
    ("1444", "力麗"),
    ("2019", "桂宏"),
    ("1107", "建台"),
    ("1436", "福益"),
    ("2614", "東森"),
    ("2029", "盛餘"),
    ("8725", "三采"),
    ("2512", "寶建"),
    ("1442", "名軒"),
    ("1902", "台紙"),
    ("1711", "永光"),
    ("2361", "鴻友"),
    ("1408", "中紡"),
    ("2536", "宏普"),
    ("2106", "建大"),
    ("9908", "大台北"),
    ("1458", "嘉畜"),
    ("5901", "中友"),
    ("1603", "華電"),
    ("2538", "基泰"),
    ("2514", "龍邦"),
    ("1510", "台安"),
    ("1704", "榮化"),
    ("2359", "所羅門"),
    ("1606", "歌林"),
    ("2345", "智邦"),
    ("1716", "永信"),
    ("2333", "碧悠"),
    ("1308", "亞聚"),
    ("2910", "統領"),
    ("1905", "華紙"),
    ("1431", "新燕"),
    ("2379", "瑞昱"),
    ("1311", "福聚"),
    ("1110", "東泥"),
    ("2706", "第一店"),
    ("1414", "東和"),
    ("5017", "新泰伸"),
    ("2014", "中鴻"),
    ("2906", "高林"),
    ("1447", "力鵬"),
    ("5307", "耀文"),
    ("1609", "大亞"),
    ("1449", "佳和"),
    ("2318", "佳錄"),
    ("2358", "美格"),
    ("2355", "敬鵬"),
    ("1305", "華夏"),
    ("1517", "利奇"),
    ("9922", "優美"),
    ("5602", "榮櫃"),
    ("1209", "益華"),
    ("8718", "工礦"),
    ("1108", "幸福"),
    ("2601", "益航"),
    ("2314", "台揚"),
    ("2343", "精業"),
    ("5304", "鼎創達"),
    ("2010", "春源"),
    ("4424", "民興"),
    ("1437", "勤益"),
    ("2353", "宏碁"),
    ("1459", "聯發"),
    ("9921", "巨大"),
    ("2537", "聯上發"),
    ("2517", "長谷"),
    ("1612", "宏泰"),
    ("1104", "環泥"),
    ("2438", "英誌"),
    ("1614", "三洋"),
    ("2901", "欣欣"),
    ("2902", "中信"),
    ("5604", "中連"),
    ("9905", "大華"),
    ("2521", "宏總"),
    ("2104", "中橡")
]

def close_ad(driver, timeout=5):
    """關閉廣告彈窗"""
    try:
        close_btn = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.ID, "ats-interstitial-button"))
        )
        close_btn.click()
        print("廣告已關閉")
        time.sleep(1)
    except TimeoutException:
        print("沒有廣告彈窗或已經關閉")
    except Exception as e:
        print(f"關閉廣告時發生錯誤: {e}")

def safe_find_element(driver, by, value, timeout=10):
    """安全尋找元素"""
    try:
        return WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )
    except TimeoutException:
        print(f"找不到元素: {by}={value}")
        return None

def safe_click_element(driver, by, value, timeout=10):
    """安全點擊元素"""
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((by, value))
        )
        element.click()
        return True
    except TimeoutException:
        print(f"無法點擊元素: {by}={value}")
        return False

def setup_driver():
    """設置瀏覽器"""
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-web-security")
    options.add_argument("--allow-running-insecure-content")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        return driver
    except Exception as e:
        print(f"設置瀏覽器失敗: {e}")
        return None

def row_catch(driver, number, count=None):
    """抓取特定行的資料"""
    try:
        row = driver.find_element(By.ID, "row" + str(number))
        tds = row.find_elements(By.XPATH, './td')
        if count is not None:
            tds = tds[:count]
        texts = [td.text for td in tds]
        return texts
    except Exception as e:
        print(f"Error at row {number}: {e}")
        return []

def scrape_single_stock(driver, stock_id, stock_name):
    """爬取單支股票的完整資料"""
    print(f"開始爬取 {stock_id} {stock_name}")
    
    try:
        # 搜尋股票
        driver.get("https://goodinfo.tw/tw/index.asp")
        close_ad(driver)
        
        stock_input = safe_find_element(driver, By.ID, "txtStockCode")
        if not stock_input:
            return None
            
        stock_input.clear()
        close_ad(driver)
        stock_input.send_keys(stock_id)
        close_ad(driver)
        
        if not safe_click_element(driver, By.ID, "btnStockSearch"):
            return None
        
        close_ad(driver)
        WebDriverWait(driver, 15).until(EC.title_contains(stock_id))
        close_ad(driver)
        
        # 點擊經營績效
        if not safe_click_element(driver, By.XPATH, "//a[contains(@class, 'link_blue') and text()='經營績效']"):
            return None
        
        close_ad(driver)
        
        # 抓取基本績效資料
        df1 = scrape_performance_data(driver)
        if df1 is None:
            return None
        
        # 抓取年增統計
        df2 = scrape_growth_data(driver)
        if df2 is None:
            return None
        
        # 抓取PER/PBR
        df3 = scrape_per_pbr_data(driver)
        if df3 is None:
            return None
        
        # 合併績效資料
        df1['年度'] = df1['年度'].astype(str)
        df2['年度'] = df2['年度'].astype(str)
        df3['年度'] = df3['年度'].astype(str)
        df_performance = pd.merge(df1, df2, on="年度")
        df_performance = pd.merge(df_performance, df3, on="年度")
        
        # 抓取財務比率
        df_financial = scrape_financial_ratios(driver)
        if df_financial is None:
            return None
        
        # 最終合併
        df_financial['年度'] = df_financial['年度'].astype(str)
        df_final = pd.merge(df_performance, df_financial, on="年度")
        
        # 加入股票資訊
        df_final.insert(0, "證券代號", stock_id)
        df_final.insert(1, "公司名稱", stock_name)
        
        # 資料處理
        df_final = process_stock_data(df_final)
        
        print(f"完成爬取 {stock_id} {stock_name}")
        return df_final
        
    except Exception as e:
        print(f"爬取 {stock_id} {stock_name} 時發生錯誤: {e}")
        return None

def scrape_performance_data(driver):
    """抓取經營績效資料"""
    try:
        table = safe_find_element(driver, By.XPATH, "//table[contains(@id,'tblDetail')]")
        if not table:
            return None
        
        rows = table.find_elements(By.TAG_NAME, "tr")
        data = []
        
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) == 0:
                continue
            
            year_text = cells[0].text.strip()
            try:
                year = int(year_text)
            except ValueError:
                continue

            if 2009 <= year <= 2024:
                selected_fields = [
                    cells[0].text.strip(),  # 年度
                    cells[1].text.strip(),  # 股本(億)
                    cells[3].text.strip(),  # 收盤價(元)
                    cells[4].text.strip(),  # 平均股價(元)
                    cells[7].text.strip(),  # 營業收入(億)
                    cells[11].text.strip(), # 稅後淨利(億)
                    cells[13].text.strip(), # 營業利益率(%)
                    cells[16].text.strip(), # ROE(%)
                    cells[17].text.strip()  # ROA(%)
                ]
                data.append(selected_fields)
        
        columns = [
            "年度", "股本(億)", "收盤價(元)", "平均股價(元)", "營業收入(億)",
            "稅後淨利(億)", "營業利益率(%)", "ROE(%)", "ROA(%)"
        ]
        
        return pd.DataFrame(data, columns=columns)
        
    except Exception as e:
        print(f"抓取績效資料失敗: {e}")
        return None

def scrape_growth_data(driver):
    """抓取年增統計資料"""
    try:
        sheet_select = Select(driver.find_element(By.ID, "selSheet"))
        sheet_select.select_by_visible_text("年增統計")
        close_ad(driver)
        
        table = driver.find_element(By.XPATH, "//table[contains(@id,'tblDetail')]")
        rows = table.find_elements(By.TAG_NAME, "tr")
        data = []
        
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) == 0:
                continue
            
            year_text = cells[0].text.strip()
            try:
                year = int(year_text)
            except ValueError:
                continue

            if 2009 <= year <= 2024:
                selected_fields = [
                    year_text,
                    cells[11].text.strip(), # M營業利益成長率
                    cells[16].text.strip()  # M稅後淨利成長率
                ]
                data.append(selected_fields)
        
        return pd.DataFrame(data, columns=["年度", "M營業利益成長率", "M稅後淨利成長率"])
        
    except Exception as e:
        print(f"抓取年增統計失敗: {e}")
        return None

def scrape_per_pbr_data(driver):
    """抓取PER/PBR資料"""
    try:
        sheet_select = Select(driver.find_element(By.ID, "selSheet"))
        sheet_select.select_by_visible_text("PER/PBR")
        close_ad(driver)
        
        table = driver.find_element(By.XPATH, "//table[contains(@id,'tblDetail')]")
        rows = table.find_elements(By.TAG_NAME, "tr")
        data = []
        
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) == 0:
                continue
            
            year_text = cells[0].text.strip()
            try:
                year = int(year_text)
            except ValueError:
                continue

            if 2009 <= year <= 2024:
                selected_fields = [
                    year_text,
                    cells[12].text.strip(), # PER
                    cells[16].text.strip()  # PBR
                ]
                data.append(selected_fields)
        
        return pd.DataFrame(data, columns=["年度", "PER", "PBR"])
        
    except Exception as e:
        print(f"抓取PER/PBR失敗: {e}")
        return None

def scrape_financial_ratios(driver):
    """抓取財務比率資料"""
    try:
        # 點擊財務比率表
        if not safe_click_element(driver, By.XPATH, "//a[contains(@class, 'link_blue') and text()='財務比率表']"):
            return None
        
        # 選擇合併報表年度
        sheet_select = Select(driver.find_element(By.ID, "RPT_CAT"))
        sheet_select.select_by_visible_text("合併報表 – 年度")
        
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//table[contains(@id,'tblFinDetail')]"))
        )
        close_ad(driver)
        
        # 抓取2013-2024年資料
        table = driver.find_element(By.XPATH, "//table[contains(@id,'tblFinDetail')]")
        ths = table.find_elements(By.XPATH, './/tbody/tr[1]/th')
        years = [th.text for th in ths[1:]]
        
        metrics1 = {
            "負債/淨值比": 72,
            "流動比率": 51,
            "速動比率": 50,
            "應收帳款周轉次": 56,
            "存貨周轉率": 60
        }
        
        data3 = {}
        for metric_name, row_num in metrics1.items():
            data3[metric_name] = row_catch(driver, row_num, 12)
        
        df3 = pd.DataFrame(data3, index=years)
        df3.insert(0, '年度', df3.index)
        df3.reset_index(drop=True, inplace=True)
        
        # 抓取2009-2012年資料
        sheet_select = Select(driver.find_element(By.ID, "QRY_TIME"))
        sheet_select.select_by_visible_text("2012 年")
        close_ad(driver)
        
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//table[contains(@id,'tblFinDetail')]"))
        )
        
        table2 = driver.find_element(By.XPATH, "//table[contains(@id,'tblFinDetail')]")
        ths = table2.find_elements(By.XPATH, './/tbody/tr[1]/th')
        years = [th.text for th in ths[1:5]]
        
        metrics2 = {
            "負債/淨值比": 73,
            "流動比率": 52,
            "速動比率": 51,
            "應收帳款周轉次": 57,
            "存貨周轉率": 61
        }
        
        data4 = {}
        for metric_name, row_num in metrics2.items():
            data4[metric_name] = row_catch(driver, row_num, 4)
        
        df4 = pd.DataFrame(data4, index=years)
        df4.insert(0, '年度', df4.index)
        df4.reset_index(drop=True, inplace=True)
        
        df_all = pd.concat([df3, df4], axis=0).reset_index(drop=True)
        df_all["年度"] = df_all["年度"].astype(int)
        
        return df_all
        
    except Exception as e:
        print(f"抓取財務比率失敗: {e}")
        return None

def process_stock_data(df):
    """處理股票資料"""
    try:
        # 轉換數值欄位
        numeric_columns = ["股本(億)", "平均股價(元)", "稅後淨利(億)", "營業收入(億)", "收盤價(元)"]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '').astype(float)
        
        df["年度"] = df["年度"].astype(int)
        
        # 計算市值
        df["市值(百萬元)"] = df["股本(億)"] * df["平均股價(元)"] * 100
        
        # 計算稅後淨利率
        df["稅後淨利率(%)"] = (df["稅後淨利(億)"] / df["營業收入(億)"]) * 100
        
        # 重新排列欄位
        cols = list(df.columns)
        
        if "市值(百萬元)" in cols:
            cols.remove("市值(百萬元)")
            cols.insert(2, "市值(百萬元)")
        
        if "稅後淨利率(%)" in cols:
            cols.remove("稅後淨利率(%)")
            cols.insert(11, "稅後淨利率(%)")
        
        df = df[cols]
        
        return df
        
    except Exception as e:
        print(f"處理資料時發生錯誤: {e}")
        return df

def calculate_returns(df_all):
    """計算報酬率"""
    try:
        # 排序資料
        df_all = df_all.sort_values(by=['證券代號', '年度']).reset_index(drop=True)
        
        # 新增 Return 欄位
        df_all['Return'] = pd.NA
        
        # 計算報酬率
        for stock_id in df_all['證券代號'].unique():
            stock_df = df_all[df_all['證券代號'] == stock_id].reset_index()
            for i in range(1, len(stock_df)):
                P1 = stock_df.loc[i-1, '收盤價(元)']
                P2 = stock_df.loc[i, '收盤價(元)']
                ret = (P2 - P1) / P1 * 100
                df_all.loc[stock_df.loc[i, 'index'], 'Return'] = ret
        
        df_all['Return'] = pd.to_numeric(df_all['Return'], errors='coerce')
        
        # 計算年度平均報酬率
        mean_return_by_year = df_all.groupby('年度')['Return'].mean()
        
        # 標記是否高於平均
        df_all['ReturnMean_year_Label'] = pd.NA
        
        for i in df_all.index:
            year = df_all.loc[i, '年度']
            stock_return = df_all.loc[i, 'Return']
            if year in mean_return_by_year.index:
                mean_return = mean_return_by_year.loc[year]
                if pd.notna(stock_return):
                    if stock_return > mean_return:
                        df_all.loc[i, 'ReturnMean_year_Label'] = 1
                    else:
                        df_all.loc[i, 'ReturnMean_year_Label'] = -1
        
        return df_all.sort_values(by=['證券代號', '年度'], ascending=[True, False]).reset_index(drop=True)
        
    except Exception as e:
        print(f"計算報酬率時發生錯誤: {e}")
        return df_all

def main():
    """主程式"""
    driver = setup_driver()
    if not driver:
        print("無法設置瀏覽器，程式結束")
        return
    
    all_data = []
    failed_stocks = []
    
    try:
        for i, (stock_id, stock_name) in enumerate(stock_list):
            print(f"進度: {i+1}/{len(stock_list)} - 處理 {stock_id} {stock_name}")
            
            try:
                df_stock = scrape_single_stock(driver, stock_id, stock_name)
                if df_stock is not None and not df_stock.empty:
                    all_data.append(df_stock)
                    print(f"成功: {stock_id} {stock_name}")
                else:
                    failed_stocks.append((stock_id, stock_name))
                    print(f"失敗: {stock_id} {stock_name}")
                
                # 每10支股票休息一下
                if (i + 1) % 10 == 0:
                    print(f"已完成 {i+1} 支股票，休息 5 秒...")
                    time.sleep(5)
                    
            except Exception as e:
                print(f"處理 {stock_id} {stock_name} 時發生錯誤: {e}")
                failed_stocks.append((stock_id, stock_name))
                continue
        
        if all_data:
            # 合併所有資料
            df_final = pd.concat(all_data, ignore_index=True)
            
            # 計算報酬率
            df_final = calculate_returns(df_final)
            
            # 儲存結果
            df_final.to_excel('all_stocks_output.xlsx', index=False)
            print(f"成功爬取 {len(all_data)} 支股票，資料已儲存至 all_stocks_output.xlsx")
            
            if failed_stocks:
                print(f"失敗的股票 ({len(failed_stocks)} 支):")
                for stock_id, stock_name in failed_stocks:
                    print(f"  {stock_id} {stock_name}")
        else:
            print("沒有成功爬取任何股票資料")
            
    except Exception as e:
        print(f"主程式執行時發生錯誤: {e}")
    finally:
        driver.quit()
        print("瀏覽器已關閉")

if __name__ == "__main__":
    main()