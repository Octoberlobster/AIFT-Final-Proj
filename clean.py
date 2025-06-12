import pandas as pd
import numpy as np

def process_top200_data(input_file='top200ori.xlsx', output_file='top200craw.xlsx'):
    """
    處理top200原始資料，將'-'替換為空白
    """
    # 讀取原始資料
    df = pd.read_excel(input_file)
    
    print(f"原始資料形狀: {df.shape}")
    print(f"發現 '-' 的數量: {(df == '-').sum().sum()}")
    
    # 將所有只包含'-'的儲存格替換為NaN（空白）
    df_cleaned = df.replace('-', np.nan)
    
    # 儲存處理後的檔案
    df_cleaned.to_excel(output_file, index=False)
    
    print(f"處理完成！檔案已儲存至: {output_file}")
    print(f"處理後空白儲存格數量: {df_cleaned.isnull().sum().sum()}")
    
    return df_cleaned

# 執行資料處理
cleaned_data = process_top200_data()
