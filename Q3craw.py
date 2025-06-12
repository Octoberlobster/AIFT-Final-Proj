import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
import os
import platform
warnings.filterwarnings('ignore')

# 設定中文字體
def setup_chinese_font():
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Noto Sans CJK TC']
    plt.rcParams['axes.unicode_minus'] = False

setup_chinese_font()

class Paper2StockSelector:
    def __init__(self, data_path, output_dir='Q3outputcraw'):
        """初始化Paper 2股票選股模型（基於機器學習的進階方法）"""
        self.output_dir = output_dir
        self.create_output_directory()
        
        self.data = self.load_and_preprocess_data(data_path)
        # 根據實際Excel欄位調整特徵欄位名稱
        self.feature_columns = [
            '市值(百萬元)', '收盤價(元)', '營業收入(億)', '稅後淨利(億)',
            '營業利益率(%)', 'ROE(%)', '稅後淨利率(%)', 'ROA(%)',
            'PER', 'PBR', '負債/淨值比', '流動比率', '速動比率',
            '應收帳款周轉次', '存貨周轉率', 'M營業利益成長率', 'M稅後淨利成長率'
        ]
        self.results = []
        
    def create_output_directory(self):
        """創建輸出資料夾"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"已創建輸出資料夾: {self.output_dir}")
        else:
            print(f"輸出資料夾已存在: {self.output_dir}")
        
    def load_and_preprocess_data(self, data_path):
        """載入並預處理資料"""
        # 讀取Excel檔案
        data = pd.read_excel(data_path, sheet_name='Sheet1')
        
        # 創建年份欄位
        data['年份'] = data['年月'].astype(str).str[:4].astype(int)
        
        # 移除特定年月資料（如果需要）
        data = data[data['年月'] != 200912]
        
        # 創建報酬率標籤
        self.create_return_labels(data)
        
        # 處理缺失值
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
        
        return data
    
    def create_return_labels(self, data):
        """創建報酬率標籤"""
        # 計算每年的平均報酬率
        yearly_mean_return = data.groupby('年份')['Return'].mean()
        
        # 為每筆資料標記是否高於該年平均
        data['ReturnMean_year_Label'] = 0
        for year in yearly_mean_return.index:
            year_mask = data['年份'] == year
            mean_return = yearly_mean_return[year]
            data.loc[year_mask & (data['Return'] > mean_return), 'ReturnMean_year_Label'] = 1
    
    def create_advanced_features(self, data):
        """創建進階特徵（類似Paper 2的特徵工程）"""
        enhanced_data = data.copy()
        
        # 1. 異常值指標（Paper 2的核心概念）
        for col in self.feature_columns:
            if col in enhanced_data.columns:
                # 計算Z-score作為異常值指標
                mean_val = enhanced_data.groupby('年份')[col].transform('mean')
                std_val = enhanced_data.groupby('年份')[col].transform('std')
                enhanced_data[f'{col}_zscore'] = (enhanced_data[col] - mean_val) / (std_val + 1e-6)
                enhanced_data[f'{col}_outlier'] = (np.abs(enhanced_data[f'{col}_zscore']) > 2).astype(int)
        
        # 2. 比率特徵
        if 'ROA(%)' in enhanced_data.columns and 'ROE(%)' in enhanced_data.columns:
            enhanced_data['ROA_ROE_比'] = enhanced_data['ROA(%)'] / (enhanced_data['ROE(%)'] + 1e-6)
        
        if '營業利益率(%)' in enhanced_data.columns and '稅後淨利率(%)' in enhanced_data.columns:
            enhanced_data['營業利益率_淨利率比'] = enhanced_data['營業利益率(%)'] / (enhanced_data['稅後淨利率(%)'] + 1e-6)
        
        if '流動比率' in enhanced_data.columns and '速動比率' in enhanced_data.columns:
            enhanced_data['流動比_速動比差'] = enhanced_data['流動比率'] - enhanced_data['速動比率']
        
        # 3. 綜合指標
        if all(col in enhanced_data.columns for col in ['ROA(%)', 'ROE(%)', '營業利益率(%)']):
            enhanced_data['獲利能力指標'] = (enhanced_data['ROA(%)'] + enhanced_data['ROE(%)'] + 
                                       enhanced_data['營業利益率(%)']) / 3
        
        if all(col in enhanced_data.columns for col in ['流動比率', '速動比率']):
            enhanced_data['償債能力指標'] = (enhanced_data['流動比率'] + enhanced_data['速動比率']) / 2
        
        if all(col in enhanced_data.columns for col in ['存貨周轉率', '應收帳款周轉次']):
            enhanced_data['營運效率指標'] = (enhanced_data['存貨周轉率'] + enhanced_data['應收帳款周轉次']) / 2
        
        # 4. 成長性指標
        if all(col in enhanced_data.columns for col in ['M營業利益成長率', 'M稅後淨利成長率']):
            enhanced_data['綜合成長率'] = (enhanced_data['M營業利益成長率'] + enhanced_data['M稅後淨利成長率']) / 2
        
        # 5. 風險指標
        if all(col in enhanced_data.columns for col in ['負債/淨值比', '流動比率']):
            enhanced_data['財務風險指標'] = enhanced_data['負債/淨值比'] / (enhanced_data['流動比率'] + 1e-6)
        
        # 6. 市場估值指標
        if all(col in enhanced_data.columns for col in ['PER', 'PBR']):
            enhanced_data['估值綜合指標'] = (enhanced_data['PER'] + enhanced_data['PBR']) / 2
        
        # 7. 對數變換
        for col in ['市值(百萬元)', '收盤價(元)']:
            if col in enhanced_data.columns:
                enhanced_data[f'log_{col}'] = np.log1p(enhanced_data[col])
        
        # 8. 分位數特徵
        for col in self.feature_columns:
            if col in enhanced_data.columns:
                enhanced_data[f'{col}_rank'] = enhanced_data.groupby('年份')[col].rank(pct=True)
        
        # 9. 移動平均特徵（模擬時間序列特徵）
        enhanced_data = enhanced_data.sort_values(['證券代號', '年份'])
        for col in ['Return', '市值(百萬元)', 'ROA(%)']:
            if col in enhanced_data.columns:
                enhanced_data[f'{col}_ma2'] = enhanced_data.groupby('證券代號')[col].rolling(2, min_periods=1).mean().reset_index(0, drop=True)
        
        return enhanced_data
    
    def get_model_configurations(self):
        """獲取不同模型配置（類似Paper 2的多模型方法）"""
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'NaiveBayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7]
                }
            }
        }
        return models
    
    def feature_selection_methods(self, X_train, y_train, X_test):
        """特徵選擇方法（Paper 2的核心）"""
        feature_selection_results = {}
        
        # 1. 統計方法特徵選擇
        selector_f = SelectKBest(score_func=f_classif, k=min(20, X_train.shape[1]))
        X_train_f = selector_f.fit_transform(X_train, y_train)
        X_test_f = selector_f.transform(X_test)
        feature_selection_results['f_classif'] = (X_train_f, X_test_f, selector_f.get_support())
        
        # 2. 互信息特徵選擇
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(20, X_train.shape[1]))
        X_train_mi = selector_mi.fit_transform(X_train, y_train)
        X_test_mi = selector_mi.transform(X_test)
        feature_selection_results['mutual_info'] = (X_train_mi, X_test_mi, selector_mi.get_support())
        
        # 3. 隨機森林特徵重要性
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_selector.fit(X_train, y_train)
        feature_importance = rf_selector.feature_importances_
        top_features = np.argsort(feature_importance)[-min(20, X_train.shape[1]):]
        feature_mask = np.zeros(X_train.shape[1], dtype=bool)
        feature_mask[top_features] = True
        X_train_rf = X_train[:, feature_mask]
        X_test_rf = X_test[:, feature_mask]
        feature_selection_results['random_forest'] = (X_train_rf, X_test_rf, feature_mask)
        
        return feature_selection_results
    
    def train_and_evaluate(self, train_year, test_year):
        """訓練並評估模型"""
        train_data = self.data[self.data['年份'] == train_year].copy()
        test_data = self.data[self.data['年份'] == test_year].copy()
        
        if len(train_data) == 0 or len(test_data) == 0:
            return None
        
        # 創建進階特徵
        train_enhanced = self.create_advanced_features(train_data)
        test_enhanced = self.create_advanced_features(test_data)
        
        # 獲取所有可用特徵（排除非特徵欄位）
        exclude_cols = ['證券代號', '公司名稱', '年月', '年份', 'Return', 'ReturnMean_year_Label']
        feature_cols = [col for col in train_enhanced.columns if col not in exclude_cols]
        
        # 確保特徵欄位存在且為數值型
        feature_cols = [col for col in feature_cols if col in train_enhanced.columns and 
                       train_enhanced[col].dtype in ['int64', 'float64']]
        
        if len(feature_cols) == 0:
            print(f"警告：{train_year}->{test_year} 沒有可用的特徵欄位")
            return None
        
        X_train = train_enhanced[feature_cols].values
        y_train = train_enhanced['ReturnMean_year_Label'].values
        X_test = test_enhanced[feature_cols].values
        
        # 檢查是否有無窮大或NaN值
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
        
        # 標準化
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        best_return = -np.inf
        best_return_top10 = -np.inf
        best_config = {}
        best_selected_stocks = None
        best_top10_stocks = None
        best_model = None
        
        models = self.get_model_configurations()
        
        # 特徵選擇
        feature_selection_results = self.feature_selection_methods(X_train_scaled, y_train, X_test_scaled)
        
        for selection_method, (X_train_fs, X_test_fs, feature_mask) in feature_selection_results.items():
            for model_name, model_config in models.items():
                try:
                    # 簡化參數搜索
                    param_grid = {k: v[:2] for k, v in model_config['params'].items()}
                    
                    grid_search = GridSearchCV(
                        model_config['model'], 
                        param_grid, 
                        cv=3, 
                        scoring='accuracy',
                        n_jobs=-1
                    )
                    
                    grid_search.fit(X_train_fs, y_train)
                    
                    # 預測測試集
                    y_pred = grid_search.predict(X_test_fs)
                    y_pred_proba = grid_search.predict_proba(X_test_fs)[:, 1]
                    
                    # 根據預測概率排序，選擇前10支股票
                    test_data_with_proba = test_data.copy()
                    test_data_with_proba['prediction_proba'] = y_pred_proba
                    test_data_sorted = test_data_with_proba.sort_values('prediction_proba', ascending=False)
                    
                    # 選擇前10支股票
                    top10_stocks = test_data_sorted.head(10)
                    avg_return_top10 = top10_stocks['Return'].mean()
                    
                    # 原本的選股方法（選擇預測為1的股票）
                    selected_indices = np.where(y_pred == 1)[0]
                    
                    if len(selected_indices) > 0:
                        selected_stocks = test_data.iloc[selected_indices]
                        avg_return = selected_stocks['Return'].mean()
                        
                        # 以前10支股票的報酬率作為主要評估標準
                        if avg_return_top10 > best_return_top10:
                            best_return_top10 = avg_return_top10
                            best_return = avg_return
                            best_config = {
                                'model_name': model_name,
                                'selection_method': selection_method,
                                'best_params': grid_search.best_params_,
                                'num_features': X_train_fs.shape[1],
                                'accuracy': grid_search.best_score_,
                                'feature_mask': feature_mask
                            }
                            best_selected_stocks = selected_stocks
                            best_top10_stocks = top10_stocks
                            best_model = grid_search.best_estimator_
                    else:
                        # 如果沒有預測為1的股票，仍然記錄前10支股票
                        if avg_return_top10 > best_return_top10:
                            best_return_top10 = avg_return_top10
                            best_return = 0
                            best_config = {
                                'model_name': model_name,
                                'selection_method': selection_method,
                                'best_params': grid_search.best_params_,
                                'num_features': X_train_fs.shape[1],
                                'accuracy': grid_search.best_score_,
                                'feature_mask': feature_mask
                            }
                            best_selected_stocks = None
                            best_top10_stocks = top10_stocks
                            best_model = grid_search.best_estimator_
                            
                except Exception as e:
                    print(f"模型 {model_name} 與特徵選擇 {selection_method} 發生錯誤: {str(e)}")
                    continue
        
        return {
            'train_year': train_year,
            'test_year': test_year,
            'best_return': best_return,
            'best_return_top10': best_return_top10,
            'best_config': best_config,
            'num_selected_stocks': len(best_selected_stocks) if best_selected_stocks is not None else 0,
            'selected_stocks': best_selected_stocks,
            'top10_stocks': best_top10_stocks,
            'best_model': best_model
        }
    
    def run_rolling_window_analysis(self):
        """執行滾動視窗分析"""
        years = sorted(self.data['年份'].unique())
        
        print(f"開始Paper 2進階演算法分析")
        print(f"使用模型: RandomForest, GradientBoosting, LogisticRegression, SVM, NaiveBayes")
        print(f"特徵選擇方法: F-test, 互信息, 隨機森林重要性")
        print(f"年份範圍: {years}")
        print(f"可用特徵欄位: {[col for col in self.feature_columns if col in self.data.columns]}")
        
        for i in range(len(years) - 1):
            train_year = years[i]
            test_year = years[i + 1]
            
            print(f"\n分析 {train_year} -> {test_year}")
            
            result = self.train_and_evaluate(train_year, test_year)
            
            if result and result['best_return_top10'] != -np.inf:
                self.results.append(result)
                print(f"前10支股票平均報酬率: {result['best_return_top10']:.4f}%")
                print(f"所有選中股票平均報酬率: {result['best_return']:.4f}%")
                print(f"最佳模型: {result['best_config']['model_name']}")
                print(f"特徵選擇方法: {result['best_config']['selection_method']}")
                print(f"模型準確率: {result['best_config']['accuracy']:.4f}")
                print(f"特徵數量: {result['best_config']['num_features']}")
                print(f"選中股票數量: {result['num_selected_stocks']}")
            else:
                print("未找到有效結果")
    
    def save_results_to_csv(self):
        """儲存結果到CSV檔案"""
        if not self.results:
            print("沒有結果可儲存")
            return
        
        results_data = []
        selected_stocks_data = []
        top10_stocks_data = []
        
        for result in self.results:
            config = result['best_config']
            results_data.append({
                '訓練年份': result['train_year'],
                '測試年份': result['test_year'],
                '前10支股票平均報酬率(%)': result['best_return_top10'],
                '所有選中股票平均報酬率(%)': result['best_return'],
                '最佳模型': config['model_name'],
                '特徵選擇方法': config['selection_method'],
                '模型準確率': config['accuracy'],
                '特徵數量': config['num_features'],
                '選中股票數量': result['num_selected_stocks'],
                '最佳參數': str(config['best_params'])
            })
            
            # 儲存所有選中股票詳細資訊
            if result['selected_stocks'] is not None:
                stocks = result['selected_stocks'].copy()
                stocks['訓練年份'] = result['train_year']
                stocks['測試年份'] = result['test_year']
                stocks['使用模型'] = config['model_name']
                stocks['特徵選擇方法'] = config['selection_method']
                stocks['選股類型'] = '預測為高於平均'
                selected_stocks_data.append(stocks)
            
            # 儲存前10支股票詳細資訊
            if result['top10_stocks'] is not None:
                top10_stocks = result['top10_stocks'].copy()
                top10_stocks['訓練年份'] = result['train_year']
                top10_stocks['測試年份'] = result['test_year']
                top10_stocks['使用模型'] = config['model_name']
                top10_stocks['特徵選擇方法'] = config['selection_method']
                top10_stocks['排名'] = range(1, len(top10_stocks) + 1)
                top10_stocks_data.append(top10_stocks)
        
        # 儲存主要結果
        results_df = pd.DataFrame(results_data)
        results_path = os.path.join(self.output_dir, 'paper2_stock_selection_results.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        
        # 儲存所有選中股票詳細資訊
        if selected_stocks_data:
            all_selected_stocks = pd.concat(selected_stocks_data, ignore_index=True)
            stocks_path = os.path.join(self.output_dir, 'paper2_selected_stocks_details.csv')
            all_selected_stocks.to_csv(stocks_path, index=False, encoding='utf-8-sig')
        
        # 儲存前10支股票詳細資訊
        if top10_stocks_data:
            all_top10_stocks = pd.concat(top10_stocks_data, ignore_index=True)
            top10_path = os.path.join(self.output_dir, 'paper2_top10_stocks_details.csv')
            all_top10_stocks.to_csv(top10_path, index=False, encoding='utf-8-sig')
        
        print(f"結果已儲存到 {self.output_dir} 資料夾:")
        print(f"- paper2_stock_selection_results.csv (主要結果)")
        print(f"- paper2_selected_stocks_details.csv (所有選中股票詳細資訊)")
        print(f"- paper2_top10_stocks_details.csv (前10支股票詳細資訊)")
        
        return results_df
    
    def create_visualizations(self):
        """創建視覺化圖表"""
        if not self.results:
            print("沒有結果可視覺化")
            return
        
        setup_chinese_font()
        
        years = [r['test_year'] for r in self.results]
        returns = [r['best_return'] for r in self.results]
        returns_top10 = [r['best_return_top10'] for r in self.results]
        models = [r['best_config']['model_name'] for r in self.results]
        selection_methods = [r['best_config']['selection_method'] for r in self.results]
        num_features = [r['best_config']['num_features'] for r in self.results]
        accuracies = [r['best_config']['accuracy'] for r in self.results]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Paper 2 進階股票選股模型分析結果', fontsize=16, fontweight='bold')
        
        # 1. 年度報酬率趨勢比較（前10 vs 所有選中）
        axes[0, 0].plot(years, returns_top10, marker='o', linewidth=2, markersize=8, color='green', label='前10支股票')
        axes[0, 0].plot(years, returns, marker='s', linewidth=2, markersize=6, color='blue', label='所有選中股票')
        axes[0, 0].set_title('年度最佳平均報酬率趨勢比較')
        axes[0, 0].set_xlabel('年份')
        axes[0, 0].set_ylabel('平均報酬率 (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].legend()
        
        # 2. 最佳模型分布
        model_counts = pd.Series(models).value_counts()
        axes[0, 1].bar(model_counts.index, model_counts.values, alpha=0.7)
        axes[0, 1].set_title('最佳模型分布')
        axes[0, 1].set_xlabel('模型類型')
        axes[0, 1].set_ylabel('次數')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 特徵選擇方法分布
        selection_counts = pd.Series(selection_methods).value_counts()
        axes[0, 2].bar(selection_counts.index, selection_counts.values, alpha=0.7, color='orange')
        axes[0, 2].set_title('特徵選擇方法分布')
        axes[0, 2].set_xlabel('特徵選擇方法')
        axes[0, 2].set_ylabel('次數')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 特徵數量 vs 前10支股票報酬率
        scatter = axes[1, 0].scatter(num_features, returns_top10, c=years, cmap='viridis', s=100, alpha=0.7)
        axes[1, 0].set_title('特徵數量 vs 前10支股票平均報酬率')
        axes[1, 0].set_xlabel('特徵數量')
        axes[1, 0].set_ylabel('平均報酬率 (%)')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='年份')
        
        # 5. 模型準確率 vs 前10支股票報酬率
        axes[1, 1].scatter(accuracies, returns_top10, alpha=0.7, color='red', s=100)
        axes[1, 1].set_title('模型準確率 vs 前10支股票平均報酬率')
        axes[1, 1].set_xlabel('模型準確率')
        axes[1, 1].set_ylabel('平均報酬率 (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 累積報酬率比較
        cumulative_returns = np.cumsum(returns)
        cumulative_returns_top10 = np.cumsum(returns_top10)
        axes[1, 2].plot(years, cumulative_returns_top10, marker='o', linewidth=2, markersize=6, color='green', label='前10支股票')
        axes[1, 2].plot(years, cumulative_returns, marker='s', linewidth=2, markersize=6, color='blue', label='所有選中股票')
        axes[1, 2].fill_between(years, cumulative_returns_top10, alpha=0.3, color='green')
        axes[1, 2].set_title('累積報酬率趨勢比較')
        axes[1, 2].set_xlabel('年份')
        axes[1, 2].set_ylabel('累積報酬率 (%)')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()
        
        plt.tight_layout()
        chart_path = os.path.join(self.output_dir, 'paper2_analysis_results.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary(self):
        """印出分析摘要"""
        if not self.results:
            print("沒有結果可顯示")
            return
        
        returns = [r['best_return'] for r in self.results]
        returns_top10 = [r['best_return_top10'] for r in self.results]
        
        print("\n" + "="*50)
        print("Paper 2 進階股票選股模型分析摘要")
        print("="*50)
        print(f"分析期間: {self.results[0]['train_year']}-{self.results[-1]['test_year']}")
        print(f"總測試年數: {len(self.results)}")
        
        print(f"\n前10支股票表現:")
        print(f"  平均年報酬率: {np.mean(returns_top10):.4f}%")
        print(f"  報酬率標準差: {np.std(returns_top10):.4f}%")
        print(f"  最高年報酬率: {np.max(returns_top10):.4f}%")
        print(f"  最低年報酬率: {np.min(returns_top10):.4f}%")
        print(f"  正報酬年數: {sum(1 for r in returns_top10 if r > 0)}/{len(returns_top10)}")
        
        print(f"\n所有選中股票表現:")
        print(f"  平均年報酬率: {np.mean(returns):.4f}%")
        print(f"  報酬率標準差: {np.std(returns):.4f}%")
        print(f"  最高年報酬率: {np.max(returns):.4f}%")
        print(f"  最低年報酬率: {np.min(returns):.4f}%")
        print(f"  正報酬年數: {sum(1 for r in returns if r > 0)}/{len(returns)}")

def main():
    # 使用調整後的檔案名稱
    selector = Paper2StockSelector('top200craw.xlsx', output_dir='Q3outputcraw')
    
    print("開始執行Paper 2進階股票選股分析...")
    selector.run_rolling_window_analysis()
    
    if selector.results:
        selector.save_results_to_csv()
        selector.create_visualizations()
        selector.print_summary()
        print(f"\n所有結果已儲存至 Q3outputcraw 資料夾")
    else:
        print("沒有產生任何結果，請檢查資料格式和欄位名稱")
    
    return selector

if __name__ == "__main__":
    selector = main()
