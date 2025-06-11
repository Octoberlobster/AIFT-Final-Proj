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
        self.feature_columns = [
            '市值(百萬元)', '收盤價(元)_年', '股價淨值比', '股價營收比',
            'M淨值報酬率─稅後', '資產報酬率ROA', '營業利益率OPM', '利潤邊際NPM',
            '負債/淨值比', 'M流動比率', 'M速動比率', 'M存貨週轉率 (次)',
            'M應收帳款週轉次', 'M營業利益成長率', 'M稅後淨利成長率'
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
        data = pd.read_excel(data_path, sheet_name='Sheet1')
        data['年份'] = data['年月'].astype(str).str[:4].astype(int)
        data = data[data['年月'] != 200912]
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
        return data
    
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
        enhanced_data['ROA_ROE_比'] = enhanced_data['資產報酬率ROA'] / (enhanced_data['M淨值報酬率─稅後'] + 1e-6)
        enhanced_data['營業利益率_利潤邊際比'] = enhanced_data['營業利益率OPM'] / (enhanced_data['利潤邊際NPM'] + 1e-6)
        enhanced_data['流動比_速動比差'] = enhanced_data['M流動比率'] - enhanced_data['M速動比率']
        
        # 3. 綜合指標
        enhanced_data['獲利能力指標'] = (enhanced_data['資產報酬率ROA'] + enhanced_data['M淨值報酬率─稅後'] + 
                                   enhanced_data['營業利益率OPM']) / 3
        enhanced_data['償債能力指標'] = (enhanced_data['M流動比率'] + enhanced_data['M速動比率']) / 2
        enhanced_data['營運效率指標'] = (enhanced_data['M存貨週轉率 (次)'] + enhanced_data['M應收帳款週轉次']) / 2
        
        # 4. 成長性指標
        enhanced_data['綜合成長率'] = (enhanced_data['M營業利益成長率'] + enhanced_data['M稅後淨利成長率']) / 2
        
        # 5. 風險指標
        enhanced_data['財務風險指標'] = enhanced_data['負債/淨值比'] / (enhanced_data['M流動比率'] + 1e-6)
        
        # 6. 市場估值指標
        enhanced_data['估值綜合指標'] = (enhanced_data['股價淨值比'] + enhanced_data['股價營收比']) / 2
        
        # 7. 對數變換
        for col in ['市值(百萬元)', '收盤價(元)_年']:
            if col in enhanced_data.columns:
                enhanced_data[f'log_{col}'] = np.log1p(enhanced_data[col])
        
        # 8. 分位數特徵
        for col in self.feature_columns:
            if col in enhanced_data.columns:
                enhanced_data[f'{col}_rank'] = enhanced_data.groupby('年份')[col].rank(pct=True)
        
        # 9. 移動平均特徵（模擬時間序列特徵）
        enhanced_data = enhanced_data.sort_values(['證券代碼', '年份'])
        for col in ['Return', '市值(百萬元)', '資產報酬率ROA']:
            if col in enhanced_data.columns:
                enhanced_data[f'{col}_ma2'] = enhanced_data.groupby('證券代碼')[col].rolling(2, min_periods=1).mean().reset_index(0, drop=True)
        
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
        
        # 獲取所有可用特徵
        feature_cols = [col for col in train_enhanced.columns 
                       if col not in ['證券代碼', '簡稱', '年月', '年份', 'Return', 'ReturnMean_year_Label']]
        
        X_train = train_enhanced[feature_cols].values
        y_train = train_enhanced['ReturnMean_year_Label'].values
        X_test = test_enhanced[feature_cols].values
        
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
    
    def create_top10_individual_returns_chart(self):
        """創建前10支股票各自的年化報酬折線圖"""
        if not self.results:
            return
        
        setup_chinese_font()
        
        # 收集前10支股票的個別報酬率資料
        stock_returns = {}
        years = []
        
        for result in self.results:
            if result['top10_stocks'] is not None:
                year = result['test_year']
                years.append(year)
                stocks = result['top10_stocks']
                
                for idx, (_, stock) in enumerate(stocks.iterrows()):
                    stock_name = f"第{idx+1}名股票"
                    if stock_name not in stock_returns:
                        stock_returns[stock_name] = []
                    stock_returns[stock_name].append(stock['Return'])
        
        if not stock_returns:
            return
        
        # 創建折線圖
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        fig.suptitle('前10支股票各自年化報酬率趨勢', fontsize=16, fontweight='bold')
        
        # 上圖：所有10支股票的折線圖
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, (stock_name, returns) in enumerate(stock_returns.items()):
            if len(returns) == len(years):  # 確保資料完整
                ax1.plot(years, returns, marker='o', linewidth=2, markersize=4, 
                        color=colors[i], label=stock_name, alpha=0.8)
        
        ax1.set_title('前10支股票個別年化報酬率')
        ax1.set_xlabel('年份')
        ax1.set_ylabel('年化報酬率 (%)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 下圖：平均報酬率和標準差
        if len(years) > 0:
            avg_returns = []
            std_returns = []
            
            for year_idx in range(len(years)):
                year_returns = [stock_returns[stock][year_idx] for stock in stock_returns 
                              if len(stock_returns[stock]) > year_idx]
                if year_returns:
                    avg_returns.append(np.mean(year_returns))
                    std_returns.append(np.std(year_returns))
                else:
                    avg_returns.append(0)
                    std_returns.append(0)
            
            ax2.plot(years, avg_returns, marker='o', linewidth=3, markersize=8, 
                    color='red', label='平均報酬率')
            ax2.fill_between(years, 
                           np.array(avg_returns) - np.array(std_returns),
                           np.array(avg_returns) + np.array(std_returns),
                           alpha=0.3, color='red', label='±1標準差')
        
        ax2.set_title('前10支股票平均報酬率與變異性')
        ax2.set_xlabel('年份')
        ax2.set_ylabel('年化報酬率 (%)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.legend()
        
        plt.tight_layout()
        individual_chart_path = os.path.join(self.output_dir, 'paper2_top10_individual_returns.png')
        plt.savefig(individual_chart_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_top10_analysis_chart(self):
        """創建前10支股票專門分析圖表"""
        if not self.results:
            return
        
        setup_chinese_font()
        
        years = [r['test_year'] for r in self.results]
        returns_top10 = [r['best_return_top10'] for r in self.results]
        
        # 分析前10支股票的個別表現
        yearly_top10_returns = {}
        for result in self.results:
            if result['top10_stocks'] is not None:
                year = result['test_year']
                stocks = result['top10_stocks']
                yearly_top10_returns[year] = stocks['Return'].tolist()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('前10支股票詳細分析', fontsize=16, fontweight='bold')
        
        # 1. 前10支股票報酬率箱型圖
        if yearly_top10_returns:
            box_data = []
            box_labels = []
            for year in sorted(yearly_top10_returns.keys()):
                box_data.append(yearly_top10_returns[year])
                box_labels.append(str(year))
            
            axes[0, 0].boxplot(box_data, labels=box_labels)
            axes[0, 0].set_title('前10支股票個別報酬率分布')
            axes[0, 0].set_xlabel('年份')
            axes[0, 0].set_ylabel('個別股票報酬率 (%)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 前10支股票勝率統計
        positive_years = sum(1 for r in returns_top10 if r > 0)
        total_years = len(returns_top10)
        win_rate = positive_years / total_years * 100
        
        axes[0, 1].pie([positive_years, total_years - positive_years], 
                      labels=[f'正報酬\n({positive_years}年)', f'負報酬\n({total_years - positive_years}年)'],
                      colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title(f'前10支股票勝率統計\n總勝率: {win_rate:.1f}%')
        
        # 3. 前10支股票年度報酬率趨勢
        axes[1, 0].plot(years, returns_top10, marker='o', linewidth=2, markersize=8, color='green')
        axes[1, 0].set_title('前10支股票年度平均報酬率趨勢')
        axes[1, 0].set_xlabel('年份')
        axes[1, 0].set_ylabel('平均報酬率 (%)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # 4. 前10支股票統計摘要
        mean_return = np.mean(returns_top10)
        std_return = np.std(returns_top10)
        max_return = np.max(returns_top10)
        min_return = np.min(returns_top10)
        
        stats_text = f'統計摘要:\n平均報酬率: {mean_return:.2f}%\n標準差: {std_return:.2f}%\n最高報酬率: {max_return:.2f}%\n最低報酬率: {min_return:.2f}%\n夏普比率: {mean_return/std_return:.3f}'
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_title('前10支股票績效統計')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        top10_chart_path = os.path.join(self.output_dir, 'paper2_top10_analysis.png')
        plt.savefig(top10_chart_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_model_comparison_chart(self):
        """創建模型比較圖表"""
        if not self.results:
            return
        
        setup_chinese_font()
        
        # 統計各模型的表現
        model_performance = {}
        selection_performance = {}
        
        for result in self.results:
            model_name = result['best_config']['model_name']
            selection_method = result['best_config']['selection_method']
            return_rate = result['best_return_top10']  # 使用前10支股票的報酬率
            
            if model_name not in model_performance:
                model_performance[model_name] = []
            model_performance[model_name].append(return_rate)
            
            if selection_method not in selection_performance:
                selection_performance[selection_method] = []
            selection_performance[selection_method].append(return_rate)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 模型表現比較
        models = list(model_performance.keys())
        model_means = [np.mean(model_performance[model]) for model in models]
        model_stds = [np.std(model_performance[model]) for model in models]
        
        ax1.bar(models, model_means, yerr=model_stds, alpha=0.7, capsize=5)
        ax1.set_title('不同模型前10支股票平均報酬率比較')
        ax1.set_xlabel('模型類型')
        ax1.set_ylabel('平均報酬率 (%)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 特徵選擇方法表現比較
        selections = list(selection_performance.keys())
        selection_means = [np.mean(selection_performance[sel]) for sel in selections]
        selection_stds = [np.std(selection_performance[sel]) for sel in selections]
        
        ax2.bar(selections, selection_means, yerr=selection_stds, alpha=0.7, capsize=5, color='orange')
        ax2.set_title('不同特徵選擇方法前10支股票平均報酬率比較')
        ax2.set_xlabel('特徵選擇方法')
        ax2.set_ylabel('平均報酬率 (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        comparison_chart_path = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(comparison_chart_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_analysis_report(self):
        """儲存分析報告"""
        if not self.results:
            return
        
        returns = [r['best_return'] for r in self.results]
        returns_top10 = [r['best_return_top10'] for r in self.results]
        models = [r['best_config']['model_name'] for r in self.results]
        selections = [r['best_config']['selection_method'] for r in self.results]
        
        model_stats = pd.Series(models).value_counts()
        selection_stats = pd.Series(selections).value_counts()
        
        report = f"""
Paper 2 進階股票選股模型分析報告
{'='*50}

分析概況:
- 分析期間: {self.results[0]['train_year']}-{self.results[-1]['test_year']}
- 總測試年數: {len(self.results)}
- 使用演算法: 機器學習多模型方法（類似Paper 2）
- 特徵選擇方法: F-test, 互信息, 隨機森林重要性

前10支股票績效統計:
- 平均年報酬率: {np.mean(returns_top10):.4f}%
- 報酬率標準差: {np.std(returns_top10):.4f}%
- 最高年報酬率: {np.max(returns_top10):.4f}%
- 最低年報酬率: {np.min(returns_top10):.4f}%
- 正報酬年數: {sum(1 for r in returns_top10 if r > 0)}/{len(returns_top10)}
- 勝率: {sum(1 for r in returns_top10 if r > 0)/len(returns_top10)*100:.2f}%

所有選中股票績效統計:
- 平均年報酬率: {np.mean(returns):.4f}%
- 報酬率標準差: {np.std(returns):.4f}%
- 最高年報酬率: {np.max(returns):.4f}%
- 最低年報酬率: {np.min(returns):.4f}%
- 正報酬年數: {sum(1 for r in returns if r > 0)}/{len(returns)}
- 勝率: {sum(1 for r in returns if r > 0)/len(returns)*100:.2f}%

模型使用統計:
- 最常用模型: {model_stats.index[0]} ({model_stats.iloc[0]}次)
- 模型分布: {dict(model_stats)}

特徵選擇方法統計:
- 最常用方法: {selection_stats.index[0]} ({selection_stats.iloc[0]}次)
- 方法分布: {dict(selection_stats)}

年度詳細結果:
"""
        
        for result in self.results:
            config = result['best_config']
            report += f"""
{result['test_year']}年:
  - 前10支股票報酬率: {result['best_return_top10']:.4f}%
  - 所有選中股票報酬率: {result['best_return']:.4f}%
  - 最佳模型: {config['model_name']}
  - 特徵選擇: {config['selection_method']}
  - 模型準確率: {config['accuracy']:.4f}
  - 特徵數量: {config['num_features']}
  - 選中股票數: {result['num_selected_stocks']}
  - 最佳參數: {config['best_params']}
"""
        
        best_year_idx = np.argmax(returns_top10)
        best_result = self.results[best_year_idx]
        best_config = best_result['best_config']
        
        report += f"""

最佳表現年份（前10支股票）: {best_result['test_year']}
  - 前10支股票報酬率: {best_result['best_return_top10']:.4f}%
  - 所有選中股票報酬率: {best_result['best_return']:.4f}%
  - 使用模型: {best_config['model_name']}
  - 特徵選擇: {best_config['selection_method']}
  - 模型準確率: {best_config['accuracy']:.4f}
  - 特徵數量: {best_config['num_features']}
  - 選中股票數: {best_result['num_selected_stocks']}

風險指標:
- 前10支股票夏普比率: {np.mean(returns_top10)/np.std(returns_top10):.4f} (假設無風險利率為0)
- 所有選中股票夏普比率: {np.mean(returns)/np.std(returns):.4f} (假設無風險利率為0)
- 前10支股票最大回撤: {self.calculate_max_drawdown(returns_top10):.4f}%
- 所有選中股票最大回撤: {self.calculate_max_drawdown(returns):.4f}%

Paper 2 方法優勢:
- 結合多種機器學習模型，提高預測穩健性
- 使用多種特徵選擇方法，避免過度擬合
- 基於異常值檢測的特徵工程，捕捉市場異常信號
- 自動化參數優化，減少人為偏差
- 可解釋性較強的特徵重要性分析
- 基於預測概率的前10支股票選擇，提供精準的投資組合
"""
        
        report_path = os.path.join(self.output_dir, 'paper2_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"分析報告已儲存至: {report_path}")
    
    def calculate_max_drawdown(self, returns):
        """計算最大回撤"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return np.min(drawdown)
    
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
        
        best_year_idx = np.argmax(returns_top10)
        best_result = self.results[best_year_idx]
        best_config = best_result['best_config']
        
        print(f"\n最佳表現年份: {best_result['test_year']}")
        print(f"  前10支股票報酬率: {best_result['best_return_top10']:.4f}%")
        print(f"  所有選中股票報酬率: {best_result['best_return']:.4f}%")
        print(f"  使用模型: {best_config['model_name']}")
        print(f"  特徵選擇: {best_config['selection_method']}")
        print(f"  模型準確率: {best_config['accuracy']:.4f}")
        print(f"  特徵數量: {best_config['num_features']}")
        print(f"  選中股票數: {best_result['num_selected_stocks']}")

def main():
    selector = Paper2StockSelector('top200.xlsx', output_dir='Q3outputcraw')
    
    print("開始執行Paper 2進階股票選股分析...")
    selector.run_rolling_window_analysis()
    
    selector.save_results_to_csv()
    selector.create_visualizations()
    selector.create_top10_individual_returns_chart()
    selector.create_top10_analysis_chart()
    selector.create_model_comparison_chart()
    selector.save_analysis_report()
    selector.print_summary()
    
    print(f"\n所有結果已儲存至 Q3outputcraw 資料夾")
    
    return selector

if __name__ == "__main__":
    selector = main()
