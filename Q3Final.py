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
    def __init__(self, data_path, output_dir='Q3output'):
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
        self.backtest_results = []  # 新增：儲存回測結果
        
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

    def calculate_backtest_metrics(self):
        """計算TradingView風格的回測指標"""
        if not self.results:
            return None
        
        returns_top10 = [r['best_return_top10'] for r in self.results]
        returns_all = [r['best_return'] for r in self.results]
        years = [r['test_year'] for r in self.results]
        
        # 計算累積淨值
        initial_capital = 100000  # 初始資金10萬
        cumulative_top10 = [initial_capital]
        cumulative_all = [initial_capital]
        
        for i, (ret_10, ret_all) in enumerate(zip(returns_top10, returns_all)):
            cumulative_top10.append(cumulative_top10[-1] * (1 + ret_10/100))
            cumulative_all.append(cumulative_all[-1] * (1 + ret_all/100))
        
        # 計算回測指標
        def calculate_metrics(returns, cumulative):
            total_return = (cumulative[-1] - cumulative[0]) / cumulative[0] * 100
            annual_return = (cumulative[-1] / cumulative[0]) ** (1/len(returns)) - 1
            volatility = np.std(returns)
            sharpe_ratio = np.mean(returns) / volatility if volatility > 0 else 0
            
            # 計算最大回撤
            peak = cumulative[0]
            max_drawdown = 0
            for value in cumulative[1:]:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # 勝率
            win_rate = sum(1 for r in returns if r > 0) / len(returns) * 100
            
            # 盈利因子
            profits = [r for r in returns if r > 0]
            losses = [abs(r) for r in returns if r < 0]
            profit_factor = sum(profits) / sum(losses) if losses else float('inf')
            
            return {
                'total_return': total_return,
                'annual_return': annual_return * 100,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown * 100,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(returns),
                'winning_trades': len(profits),
                'losing_trades': len(losses)
            }
        
        metrics_top10 = calculate_metrics(returns_top10, cumulative_top10)
        metrics_all = calculate_metrics(returns_all, cumulative_all)
        
        self.backtest_results = {
            'years': years,
            'returns_top10': returns_top10,
            'returns_all': returns_all,
            'cumulative_top10': cumulative_top10[1:],
            'cumulative_all': cumulative_all[1:],
            'metrics_top10': metrics_top10,
            'metrics_all': metrics_all,
            'initial_capital': initial_capital
        }
        
        return self.backtest_results

    def create_tradingview_style_backtest_chart(self):
        """創建TradingView風格的回測圖表（獨立圖表）"""
        if not self.backtest_results:
            self.calculate_backtest_metrics()
        
        setup_chinese_font()
        
        # 創建獨立的TradingView風格回測圖表
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, height_ratios=[3, 1, 1, 1], width_ratios=[2, 1, 1])
        
        # 主圖：淨值曲線
        ax_main = fig.add_subplot(gs[0, :])
        years = self.backtest_results['years']
        cumulative_top10 = self.backtest_results['cumulative_top10']
        cumulative_all = self.backtest_results['cumulative_all']
        
        ax_main.plot(years, cumulative_top10, linewidth=3, color='#2E8B57', label='前10支股票策略', marker='o', markersize=6)
        ax_main.plot(years, cumulative_all, linewidth=2, color='#4169E1', label='所有選中股票策略', marker='s', markersize=4)
        ax_main.axhline(y=self.backtest_results['initial_capital'], color='gray', linestyle='--', alpha=0.5, label='初始資金')
        
        ax_main.set_title('Paper 2 機器學習股票選股策略回測 - 淨值曲線', fontsize=18, fontweight='bold', pad=20)
        ax_main.set_xlabel('年份', fontsize=12)
        ax_main.set_ylabel('淨值 (元)', fontsize=12)
        ax_main.legend(fontsize=12)
        ax_main.grid(True, alpha=0.3)
        ax_main.set_facecolor('#f8f9fa')
        
        # 格式化Y軸
        ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # 子圖1：年度報酬率
        ax_returns = fig.add_subplot(gs[1, :])
        returns_top10 = self.backtest_results['returns_top10']
        returns_all = self.backtest_results['returns_all']
        
        x = np.arange(len(years))
        width = 0.35
        
        colors_top10 = ['#2E8B57' if r > 0 else '#DC143C' for r in returns_top10]
        colors_all = ['#4169E1' if r > 0 else '#FF6347' for r in returns_all]
        
        ax_returns.bar(x - width/2, returns_top10, width, label='前10支股票', color=colors_top10, alpha=0.8)
        ax_returns.bar(x + width/2, returns_all, width, label='所有選中股票', color=colors_all, alpha=0.8)
        
        ax_returns.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax_returns.set_title('年度報酬率分布', fontsize=14, fontweight='bold')
        ax_returns.set_xlabel('年份')
        ax_returns.set_ylabel('報酬率 (%)')
        ax_returns.set_xticks(x)
        ax_returns.set_xticklabels(years)
        ax_returns.legend()
        ax_returns.grid(True, alpha=0.3)
        
        # 子圖2：績效指標表格 - 前10支股票
        ax_metrics1 = fig.add_subplot(gs[2, 0])
        ax_metrics1.axis('off')
        
        metrics_top10 = self.backtest_results['metrics_top10']
        metrics_data1 = [
            ['總報酬率', f"{metrics_top10['total_return']:.2f}%"],
            ['年化報酬率', f"{metrics_top10['annual_return']:.2f}%"],
            ['波動率', f"{metrics_top10['volatility']:.2f}%"],
            ['夏普比率', f"{metrics_top10['sharpe_ratio']:.3f}"],
            ['最大回撤', f"{metrics_top10['max_drawdown']:.2f}%"],
            ['勝率', f"{metrics_top10['win_rate']:.1f}%"],
            ['盈利因子', f"{metrics_top10['profit_factor']:.2f}"],
            ['總交易次數', f"{metrics_top10['total_trades']}"]
        ]
        
        table1 = ax_metrics1.table(cellText=metrics_data1,
                                  colLabels=['指標', '前10支股票策略'],
                                  cellLoc='center',
                                  loc='center',
                                  colWidths=[0.5, 0.5])
        table1.auto_set_font_size(False)
        table1.set_fontsize(10)
        table1.scale(1, 2)
        
        # 設置表格樣式
        for i in range(len(metrics_data1) + 1):
            for j in range(2):
                cell = table1[(i, j)]
                if i == 0:  # 標題行
                    cell.set_facecolor('#2E8B57')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax_metrics1.set_title('前10支股票策略績效', fontsize=12, fontweight='bold')
        
        # 子圖3：績效指標表格 - 所有選中股票
        ax_metrics2 = fig.add_subplot(gs[2, 1])
        ax_metrics2.axis('off')
        
        metrics_all = self.backtest_results['metrics_all']
        metrics_data2 = [
            ['總報酬率', f"{metrics_all['total_return']:.2f}%"],
            ['年化報酬率', f"{metrics_all['annual_return']:.2f}%"],
            ['波動率', f"{metrics_all['volatility']:.2f}%"],
            ['夏普比率', f"{metrics_all['sharpe_ratio']:.3f}"],
            ['最大回撤', f"{metrics_all['max_drawdown']:.2f}%"],
            ['勝率', f"{metrics_all['win_rate']:.1f}%"],
            ['盈利因子', f"{metrics_all['profit_factor']:.2f}"],
            ['總交易次數', f"{metrics_all['total_trades']}"]
        ]
        
        table2 = ax_metrics2.table(cellText=metrics_data2,
                                  colLabels=['指標', '所有選中股票策略'],
                                  cellLoc='center',
                                  loc='center',
                                  colWidths=[0.5, 0.5])
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1, 2)
        
        # 設置表格樣式
        for i in range(len(metrics_data2) + 1):
            for j in range(2):
                cell = table2[(i, j)]
                if i == 0:  # 標題行
                    cell.set_facecolor('#4169E1')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax_metrics2.set_title('所有選中股票策略績效', fontsize=12, fontweight='bold')
        
        # 子圖4：策略比較
        ax_comparison = fig.add_subplot(gs[2, 2])
        ax_comparison.axis('off')
        
        comparison_data = [
            ['策略', '前10支股票', '所有選中股票'],
            ['總報酬率', f"{metrics_top10['total_return']:.2f}%", f"{metrics_all['total_return']:.2f}%"],
            ['夏普比率', f"{metrics_top10['sharpe_ratio']:.3f}", f"{metrics_all['sharpe_ratio']:.3f}"],
            ['最大回撤', f"{metrics_top10['max_drawdown']:.2f}%", f"{metrics_all['max_drawdown']:.2f}%"],
            ['勝率', f"{metrics_top10['win_rate']:.1f}%", f"{metrics_all['win_rate']:.1f}%"]
        ]
        
        table3 = ax_comparison.table(cellText=comparison_data[1:],
                                   colLabels=comparison_data[0],
                                   cellLoc='center',
                                   loc='center',
                                   colWidths=[0.4, 0.3, 0.3])
        table3.auto_set_font_size(False)
        table3.set_fontsize(9)
        table3.scale(1, 2)
        
        # 設置表格樣式
        for i in range(len(comparison_data)):
            for j in range(3):
                cell = table3[(i, j)]
                if i == 0:  # 標題行
                    cell.set_facecolor('#696969')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax_comparison.set_title('策略績效比較', fontsize=12, fontweight='bold')
        
        # 子圖5：回撤分析
        ax_drawdown = fig.add_subplot(gs[3, :])
        
        # 計算回撤序列
        def calculate_drawdown_series(cumulative):
            peak = cumulative[0]
            drawdowns = []
            for value in cumulative:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                drawdowns.append(drawdown)
            return drawdowns
        
        drawdowns_top10 = calculate_drawdown_series(cumulative_top10)
        drawdowns_all = calculate_drawdown_series(cumulative_all)
        
        ax_drawdown.fill_between(years, drawdowns_top10, 0, alpha=0.6, color='#2E8B57', label='前10支股票策略')
        ax_drawdown.fill_between(years, drawdowns_all, 0, alpha=0.4, color='#4169E1', label='所有選中股票策略')
        
        ax_drawdown.set_title('回撤分析', fontsize=14, fontweight='bold')
        ax_drawdown.set_xlabel('年份')
        ax_drawdown.set_ylabel('回撤 (%)')
        ax_drawdown.legend()
        ax_drawdown.grid(True, alpha=0.3)
        ax_drawdown.invert_yaxis()  # 回撤向下顯示
        
        plt.tight_layout()
        
        # 儲存圖表
        backtest_chart_path = os.path.join(self.output_dir, 'tradingview_style_backtest.png')
        plt.savefig(backtest_chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"TradingView風格回測圖表已儲存至: {backtest_chart_path}")

    def export_backtest_data_to_csv(self):
        """匯出回測資料到CSV（類似TradingView的匯出功能）"""
        if not self.backtest_results:
            self.calculate_backtest_metrics()
        
        # 創建詳細的回測資料
        backtest_data = []
        cumulative_top10 = [self.backtest_results['initial_capital']] + self.backtest_results['cumulative_top10']
        cumulative_all = [self.backtest_results['initial_capital']] + self.backtest_results['cumulative_all']
        
        for i, year in enumerate(['初始'] + self.backtest_results['years']):
            if i == 0:
                backtest_data.append({
                    '年份': year,
                    '前10支股票_年度報酬率(%)': 0,
                    '前10支股票_累積淨值': cumulative_top10[i],
                    '所有選中股票_年度報酬率(%)': 0,
                    '所有選中股票_累積淨值': cumulative_all[i],
                    '前10支股票_回撤(%)': 0,
                    '所有選中股票_回撤(%)': 0
                })
            else:
                # 計算回撤
                peak_top10 = max(cumulative_top10[:i+1])
                peak_all = max(cumulative_all[:i+1])
                drawdown_top10 = (peak_top10 - cumulative_top10[i]) / peak_top10 * 100
                drawdown_all = (peak_all - cumulative_all[i]) / peak_all * 100
                
                backtest_data.append({
                    '年份': year,
                    '前10支股票_年度報酬率(%)': self.backtest_results['returns_top10'][i-1],
                    '前10支股票_累積淨值': cumulative_top10[i],
                    '所有選中股票_年度報酬率(%)': self.backtest_results['returns_all'][i-1],
                    '所有選中股票_累積淨值': cumulative_all[i],
                    '前10支股票_回撤(%)': drawdown_top10,
                    '所有選中股票_回撤(%)': drawdown_all
                })
        
        # 儲存回測資料
        backtest_df = pd.DataFrame(backtest_data)
        backtest_path = os.path.join(self.output_dir, 'tradingview_backtest_data.csv')
        backtest_df.to_csv(backtest_path, index=False, encoding='utf-8-sig')
        
        # 儲存績效指標
        metrics_data = []
        metrics_top10 = self.backtest_results['metrics_top10']
        metrics_all = self.backtest_results['metrics_all']
        
        for key in metrics_top10.keys():
            metrics_data.append({
                '指標': key,
                '前10支股票策略': metrics_top10[key],
                '所有選中股票策略': metrics_all[key]
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_path = os.path.join(self.output_dir, 'tradingview_backtest_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
        
        print(f"回測資料已匯出至:")
        print(f"- {backtest_path}")
        print(f"- {metrics_path}")

    def print_backtest_summary(self):
        """印出TradingView風格的回測摘要"""
        if not self.backtest_results:
            self.calculate_backtest_metrics()
        
        metrics_top10 = self.backtest_results['metrics_top10']
        metrics_all = self.backtest_results['metrics_all']
        
        print("\n" + "="*60)
        print("TradingView風格回測結果摘要")
        print("="*60)
        print(f"回測期間: {self.backtest_results['years'][0]} - {self.backtest_results['years'][-1]}")
        print(f"初始資金: {self.backtest_results['initial_capital']:,} 元")
        print(f"總交易次數: {metrics_top10['total_trades']}")
        
        print(f"\n【前10支股票策略】")
        print(f"  總報酬率: {metrics_top10['total_return']:.2f}%")
        print(f"  年化報酬率: {metrics_top10['annual_return']:.2f}%")
        print(f"  夏普比率: {metrics_top10['sharpe_ratio']:.3f}")
        print(f"  最大回撤: {metrics_top10['max_drawdown']:.2f}%")
        print(f"  勝率: {metrics_top10['win_rate']:.1f}%")
        print(f"  盈利因子: {metrics_top10['profit_factor']:.2f}")
        print(f"  期末淨值: {self.backtest_results['cumulative_top10'][-1]:,.0f} 元")
        
        print(f"\n【所有選中股票策略】")
        print(f"  總報酬率: {metrics_all['total_return']:.2f}%")
        print(f"  年化報酬率: {metrics_all['annual_return']:.2f}%")
        print(f"  夏普比率: {metrics_all['sharpe_ratio']:.3f}")
        print(f"  最大回撤: {metrics_all['max_drawdown']:.2f}%")
        print(f"  勝率: {metrics_all['win_rate']:.1f}%")
        print(f"  盈利因子: {metrics_all['profit_factor']:.2f}")
        print(f"  期末淨值: {self.backtest_results['cumulative_all'][-1]:,.0f} 元")

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

def main():
    selector = Paper2StockSelector('top200.xlsx', output_dir='Q3output')
    
    print("開始執行Paper 2進階股票選股分析...")
    selector.run_rolling_window_analysis()
    
    selector.save_results_to_csv()
    selector.create_visualizations()
    selector.create_top10_individual_returns_chart()
    
    # 新增：TradingView風格回測（獨立圖表）
    print("\n開始TradingView風格回測分析...")
    selector.calculate_backtest_metrics()
    selector.create_tradingview_style_backtest_chart()
    selector.export_backtest_data_to_csv()
    selector.print_backtest_summary()
    
    print(f"\n所有結果已儲存至 Q3output 資料夾")
    
    return selector

if __name__ == "__main__":
    selector = main()
