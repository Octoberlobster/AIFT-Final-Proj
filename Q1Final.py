import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
import os
import platform

warnings.filterwarnings('ignore')

# --------- 中文字體設定 ---------
def setup_chinese_font():
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Noto Sans CJK TC', 'AR PL UMing CN']
    plt.rcParams['axes.unicode_minus'] = False

setup_chinese_font()
sns.set_style("darkgrid", {"font.sans-serif": plt.rcParams['font.sans-serif']})

class StockKNNSelector:
    def __init__(self, data_path, output_dir='Q1output'):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.data = self.load_and_preprocess_data(data_path)
        self.feature_columns = [
            '市值(百萬元)', '收盤價(元)_年', '股價淨值比', '股價營收比',
            'M淨值報酬率─稅後', '資產報酬率ROA', '營業利益率OPM', '利潤邊際NPM',
            '負債/淨值比', 'M流動比率', 'M速動比率', 'M存貨週轉率 (次)',
            'M應收帳款週轉次', 'M營業利益成長率', 'M稅後淨利成長率'
        ]
        self.results = []
        self.backtest_results = []  # 新增：儲存回測結果

    def load_and_preprocess_data(self, data_path):
        data = pd.read_excel(data_path, sheet_name='Sheet1')
        data['年份'] = data['年月'].astype(str).str[:4].astype(int)
        data = data[data['年月'] != 200912]
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
        return data

    def get_feature_combinations(self, max_features=8):
        feature_combinations = []
        for feature in self.feature_columns:
            feature_combinations.append([feature])
        for r in range(2, min(6, max_features + 1)):
            if r <= 3:
                for combo in combinations(self.feature_columns[:10], r):
                    feature_combinations.append(list(combo))
            else:
                important_features = ['市值(百萬元)', '股價淨值比', '資產報酬率ROA',
                                    '營業利益率OPM', 'M淨值報酬率─稅後']
                for combo in combinations(important_features, r):
                    feature_combinations.append(list(combo))
        return feature_combinations[:50]

    def train_and_evaluate(self, train_year, test_year, k_values, feature_combinations):
        train_data = self.data[self.data['年份'] == train_year].copy()
        test_data = self.data[self.data['年份'] == test_year].copy()
        if len(train_data) == 0 or len(test_data) == 0:
            return None
        
        best_return = -np.inf
        best_return_top10 = -np.inf
        best_params = {}
        best_selected_stocks = None
        best_top10_stocks = None
        best_model = None
        best_features = None
        
        for features in feature_combinations:
            available_features = [f for f in features if f in train_data.columns]
            if len(available_features) == 0:
                continue
            try:
                X_train = train_data[available_features].values
                y_train = train_data['ReturnMean_year_Label'].values
                X_test = test_data[available_features].values
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                for k in k_values:
                    if k >= len(train_data):
                        continue
                    knn = KNeighborsClassifier(n_neighbors=k)
                    knn.fit(X_train_scaled, y_train)
                    y_pred = knn.predict(X_test_scaled)
                    y_pred_proba = knn.predict_proba(X_test_scaled)[:, 1]
                    
                    test_data_with_proba = test_data.copy()
                    test_data_with_proba['prediction_proba'] = y_pred_proba
                    test_data_sorted = test_data_with_proba.sort_values('prediction_proba', ascending=False)
                    
                    top10_stocks = test_data_sorted.head(10)
                    avg_return_top10 = top10_stocks['Return'].mean()
                    
                    selected_indices = np.where(y_pred == 1)[0]
                    if len(selected_indices) > 0:
                        selected_stocks = test_data.iloc[selected_indices]
                        avg_return = selected_stocks['Return'].mean()
                        
                        if avg_return_top10 > best_return_top10:
                            best_return_top10 = avg_return_top10
                            best_return = avg_return
                            best_params = {
                                'k': k,
                                'features': available_features,
                                'num_features': len(available_features)
                            }
                            best_selected_stocks = selected_stocks
                            best_top10_stocks = top10_stocks
                            best_model = knn
                            best_features = available_features
                    else:
                        if avg_return_top10 > best_return_top10:
                            best_return_top10 = avg_return_top10
                            best_return = 0
                            best_params = {
                                'k': k,
                                'features': available_features,
                                'num_features': len(available_features)
                            }
                            best_selected_stocks = None
                            best_top10_stocks = top10_stocks
                            best_model = knn
                            best_features = available_features
                            
            except Exception:
                continue
                
        return {
            'train_year': train_year,
            'test_year': test_year,
            'best_return': best_return,
            'best_return_top10': best_return_top10,
            'best_k': best_params.get('k', None),
            'best_features': best_params.get('features', []),
            'num_features': best_params.get('num_features', 0),
            'num_selected_stocks': len(best_selected_stocks) if best_selected_stocks is not None else 0,
            'selected_stocks': best_selected_stocks,
            'top10_stocks': best_top10_stocks,
            'best_model': best_model
        }

    def run_rolling_window_analysis(self):
        years = sorted(self.data['年份'].unique())
        k_values = [1, 3, 5, 7, 9, 11, 15, 21]
        feature_combinations = self.get_feature_combinations()
        print(f"開始分析，共有 {len(feature_combinations)} 種特徵組合")
        print(f"K值範圍: {k_values}")
        print(f"年份範圍: {years}")
        for i in range(len(years) - 1):
            train_year = years[i]
            test_year = years[i + 1]
            print(f"\n分析 {train_year} -> {test_year}")
            result = self.train_and_evaluate(train_year, test_year, k_values, feature_combinations)
            if result and result['best_return_top10'] != -np.inf:
                self.results.append(result)
                print(f"前10支股票平均報酬率: {result['best_return_top10']:.4f}%")
                print(f"所有選中股票平均報酬率: {result['best_return']:.4f}%")
                print(f"最佳K值: {result['best_k']}")
                print(f"最佳特徵數量: {result['num_features']}")
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
            'cumulative_top10': cumulative_top10[1:],  # 移除初始值
            'cumulative_all': cumulative_all[1:],
            'metrics_top10': metrics_top10,
            'metrics_all': metrics_all,
            'initial_capital': initial_capital
        }
        
        return self.backtest_results

    def create_tradingview_style_backtest_chart(self):
        """創建TradingView風格的回測圖表"""
        if not self.backtest_results:
            self.calculate_backtest_metrics()
        
        setup_chinese_font()
        
        # 創建主要回測圖表
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
        
        ax_main.set_title('KNN股票選股策略回測 - 淨值曲線', fontsize=18, fontweight='bold', pad=20)
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

    def save_results_to_csv(self):
        if not self.results:
            print("沒有結果可儲存")
            return
        
        results_data = []
        selected_stocks_data = []
        top10_stocks_data = []
        
        for result in self.results:
            results_data.append({
                '訓練年份': result['train_year'],
                '測試年份': result['test_year'],
                '前10支股票平均報酬率(%)': result['best_return_top10'],
                '所有選中股票平均報酬率(%)': result['best_return'],
                '最佳K值': result['best_k'],
                '特徵數量': result['num_features'],
                '選中股票數量': result['num_selected_stocks'],
                '最佳特徵組合': ', '.join(result['best_features'])
            })
            
            if result['selected_stocks'] is not None:
                stocks = result['selected_stocks'].copy()
                stocks['訓練年份'] = result['train_year']
                stocks['測試年份'] = result['test_year']
                stocks['使用K值'] = result['best_k']
                stocks['選股類型'] = '預測為高於平均'
                selected_stocks_data.append(stocks)
            
            if result['top10_stocks'] is not None:
                top10_stocks = result['top10_stocks'].copy()
                top10_stocks['訓練年份'] = result['train_year']
                top10_stocks['測試年份'] = result['test_year']
                top10_stocks['使用K值'] = result['best_k']
                top10_stocks['排名'] = range(1, len(top10_stocks) + 1)
                top10_stocks_data.append(top10_stocks)
        
        results_df = pd.DataFrame(results_data)
        results_path = os.path.join(self.output_dir, 'knn_stock_selection_results.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        
        if selected_stocks_data:
            all_selected_stocks = pd.concat(selected_stocks_data, ignore_index=True)
            stocks_path = os.path.join(self.output_dir, 'selected_stocks_details.csv')
            all_selected_stocks.to_csv(stocks_path, index=False, encoding='utf-8-sig')
        
        if top10_stocks_data:
            all_top10_stocks = pd.concat(top10_stocks_data, ignore_index=True)
            top10_path = os.path.join(self.output_dir, 'top10_stocks_details.csv')
            all_top10_stocks.to_csv(top10_path, index=False, encoding='utf-8-sig')
        
        print(f"結果已儲存到 {self.output_dir} 資料夾:")
        print(f"- knn_stock_selection_results.csv (主要結果)")
        print(f"- selected_stocks_details.csv (所有選中股票詳細資訊)")
        print(f"- top10_stocks_details.csv (前10支股票詳細資訊)")
        return results_df

    def create_visualizations(self):
        if not self.results:
            print("沒有結果可視覺化")
            return
        setup_chinese_font()
        years = [r['test_year'] for r in self.results]
        returns = [r['best_return'] for r in self.results]
        returns_top10 = [r['best_return_top10'] for r in self.results]
        k_values = [r['best_k'] for r in self.results]
        num_features = [r['num_features'] for r in self.results]
        num_stocks = [r['num_selected_stocks'] for r in self.results]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('KNN股票選股模型分析結果', fontsize=16, fontweight='bold')
        
        axes[0, 0].plot(years, returns_top10, marker='o', linewidth=2, markersize=8, color='green', label='前10支股票')
        axes[0, 0].plot(years, returns, marker='s', linewidth=2, markersize=6, color='blue', label='所有選中股票')
        axes[0, 0].set_title('年度平均報酬率趨勢比較')
        axes[0, 0].set_xlabel('年份')
        axes[0, 0].set_ylabel('平均報酬率 (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].legend()
        
        k_counts = pd.Series(k_values).value_counts().sort_index()
        axes[0, 1].bar(k_counts.index, k_counts.values, alpha=0.7)
        axes[0, 1].set_title('最佳K值分布')
        axes[0, 1].set_xlabel('K值')
        axes[0, 1].set_ylabel('次數')
        axes[0, 1].grid(True, alpha=0.3)
        
        scatter = axes[0, 2].scatter(num_features, returns_top10, c=years, cmap='viridis', s=100, alpha=0.7)
        axes[0, 2].set_title('特徵數量 vs 前10支股票平均報酬率')
        axes[0, 2].set_xlabel('特徵數量')
        axes[0, 2].set_ylabel('平均報酬率 (%)')
        axes[0, 2].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 2], label='年份')
        
        axes[1, 0].plot(years, num_stocks, marker='s', linewidth=2, markersize=8, color='orange')
        axes[1, 0].set_title('選中股票數量趨勢')
        axes[1, 0].set_xlabel('年份')
        axes[1, 0].set_ylabel('股票數量')
        axes[1, 0].grid(True, alpha=0.3)
        
        colors = ['green' if r > 0 else 'red' for r in returns_top10]
        axes[1, 1].bar(years, returns_top10, color=colors, alpha=0.7)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].set_title('前10支股票年度報酬率分布')
        axes[1, 1].set_xlabel('年份')
        axes[1, 1].set_ylabel('平均報酬率 (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
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
        chart_path = os.path.join(self.output_dir, 'knn_stock_analysis_results.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.show()

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

def main():
    selector = StockKNNSelector('top200.xlsx', output_dir='Q1output')
    print("開始執行KNN股票選股分析...")
    selector.run_rolling_window_analysis()
    selector.save_results_to_csv()
    selector.create_visualizations()
    
    # 新增：TradingView風格回測
    print("\n開始TradingView風格回測分析...")
    selector.calculate_backtest_metrics()
    selector.create_tradingview_style_backtest_chart()
    selector.export_backtest_data_to_csv()
    selector.print_backtest_summary()
    
    print(f"\n所有結果已儲存至 Q1output 資料夾")
    return selector

if __name__ == "__main__":
    selector = main()
