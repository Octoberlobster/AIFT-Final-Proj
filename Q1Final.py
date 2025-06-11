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
                    y_pred_proba = knn.predict_proba(X_test_scaled)[:, 1]  # 獲取正類概率
                    
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
                        # 如果沒有預測為1的股票，仍然記錄前10支股票
                        if avg_return_top10 > best_return_top10:
                            best_return_top10 = avg_return_top10
                            best_return = 0  # 沒有選中股票
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
            
            # 儲存所有選中股票的詳細資訊
            if result['selected_stocks'] is not None:
                stocks = result['selected_stocks'].copy()
                stocks['訓練年份'] = result['train_year']
                stocks['測試年份'] = result['test_year']
                stocks['使用K值'] = result['best_k']
                stocks['選股類型'] = '預測為高於平均'
                selected_stocks_data.append(stocks)
            
            # 儲存前10支股票的詳細資訊
            if result['top10_stocks'] is not None:
                top10_stocks = result['top10_stocks'].copy()
                top10_stocks['訓練年份'] = result['train_year']
                top10_stocks['測試年份'] = result['test_year']
                top10_stocks['使用K值'] = result['best_k']
                top10_stocks['排名'] = range(1, len(top10_stocks) + 1)
                top10_stocks_data.append(top10_stocks)
        
        # 儲存主要結果
        results_df = pd.DataFrame(results_data)
        results_path = os.path.join(self.output_dir, 'knn_stock_selection_results.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        
        # 儲存所有選中股票詳細資訊
        if selected_stocks_data:
            all_selected_stocks = pd.concat(selected_stocks_data, ignore_index=True)
            stocks_path = os.path.join(self.output_dir, 'selected_stocks_details.csv')
            all_selected_stocks.to_csv(stocks_path, index=False, encoding='utf-8-sig')
        
        # 儲存前10支股票詳細資訊
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
        
        # 1. 年度報酬率趨勢比較（前10 vs 所有選中）
        axes[0, 0].plot(years, returns_top10, marker='o', linewidth=2, markersize=8, color='green', label='前10支股票')
        axes[0, 0].plot(years, returns, marker='s', linewidth=2, markersize=6, color='blue', label='所有選中股票')
        axes[0, 0].set_title('年度平均報酬率趨勢比較')
        axes[0, 0].set_xlabel('年份')
        axes[0, 0].set_ylabel('平均報酬率 (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].legend()
        
        # 2. 最佳K值分布
        k_counts = pd.Series(k_values).value_counts().sort_index()
        axes[0, 1].bar(k_counts.index, k_counts.values, alpha=0.7)
        axes[0, 1].set_title('最佳K值分布')
        axes[0, 1].set_xlabel('K值')
        axes[0, 1].set_ylabel('次數')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 特徵數量 vs 前10支股票報酬率
        scatter = axes[0, 2].scatter(num_features, returns_top10, c=years, cmap='viridis', s=100, alpha=0.7)
        axes[0, 2].set_title('特徵數量 vs 前10支股票平均報酬率')
        axes[0, 2].set_xlabel('特徵數量')
        axes[0, 2].set_ylabel('平均報酬率 (%)')
        axes[0, 2].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 2], label='年份')
        
        # 4. 選中股票數量趨勢
        axes[1, 0].plot(years, num_stocks, marker='s', linewidth=2, markersize=8, color='orange')
        axes[1, 0].set_title('選中股票數量趨勢')
        axes[1, 0].set_xlabel('年份')
        axes[1, 0].set_ylabel('股票數量')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 前10支股票年度報酬率分布
        colors = ['green' if r > 0 else 'red' for r in returns_top10]
        axes[1, 1].bar(years, returns_top10, color=colors, alpha=0.7)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].set_title('前10支股票年度報酬率分布')
        axes[1, 1].set_xlabel('年份')
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
        chart_path = os.path.join(self.output_dir, 'knn_stock_analysis_results.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.show()
        self.create_feature_importance_chart()

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
        individual_chart_path = os.path.join(self.output_dir, 'knn_top10_individual_returns.png')
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
        top10_chart_path = os.path.join(self.output_dir, 'top10_analysis.png')
        plt.savefig(top10_chart_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_feature_importance_chart(self):
        setup_chinese_font()
        feature_counts = {}
        for result in self.results:
            for feature in result['best_features']:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        if not feature_counts:
            return
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        features, counts = zip(*sorted_features)
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), counts, alpha=0.7)
        plt.yticks(range(len(features)), features)
        plt.xlabel('被選中次數')
        plt.title('特徵重要性分析（被選為最佳特徵的次數）')
        plt.grid(True, alpha=0.3)
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, str(counts[i]), ha='left', va='center')
        plt.tight_layout()
        feature_chart_path = os.path.join(self.output_dir, 'feature_importance_analysis.png')
        plt.savefig(feature_chart_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_performance_summary_chart(self):
        setup_chinese_font()
        if not self.results:
            return
        returns = [r['best_return'] for r in self.results]
        returns_top10 = [r['best_return_top10'] for r in self.results]
        years = [r['test_year'] for r in self.results]
        cumulative_returns = np.cumsum(returns)
        cumulative_returns_top10 = np.cumsum(returns_top10)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 年度報酬率比較
        x = np.arange(len(years))
        width = 0.35
        ax1.bar(x - width/2, returns_top10, width, label='前10支股票', alpha=0.7, color='green')
        ax1.bar(x + width/2, returns, width, label='所有選中股票', alpha=0.7, color='blue')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title('年度報酬率比較')
        ax1.set_xlabel('年份')
        ax1.set_ylabel('報酬率 (%)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(years)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 累積報酬率比較
        ax2.plot(years, cumulative_returns_top10, marker='o', linewidth=2, markersize=6, color='green', label='前10支股票')
        ax2.plot(years, cumulative_returns, marker='s', linewidth=2, markersize=6, color='blue', label='所有選中股票')
        ax2.fill_between(years, cumulative_returns_top10, alpha=0.3, color='green')
        ax2.fill_between(years, cumulative_returns, alpha=0.3, color='blue')
        ax2.set_title('累積報酬率趨勢比較')
        ax2.set_xlabel('年份')
        ax2.set_ylabel('累積報酬率 (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        performance_chart_path = os.path.join(self.output_dir, 'performance_summary.png')
        plt.savefig(performance_chart_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_analysis_report(self):
        if not self.results:
            return
        returns = [r['best_return'] for r in self.results]
        returns_top10 = [r['best_return_top10'] for r in self.results]
        
        report = f"""
KNN股票選股模型分析報告
{'='*50}

分析概況:
- 分析期間: {self.results[0]['train_year']}-{self.results[-1]['test_year']}
- 總測試年數: {len(self.results)}
- 使用演算法: K-Nearest Neighbor (KNN)

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

年度詳細結果:
"""
        for result in self.results:
            report += f"""
{result['test_year']}年:
  - 前10支股票報酬率: {result['best_return_top10']:.4f}%
  - 所有選中股票報酬率: {result['best_return']:.4f}%
  - 最佳K值: {result['best_k']}
  - 特徵數量: {result['num_features']}
  - 選中股票數: {result['num_selected_stocks']}
  - 主要特徵: {', '.join(result['best_features'][:3])}{'...' if len(result['best_features']) > 3 else ''}
"""
        best_year_idx = np.argmax(returns_top10)
        best_result = self.results[best_year_idx]
        report += f"""

最佳表現年份（前10支股票）: {best_result['test_year']}
  - 前10支股票報酬率: {best_result['best_return_top10']:.4f}%
  - 所有選中股票報酬率: {best_result['best_return']:.4f}%
  - 使用K值: {best_result['best_k']}
  - 特徵數量: {best_result['num_features']}
  - 選中股票數: {best_result['num_selected_stocks']}

風險指標:
- 前10支股票夏普比率: {np.mean(returns_top10)/np.std(returns_top10):.4f} (假設無風險利率為0)
- 所有選中股票夏普比率: {np.mean(returns)/np.std(returns):.4f} (假設無風險利率為0)
- 前10支股票最大回撤: {self.calculate_max_drawdown(returns_top10):.4f}%
- 所有選中股票最大回撤: {self.calculate_max_drawdown(returns):.4f}%

模型參數統計:
- 最常用K值: {pd.Series([r['best_k'] for r in self.results]).mode().iloc[0]}
- 平均特徵數量: {np.mean([r['num_features'] for r in self.results]):.2f}
- 平均選股數量: {np.mean([r['num_selected_stocks'] for r in self.results]):.2f}

KNN選股策略說明:
- 使用KNN分類器預測股票是否高於平均報酬率
- 根據預測概率排序，選出前10支最有潜力的股票
- 同時保留傳統的分類預測結果作為比較基準
"""
        report_path = os.path.join(self.output_dir, 'analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"分析報告已儲存至: {report_path}")

    def calculate_max_drawdown(self, returns):
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return np.min(drawdown)

    def print_summary(self):
        if not self.results:
            print("沒有結果可顯示")
            return
        returns = [r['best_return'] for r in self.results]
        returns_top10 = [r['best_return_top10'] for r in self.results]
        print("\n" + "="*50)
        print("KNN股票選股模型分析摘要")
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
        print(f"\n最佳表現年份: {best_result['test_year']}")
        print(f"  前10支股票報酬率: {best_result['best_return_top10']:.4f}%")
        print(f"  所有選中股票報酬率: {best_result['best_return']:.4f}%")
        print(f"  使用K值: {best_result['best_k']}")
        print(f"  特徵數量: {best_result['num_features']}")
        print(f"  選中股票數: {best_result['num_selected_stocks']}")

def main():
    selector = StockKNNSelector('top200.xlsx', output_dir='Q1output')
    print("開始執行KNN股票選股分析...")
    selector.run_rolling_window_analysis()
    selector.save_results_to_csv()
    selector.create_visualizations()
    selector.create_top10_individual_returns_chart()  # 新增：前10支股票各自的年化報酬折線圖
    selector.create_top10_analysis_chart()
    selector.create_performance_summary_chart()
    selector.save_analysis_report()
    selector.print_summary()
    print(f"\n所有結果已儲存至 Q1output 資料夾")
    return selector

if __name__ == "__main__":
    selector = main()
