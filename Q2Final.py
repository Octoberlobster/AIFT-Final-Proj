import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
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

class StockDecisionTreeSelector:
    def __init__(self, data_path, output_dir='Q2outputCraw'):
        self.output_dir = output_dir
        self.create_output_directory()
        self.data = self.load_and_preprocess_data(data_path)
        # 根據您的資料調整特徵欄位名稱
        self.feature_columns = [
            '市值(百萬元)', '收盤價(元)', 'PBR', 'PER',
            'ROE(%)', 'ROA(%)', '營業利益率(%)', '稅後淨利率(%)',
            '負債/淨值比', '流動比率', '速動比率', '存貨周轉率',
            '應收帳款周轉次', 'M營業利益成長率', 'M稅後淨利成長率'
        ]
        self.results = []
        
    def create_output_directory(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"已創建輸出資料夾: {self.output_dir}")
        else:
            print(f"輸出資料夾已存在: {self.output_dir}")
        
    def load_and_preprocess_data(self, data_path):
        # 讀取Excel檔案
        data = pd.read_excel(data_path, sheet_name='Sheet1')
        
        # 從年月欄位提取年份
        data['年份'] = data['年月'].astype(str).str[:4].astype(int)
        
        # 移除特定年月的資料（如果需要）
        data = data[data['年月'] != 200912]
        
        # 處理Return欄位 - 將空值轉換為0或移除
        data['Return'] = pd.to_numeric(data['Return'], errors='coerce')
        data = data.dropna(subset=['Return'])
        
        # 創建Return標籤（高於平均為1，低於平均為0）
        yearly_mean_returns = data.groupby('年份')['Return'].transform('mean')
        data['ReturnMean_year_Label'] = (data['Return'] > yearly_mean_returns).astype(int)
        
        # 處理數值欄位的缺失值
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
        
        # 處理包含特殊字符的欄位（如+號、逗號等）
        for col in ['M營業利益成長率', 'M稅後淨利成長率']:
            if col in data.columns:
                data[col] = data[col].astype(str).str.replace('+', '').str.replace(',', '')
                data[col] = pd.to_numeric(data[col], errors='coerce')
                data[col] = data[col].fillna(0)
        
        # 確保PER和PBR欄位存在且為數值
        if 'PER' not in data.columns and 'P/E' in data.columns:
            data['PER'] = data['P/E']
        if 'PBR' not in data.columns and 'P/B' in data.columns:
            data['PBR'] = data['P/B']
            
        return data
    
    def get_feature_combinations(self, max_features=8):
        # 檢查哪些特徵欄位實際存在於資料中
        available_features = [f for f in self.feature_columns if f in self.data.columns]
        print(f"可用特徵: {available_features}")
        
        feature_combinations = []
        
        # 單一特徵組合
        for feature in available_features:
            feature_combinations.append([feature])
        
        # 多特徵組合
        for r in range(2, min(6, max_features + 1)):
            if r <= 3:
                # 使用前10個特徵進行組合
                features_to_combine = available_features[:min(10, len(available_features))]
                for combo in combinations(features_to_combine, r):
                    feature_combinations.append(list(combo))
            else:
                # 重要特徵組合
                important_features = ['市值(百萬元)', 'PBR', 'ROA(%)', 
                                    '營業利益率(%)', 'ROE(%)']
                important_available = [f for f in important_features if f in available_features]
                if len(important_available) >= r:
                    for combo in combinations(important_available, r):
                        feature_combinations.append(list(combo))
        
        return feature_combinations[:50]
    
    def get_decision_tree_params(self):
        param_combinations = []
        criterions = ['gini', 'entropy']
        max_depths = [3, 5, 7, 10, 15, None]
        min_samples_splits = [2, 5, 10, 20]
        min_samples_leafs = [1, 2, 5, 10]
        
        for criterion in criterions:
            for max_depth in max_depths[:4]:
                for min_samples_split in min_samples_splits[:3]:
                    for min_samples_leaf in min_samples_leafs[:3]:
                        param_combinations.append({
                            'criterion': criterion,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'random_state': 42
                        })
        return param_combinations[:30]
    
    def train_and_evaluate(self, train_year, test_year, feature_combinations, param_combinations):
        train_data = self.data[self.data['年份'] == train_year].copy()
        test_data = self.data[self.data['年份'] == test_year].copy()
        
        if len(train_data) == 0 or len(test_data) == 0:
            return None
        
        best_return = -np.inf
        best_return_top10 = -np.inf
        best_params = {}
        best_selected_stocks = None
        best_top10_stocks = None
        best_tree = None
        best_features = None
        
        for features in feature_combinations:
            available_features = [f for f in features if f in train_data.columns]
            if len(available_features) == 0:
                continue
                
            try:
                X_train = train_data[available_features].values
                y_train = train_data['ReturnMean_year_Label'].values
                X_test = test_data[available_features].values
                
                # 標準化
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                for params in param_combinations:
                    dt = DecisionTreeClassifier(**params)
                    dt.fit(X_train_scaled, y_train)
                    
                    y_pred = dt.predict(X_test_scaled)
                    y_pred_proba = dt.predict_proba(X_test_scaled)[:, 1]
                    
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
                                'dt_params': params,
                                'features': available_features,
                                'num_features': len(available_features)
                            }
                            best_selected_stocks = selected_stocks
                            best_top10_stocks = top10_stocks
                            best_tree = dt
                            best_features = available_features
                    else:
                        # 如果沒有預測為1的股票，仍然記錄前10支股票
                        if avg_return_top10 > best_return_top10:
                            best_return_top10 = avg_return_top10
                            best_return = 0
                            best_params = {
                                'dt_params': params,
                                'features': available_features,
                                'num_features': len(available_features)
                            }
                            best_selected_stocks = None
                            best_top10_stocks = top10_stocks
                            best_tree = dt
                            best_features = available_features
                            
            except Exception as e:
                print(f"處理特徵組合時發生錯誤: {features}, 錯誤: {e}")
                continue
        
        return {
            'train_year': train_year,
            'test_year': test_year,
            'best_return': best_return,
            'best_return_top10': best_return_top10,
            'best_params': best_params,
            'num_selected_stocks': len(best_selected_stocks) if best_selected_stocks is not None else 0,
            'selected_stocks': best_selected_stocks,
            'top10_stocks': best_top10_stocks,
            'best_tree': best_tree,
            'best_features': best_features
        }
    
    def run_rolling_window_analysis(self):
        years = sorted(self.data['年份'].unique())
        feature_combinations = self.get_feature_combinations()
        param_combinations = self.get_decision_tree_params()
        
        print(f"開始分析，共有 {len(feature_combinations)} 種特徵組合")
        print(f"決策樹參數組合: {len(param_combinations)} 種")
        print(f"年份範圍: {years}")
        
        for i in range(len(years) - 1):
            train_year = years[i]
            test_year = years[i + 1]
            print(f"\n分析 {train_year} -> {test_year}")
            
            result = self.train_and_evaluate(train_year, test_year, feature_combinations, param_combinations)
            if result and result['best_return_top10'] != -np.inf:
                self.results.append(result)
                print(f"前10支股票平均報酬率: {result['best_return_top10']:.4f}%")
                print(f"所有選中股票平均報酬率: {result['best_return']:.4f}%")
                print(f"最佳參數: {result['best_params']['dt_params']}")
                print(f"最佳特徵數量: {result['best_params']['num_features']}")
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
            dt_params = result['best_params']['dt_params']
            results_data.append({
                '訓練年份': result['train_year'],
                '測試年份': result['test_year'],
                '前10支股票平均報酬率(%)': result['best_return_top10'],
                '所有選中股票平均報酬率(%)': result['best_return'],
                '最佳準則': dt_params['criterion'],
                '最大深度': dt_params['max_depth'],
                '最小分割樣本數': dt_params['min_samples_split'],
                '最小葉子樣本數': dt_params['min_samples_leaf'],
                '特徵數量': result['best_params']['num_features'],
                '選中股票數量': result['num_selected_stocks'],
                '最佳特徵組合': ', '.join(result['best_params']['features'])
            })
            
            # 儲存所有選中股票詳細資訊
            if result['selected_stocks'] is not None:
                stocks = result['selected_stocks'].copy()
                stocks['訓練年份'] = result['train_year']
                stocks['測試年份'] = result['test_year']
                stocks['使用準則'] = dt_params['criterion']
                stocks['最大深度'] = dt_params['max_depth']
                stocks['選股類型'] = '預測為高於平均'
                selected_stocks_data.append(stocks)
            
            # 儲存前10支股票詳細資訊
            if result['top10_stocks'] is not None:
                top10_stocks = result['top10_stocks'].copy()
                top10_stocks['訓練年份'] = result['train_year']
                top10_stocks['測試年份'] = result['test_year']
                top10_stocks['使用準則'] = dt_params['criterion']
                top10_stocks['最大深度'] = dt_params['max_depth']
                top10_stocks['排名'] = range(1, len(top10_stocks) + 1)
                top10_stocks_data.append(top10_stocks)
        
        # 儲存主要結果
        results_df = pd.DataFrame(results_data)
        results_path = os.path.join(self.output_dir, 'decision_tree_stock_selection_results.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        
        # 儲存所有選中股票詳細資訊
        if selected_stocks_data:
            all_selected_stocks = pd.concat(selected_stocks_data, ignore_index=True)
            stocks_path = os.path.join(self.output_dir, 'dt_selected_stocks_details.csv')
            all_selected_stocks.to_csv(stocks_path, index=False, encoding='utf-8-sig')
        
        # 儲存前10支股票詳細資訊
        if top10_stocks_data:
            all_top10_stocks = pd.concat(top10_stocks_data, ignore_index=True)
            top10_path = os.path.join(self.output_dir, 'dt_top10_stocks_details.csv')
            all_top10_stocks.to_csv(top10_path, index=False, encoding='utf-8-sig')
        
        print(f"結果已儲存到 {self.output_dir} 資料夾:")
        print(f"- decision_tree_stock_selection_results.csv (主要結果)")
        print(f"- dt_selected_stocks_details.csv (所有選中股票詳細資訊)")
        print(f"- dt_top10_stocks_details.csv (前10支股票詳細資訊)")
        return results_df

    def create_visualizations(self):
        if not self.results:
            print("沒有結果可視覺化")
            return
        
        setup_chinese_font()
        
        years = [r['test_year'] for r in self.results]
        returns = [r['best_return'] for r in self.results]
        returns_top10 = [r['best_return_top10'] for r in self.results]
        criterions = [r['best_params']['dt_params']['criterion'] for r in self.results]
        max_depths = [r['best_params']['dt_params']['max_depth'] for r in self.results]
        num_features = [r['best_params']['num_features'] for r in self.results]
        num_stocks = [r['num_selected_stocks'] for r in self.results]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('決策樹股票選股模型分析結果', fontsize=16, fontweight='bold')
        
        # 1. 年度報酬率趨勢比較
        axes[0, 0].plot(years, returns_top10, marker='o', linewidth=2, markersize=8, color='green', label='前10支股票')
        axes[0, 0].plot(years, returns, marker='s', linewidth=2, markersize=6, color='blue', label='所有選中股票')
        axes[0, 0].set_title('年度平均報酬率趨勢比較')
        axes[0, 0].set_xlabel('年份')
        axes[0, 0].set_ylabel('平均報酬率 (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].legend()
        
        # 2. 最佳準則分布
        criterion_counts = pd.Series(criterions).value_counts()
        axes[0, 1].bar(criterion_counts.index, criterion_counts.values, alpha=0.7, color=['skyblue', 'lightcoral'])
        axes[0, 1].set_title('最佳分割準則分布')
        axes[0, 1].set_xlabel('分割準則')
        axes[0, 1].set_ylabel('次數')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 最大深度分布
        depth_counts = pd.Series([d if d is not None else 'None' for d in max_depths]).value_counts()
        axes[0, 2].bar(range(len(depth_counts)), depth_counts.values, alpha=0.7, color='orange')
        axes[0, 2].set_xticks(range(len(depth_counts)))
        axes[0, 2].set_xticklabels(depth_counts.index, rotation=45)
        axes[0, 2].set_title('最佳最大深度分布')
        axes[0, 2].set_xlabel('最大深度')
        axes[0, 2].set_ylabel('次數')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 特徵數量 vs 前10支股票報酬率
        scatter = axes[1, 0].scatter(num_features, returns_top10, c=years, cmap='viridis', s=100, alpha=0.7)
        axes[1, 0].set_title('特徵數量 vs 前10支股票平均報酬率')
        axes[1, 0].set_xlabel('特徵數量')
        axes[1, 0].set_ylabel('平均報酬率 (%)')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='年份')
        
        # 5. 選中股票數量趨勢
        axes[1, 1].plot(years, num_stocks, marker='s', linewidth=2, markersize=8, color='orange')
        axes[1, 1].set_title('選中股票數量趨勢')
        axes[1, 1].set_xlabel('年份')
        axes[1, 1].set_ylabel('股票數量')
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
        chart_path = os.path.join(self.output_dir, 'decision_tree_analysis_results.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.create_feature_importance_chart()
        self.create_decision_tree_visualization()

    def create_feature_importance_chart(self):
        setup_chinese_font()
        
        feature_counts = {}
        for result in self.results:
            for feature in result['best_params']['features']:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        if not feature_counts:
            return
        
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        features, counts = zip(*sorted_features)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), counts, alpha=0.7, color='lightgreen')
        plt.yticks(range(len(features)), features)
        plt.xlabel('被選中次數')
        plt.title('特徵重要性分析（被選為最佳特徵的次數）')
        plt.grid(True, alpha=0.3)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    str(counts[i]), ha='left', va='center')
        
        plt.tight_layout()
        feature_chart_path = os.path.join(self.output_dir, 'dt_feature_importance_analysis.png')
        plt.savefig(feature_chart_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_decision_tree_visualization(self):
        if not self.results:
            return
        
        setup_chinese_font()
        
        best_result = max(self.results, key=lambda x: x['best_return_top10'])
        
        if best_result['best_tree'] is not None:
            plt.figure(figsize=(20, 12))
            plot_tree(best_result['best_tree'], 
                     feature_names=best_result['best_features'],
                     class_names=['低於平均', '高於平均'],
                     filled=True, 
                     rounded=True,
                     fontsize=10)
            plt.title(f'最佳決策樹視覺化 ({best_result["test_year"]}年)', fontsize=16, fontweight='bold')
            
            tree_chart_path = os.path.join(self.output_dir, 'best_decision_tree_visualization.png')
            plt.savefig(tree_chart_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            tree_rules = export_text(best_result['best_tree'], 
                                   feature_names=best_result['best_features'])
            
            rules_path = os.path.join(self.output_dir, 'decision_tree_rules.txt')
            with open(rules_path, 'w', encoding='utf-8') as f:
                f.write(f"最佳決策樹規則 ({best_result['test_year']}年)\n")
                f.write("="*50 + "\n\n")
                f.write(tree_rules)
            
            print(f"決策樹規則已儲存至: {rules_path}")

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
        print("決策樹股票選股模型分析摘要")
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
        best_dt_params = best_result['best_params']['dt_params']
        
        print(f"\n最佳表現年份: {best_result['test_year']}")
        print(f"  前10支股票報酬率: {best_result['best_return_top10']:.4f}%")
        print(f"  所有選中股票報酬率: {best_result['best_return']:.4f}%")
        print(f"  分割準則: {best_dt_params['criterion']}")
        print(f"  最大深度: {best_dt_params['max_depth']}")
        print(f"  特徵數量: {best_result['best_params']['num_features']}")
        print(f"  選中股票數: {best_result['num_selected_stocks']}")

def main():
    # 使用您的Excel檔案名稱
    selector = StockDecisionTreeSelector('top200craw.xlsx', output_dir='Q2outputCraw')
    
    print("開始執行決策樹股票選股分析...")
    selector.run_rolling_window_analysis()
    selector.save_results_to_csv()
    selector.create_visualizations()
    selector.print_summary()
    
    print(f"\n所有結果已儲存至 Q2outputCraw 資料夾")
    return selector

if __name__ == "__main__":
    selector = main()
