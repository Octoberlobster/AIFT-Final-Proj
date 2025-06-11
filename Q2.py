import pandas as pd
import numpy as np
import math
from collections import Counter
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ID3DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.feature_names = None
        
    def entropy(self, y):
        """計算熵值"""
        if len(y) == 0:
            return 0
        
        counts = Counter(y)
        total = len(y)
        entropy = 0
        
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def information_gain(self, X_column, y, threshold=None):
        """計算資訊增益"""
        parent_entropy = self.entropy(y)
        
        if threshold is not None:
            # 連續變數處理
            left_mask = X_column <= threshold
            right_mask = X_column > threshold
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                return 0
            
            left_entropy = self.entropy(y[left_mask])
            right_entropy = self.entropy(y[right_mask])
            
            left_weight = np.sum(left_mask) / len(y)
            right_weight = np.sum(right_mask) / len(y)
            
            weighted_entropy = left_weight * left_entropy + right_weight * right_entropy
        else:
            # 類別變數處理
            unique_values = np.unique(X_column)
            weighted_entropy = 0
            
            for value in unique_values:
                mask = X_column == value
                subset_entropy = self.entropy(y[mask])
                weight = np.sum(mask) / len(y)
                weighted_entropy += weight * subset_entropy
        
        return parent_entropy - weighted_entropy
    
    def find_best_split(self, X, y, feature_names):
        """找到最佳分割特徵和閾值"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for i, feature_name in enumerate(feature_names):
            X_column = X[:, i]
            
            # 處理連續變數
            if np.issubdtype(X_column.dtype, np.number):
                unique_values = np.unique(X_column)
                
                if len(unique_values) > 1:
                    for j in range(len(unique_values) - 1):
                        threshold = (unique_values[j] + unique_values[j + 1]) / 2
                        gain = self.information_gain(X_column, y, threshold)
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_feature = i
                            best_threshold = threshold
            else:
                # 處理類別變數
                gain = self.information_gain(X_column, y)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = i
                    best_threshold = None
        
        return best_feature, best_threshold, best_gain
    
    def build_tree(self, X, y, feature_names, depth=0):
        """遞迴建立決策樹"""
        # 停止條件
        if len(set(y)) == 1:
            return {'type': 'leaf', 'label': y[0]}
        
        if depth >= self.max_depth or len(feature_names) == 0 or len(y) < self.min_samples_split:
            most_common = Counter(y).most_common(1)[0][0]
            return {'type': 'leaf', 'label': most_common}
        
        # 找到最佳分割
        best_feature, best_threshold, best_gain = self.find_best_split(X, y, feature_names)
        
        if best_gain == 0 or best_feature is None:
            most_common = Counter(y).most_common(1)[0][0]
            return {'type': 'leaf', 'label': most_common}
        
        # 建立節點
        node = {
            'type': 'node',
            'feature': best_feature,
            'feature_name': feature_names[best_feature],
            'threshold': best_threshold,
            'children': {}
        }
        
        # 分割資料
        if best_threshold is not None:
            # 連續變數分割
            left_mask = X[:, best_feature] <= best_threshold
            right_mask = X[:, best_feature] > best_threshold
            
            if np.sum(left_mask) > 0:
                node['children']['left'] = self.build_tree(
                    X[left_mask], y[left_mask], feature_names, depth + 1
                )
            
            if np.sum(right_mask) > 0:
                node['children']['right'] = self.build_tree(
                    X[right_mask], y[right_mask], feature_names, depth + 1
                )
        else:
            # 類別變數分割
            unique_values = np.unique(X[:, best_feature])
            
            for value in unique_values:
                mask = X[:, best_feature] == value
                if np.sum(mask) > 0:
                    node['children'][value] = self.build_tree(
                        X[mask], y[mask], feature_names, depth + 1
                    )
        
        return node
    
    def fit(self, X, y, feature_names):
        """訓練模型"""
        self.feature_names = feature_names
        self.tree = self.build_tree(X, y, feature_names)
    
    def predict_single(self, x, tree=None):
        """預測單一樣本"""
        if tree is None:
            tree = self.tree
        
        if tree['type'] == 'leaf':
            return tree['label']
        
        feature_idx = tree['feature']
        threshold = tree['threshold']
        
        if threshold is not None:
            # 連續變數
            if x[feature_idx] <= threshold:
                if 'left' in tree['children']:
                    return self.predict_single(x, tree['children']['left'])
                else:
                    return 1  # 預設值
            else:
                if 'right' in tree['children']:
                    return self.predict_single(x, tree['children']['right'])
                else:
                    return 1  # 預設值
        else:
            # 類別變數
            value = x[feature_idx]
            if value in tree['children']:
                return self.predict_single(x, tree['children'][value])
            else:
                return 1  # 預設值
    
    def predict(self, X):
        """預測多個樣本"""
        return np.array([self.predict_single(x) for x in X])


class StockSelectionID3:
    def __init__(self, max_depth=8, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.models = {}
        self.selected_features = None
        
    def select_features(self, df, target_col='ReturnMean_year_Label'):
        """特徵選擇"""
        feature_cols = [
            '市值(百萬元)', '收盤價(元)_年', 'Unknown masked parameter',
            '股價淨值比', '股價營收比', 'M淨值報酬率─稅後',
            '資產報酬率ROA', '營業利益率OPM', '利潤邊際NPM',
            '負債/淨值比', 'M流動比率', 'M速動比率',
            'M存貨週轉率 (次)', 'M應收帳款週轉次',
            'M營業利益成長率', 'M稅後淨利成長率'
        ]
        
        # 移除缺失值過多的特徵
        available_features = []
        for col in feature_cols:
            if col in df.columns:
                missing_ratio = df[col].isnull().sum() / len(df)
                if missing_ratio < 0.3:  # 缺失值少於30%
                    available_features.append(col)
        
        return available_features
    
    def prepare_data(self, df, feature_cols, target_col):
        """資料預處理"""
        # 選擇需要的欄位
        data = df[feature_cols + [target_col]].copy()
        
        # 處理缺失值
        for col in feature_cols:
            if data[col].dtype in ['float64', 'int64']:
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 0)
        
        # 分離特徵和目標
        X = data[feature_cols].values
        y = data[target_col].values
        
        return X, y
    
    def evaluate_feature_combinations(self, train_df, test_df):
        """評估不同特徵組合"""
        all_features = self.select_features(train_df)
        
        # 定義不同的特徵組合
        feature_combinations = [
            # 基本財務指標
            ['股價淨值比', '股價營收比', 'M淨值報酬率─稅後', '資產報酬率ROA'],
            
            # 營運效率指標
            ['營業利益率OPM', '利潤邊際NPM', 'M流動比率', 'M速動比率'],
            
            # 成長性指標
            ['M營業利益成長率', 'M稅後淨利成長率', 'M存貨週轉率 (次)', 'M應收帳款週轉次'],
            
            # 綜合指標
            ['股價淨值比', '資產報酬率ROA', '營業利益率OPM', 'M淨值報酬率─稅後', 'M流動比率'],
            
            # 價值投資指標
            ['股價淨值比', '股價營收比', '市值(百萬元)', '收盤價(元)_年'],
            
            # 所有可用特徵
            all_features
        ]
        
        best_return = -float('inf')
        best_features = None
        best_model = None
        results = []
        
        for i, features in enumerate(feature_combinations):
            # 確保特徵存在於資料中
            available_features = [f for f in features if f in train_df.columns]
            
            if len(available_features) < 2:
                continue
                
            try:
                # 準備訓練資料
                X_train, y_train = self.prepare_data(
                    train_df, available_features, 'ReturnMean_year_Label'
                )
                
                # 訓練模型
                model = ID3DecisionTree(
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split
                )
                model.fit(X_train, y_train, available_features)
                
                # 準備測試資料
                X_test, y_test = self.prepare_data(
                    test_df, available_features, 'ReturnMean_year_Label'
                )
                
                # 預測
                predictions = model.predict(X_test)
                
                # 計算選中股票的平均報酬率
                positive_mask = predictions == 1
                if np.sum(positive_mask) > 0:
                    selected_returns = test_df.loc[
                        test_df.index[positive_mask], 'Return'
                    ].values
                    avg_return = np.mean(selected_returns)
                    num_selected = np.sum(positive_mask)
                else:
                    avg_return = 0
                    num_selected = 0
                
                # 計算準確率
                accuracy = np.mean(predictions == y_test)
                
                results.append({
                    'combination': i + 1,
                    'features': available_features,
                    'avg_return': avg_return,
                    'num_selected': num_selected,
                    'accuracy': accuracy,
                    'model': model
                })
                
                # 更新最佳結果
                if avg_return > best_return:
                    best_return = avg_return
                    best_features = available_features
                    best_model = model
                    
                print(f"組合 {i+1}: 平均報酬率 = {avg_return:.4f}%, 選中股票數 = {num_selected}, 準確率 = {accuracy:.4f}")
                
            except Exception as e:
                print(f"組合 {i+1} 發生錯誤: {e}")
                continue
        
        return best_model, best_features, best_return, results
    
    def run_time_series_validation(self, df):
        """時間序列交叉驗證"""
        # 獲取所有年份
        years = sorted(df['年月'].unique())
        
        results = []
        all_combination_results = []
        
        # 時間序列驗證
        for i in range(len(years) - 1):
            train_year = years[i]
            test_year = years[i + 1]
            
            print(f"\n=== 訓練年份: {train_year}, 測試年份: {test_year} ===")
            
            train_data = df[df['年月'] == train_year].copy()
            test_data = df[df['年月'] == test_year].copy()
            
            # 重設索引
            train_data = train_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)
            
            # 訓練並評估
            best_model, best_features, best_return, combination_results = self.evaluate_feature_combinations(
                train_data, test_data
            )
            
            results.append({
                'train_year': train_year,
                'test_year': test_year,
                'best_features': best_features,
                'avg_return': best_return,
                'model': best_model
            })
            
            all_combination_results.append({
                'train_year': train_year,
                'test_year': test_year,
                'combinations': combination_results
            })
            
            print(f"最佳平均報酬率: {best_return:.4f}%")
            print(f"最佳特徵組合: {best_features}")
            print("-" * 80)
        
        return results, all_combination_results
    
    def save_results_to_csv(self, results, all_combination_results, output_dir='results'):
        """將結果儲存為CSV檔案"""
        # 建立輸出目錄
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 1. 年度最佳結果
        yearly_results = []
        for result in results:
            yearly_results.append({
                '訓練年份': result['train_year'],
                '測試年份': result['test_year'],
                '最佳報酬率(%)': result['avg_return'],
                '最佳特徵組合': ', '.join(result['best_features']) if result['best_features'] else ''
            })
        
        yearly_df = pd.DataFrame(yearly_results)
        yearly_df.to_csv(f'{output_dir}/yearly_best_results.csv', index=False, encoding='utf-8-sig')
        
        # 2. 所有組合詳細結果
        all_results = []
        for year_result in all_combination_results:
            train_year = year_result['train_year']
            test_year = year_result['test_year']
            
            for combo in year_result['combinations']:
                all_results.append({
                    '訓練年份': train_year,
                    '測試年份': test_year,
                    '特徵組合編號': combo['combination'],
                    '特徵組合': ', '.join(combo['features']),
                    '平均報酬率(%)': combo['avg_return'],
                    '選中股票數': combo['num_selected'],
                    '準確率': combo['accuracy']
                })
        
        all_df = pd.DataFrame(all_results)
        all_df.to_csv(f'{output_dir}/all_combinations_results.csv', index=False, encoding='utf-8-sig')
        
        # 3. 特徵重要性統計
        feature_usage = {}
        for result in results:
            if result['best_features']:
                for feature in result['best_features']:
                    feature_usage[feature] = feature_usage.get(feature, 0) + 1
        
        feature_importance = []
        for feature, count in feature_usage.items():
            feature_importance.append({
                '特徵名稱': feature,
                '使用次數': count,
                '使用率(%)': count / len(results) * 100
            })
        
        feature_df = pd.DataFrame(feature_importance)
        feature_df = feature_df.sort_values('使用次數', ascending=False)
        feature_df.to_csv(f'{output_dir}/feature_importance.csv', index=False, encoding='utf-8-sig')
        
        print(f"\n結果已儲存至 {output_dir} 目錄：")
        print(f"- yearly_best_results.csv: 年度最佳結果")
        print(f"- all_combinations_results.csv: 所有組合詳細結果")
        print(f"- feature_importance.csv: 特徵重要性統計")
    
    def plot_results(self, results, all_combination_results, output_dir='results'):
        """繪製結果圖表"""
        # 建立輸出目錄
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 1. 年度報酬率趨勢圖
        plt.figure(figsize=(12, 6))
        years = [f"{r['train_year']}-{r['test_year']}" for r in results]
        returns = [r['avg_return'] for r in results]
        
        plt.plot(years, returns, marker='o', linewidth=2, markersize=8)
        plt.title('ID3決策樹股票選擇 - 年度報酬率趨勢', fontsize=16, fontweight='bold')
        plt.xlabel('年份區間', fontsize=12)
        plt.ylabel('平均報酬率 (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/yearly_returns_trend.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 特徵重要性圖
        feature_usage = {}
        for result in results:
            if result['best_features']:
                for feature in result['best_features']:
                    feature_usage[feature] = feature_usage.get(feature, 0) + 1
        
        if feature_usage:
            plt.figure(figsize=(12, 8))
            features = list(feature_usage.keys())
            counts = list(feature_usage.values())
            
            # 按使用次數排序
            sorted_data = sorted(zip(features, counts), key=lambda x: x[1], reverse=True)
            features, counts = zip(*sorted_data)
            
            plt.barh(features, counts, color='skyblue', edgecolor='navy', alpha=0.7)
            plt.title('特徵使用頻率統計', fontsize=16, fontweight='bold')
            plt.xlabel('使用次數', fontsize=12)
            plt.ylabel('特徵名稱', fontsize=12)
            
            # 在每個條形上顯示數值
            for i, count in enumerate(counts):
                plt.text(count + 0.1, i, str(count), va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. 報酬率分佈直方圖
        plt.figure(figsize=(10, 6))
        returns = [r['avg_return'] for r in results if r['avg_return'] is not None]
        
        plt.hist(returns, bins=10, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        plt.title('報酬率分佈直方圖', fontsize=16, fontweight='bold')
        plt.xlabel('平均報酬率 (%)', fontsize=12)
        plt.ylabel('頻率', fontsize=12)
        plt.axvline(np.mean(returns), color='red', linestyle='--', 
                   label=f'平均值: {np.mean(returns):.2f}%', linewidth=2)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/returns_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. 組合表現比較圖
        if all_combination_results:
            plt.figure(figsize=(15, 8))
            
            # 收集所有組合的平均表現
            combo_performance = {}
            for year_result in all_combination_results:
                for combo in year_result['combinations']:
                    combo_id = combo['combination']
                    if combo_id not in combo_performance:
                        combo_performance[combo_id] = []
                    combo_performance[combo_id].append(combo['avg_return'])
            
            # 計算每個組合的平均表現
            combo_means = {}
            combo_stds = {}
            for combo_id, returns in combo_performance.items():
                combo_means[combo_id] = np.mean(returns)
                combo_stds[combo_id] = np.std(returns)
            
            combo_ids = list(combo_means.keys())
            means = list(combo_means.values())
            stds = list(combo_stds.values())
            
            plt.bar(combo_ids, means, yerr=stds, capsize=5, 
                   color='lightcoral', edgecolor='darkred', alpha=0.7)
            plt.title('不同特徵組合的平均表現比較', fontsize=16, fontweight='bold')
            plt.xlabel('特徵組合編號', fontsize=12)
            plt.ylabel('平均報酬率 (%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/combination_performance.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 5. 統計摘要表
        self.create_summary_table(results, output_dir)
        
        print(f"\n圖表已儲存至 {output_dir} 目錄：")
        print(f"- yearly_returns_trend.png: 年度報酬率趨勢圖")
        print(f"- feature_importance.png: 特徵重要性圖")
        print(f"- returns_distribution.png: 報酬率分佈直方圖")
        print(f"- combination_performance.png: 組合表現比較圖")
        print(f"- summary_table.png: 統計摘要表")
    
    def create_summary_table(self, results, output_dir):
        """建立統計摘要表"""
        returns = [r['avg_return'] for r in results if r['avg_return'] is not None]
        
        if not returns:
            return
        
        # 計算統計數據
        stats = {
            '項目': ['平均報酬率', '最佳報酬率', '最差報酬率', '標準差', '正報酬率次數', '總測試次數'],
            '數值': [
                f"{np.mean(returns):.4f}%",
                f"{max(returns):.4f}%",
                f"{min(returns):.4f}%",
                f"{np.std(returns):.4f}%",
                f"{sum(1 for r in returns if r > 0)}",
                f"{len(returns)}"
            ]
        }
        
        # 建立表格圖
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=list(zip(stats['項目'], stats['數值'])),
                        colLabels=['統計項目', '數值'],
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # 設定表格樣式
        for i in range(len(stats['項目']) + 1):
            for j in range(2):
                if i == 0:  # 標題行
                    table[(i, j)].set_facecolor('#4CAF50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.title('ID3決策樹股票選擇 - 統計摘要', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(f'{output_dir}/summary_table.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_results(self, results):
        """分析結果"""
        print(f"\n{'='*60}")
        print(f"{'ID3決策樹股票選擇結果分析':^60}")
        print(f"{'='*60}")
        
        # 計算整體表現
        avg_returns = [r['avg_return'] for r in results if r['avg_return'] is not None]
        
        if avg_returns:
            overall_avg_return = np.mean(avg_returns)
            best_return = max(avg_returns)
            worst_return = min(avg_returns)
            std_return = np.std(avg_returns)
            
            print(f"平均報酬率: {overall_avg_return:.4f}%")
            print(f"最佳報酬率: {best_return:.4f}%")
            print(f"最差報酬率: {worst_return:.4f}%")
            print(f"報酬率標準差: {std_return:.4f}%")
            print(f"正報酬率次數: {sum(1 for r in avg_returns if r > 0)}/{len(avg_returns)}")
        
        # 特徵重要性分析
        feature_usage = {}
        for result in results:
            if result['best_features']:
                for feature in result['best_features']:
                    feature_usage[feature] = feature_usage.get(feature, 0) + 1
        
        print(f"\n{'特徵使用頻率分析':^60}")
        print("-" * 60)
        for feature, count in sorted(feature_usage.items(), 
                                    key=lambda x: x[1], reverse=True):
            usage_rate = count / len(results) * 100
            print(f"{feature:<25}: {count:>2}/{len(results)} ({usage_rate:>5.1f}%)")
        
        return {
            'overall_avg_return': overall_avg_return if avg_returns else 0,
            'best_return': best_return if avg_returns else 0,
            'worst_return': worst_return if avg_returns else 0,
            'std_return': std_return if avg_returns else 0,
            'feature_usage': feature_usage
        }


def main():
    """主程式"""
    print("載入資料...")
    
    # 讀取資料
    try:
        df = pd.read_excel('top200.xlsx')
        print(f"成功載入資料，共 {len(df)} 筆記錄")
    except FileNotFoundError:
        print("錯誤: 找不到 top200.xlsx 檔案")
        return
    except Exception as e:
        print(f"載入資料時發生錯誤: {e}")
        return
    
    # 過濾掉200912的資料
    original_count = len(df)
    df = df[df['年月'] != 200912]
    filtered_count = len(df)
    print(f"過濾後剩餘 {filtered_count} 筆記錄 (移除 {original_count - filtered_count} 筆)")
    
    # 檢查必要欄位
    required_columns = ['年月', 'Return', 'ReturnMean_year_Label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"錯誤: 缺少必要欄位: {missing_columns}")
        return
    
    # 初始化股票選擇器
    print("\n初始化ID3決策樹股票選擇器...")
    stock_selector = StockSelectionID3(max_depth=6, min_samples_split=10)
    
    # 執行時間序列驗證
    print("開始執行時間序列交叉驗證...")
    results, combination_results = stock_selector.run_time_series_validation(df)
    
    # 分析結果
    analysis = stock_selector.analyze_results(results)
    
    # 儲存CSV結果
    print("\n儲存結果到CSV檔案...")
    stock_selector.save_results_to_csv(results, combination_results)
    
    # 繪製圖表
    print("\n繪製結果圖表...")
    stock_selector.plot_results(results, combination_results)
    
    # 顯示詳細結果
    print(f"\n{'年度詳細結果':^60}")
    print("-" * 60)
    for result in results:
        print(f"{result['train_year']} → {result['test_year']}: {result['avg_return']:>8.4f}%")
    
    return results, combination_results, analysis


if __name__ == "__main__":
    # 執行主程式
    results, combination_results, analysis = main()
    
    print(f"\n{'='*60}")
    print("程式執行完成！")
    print("結果檔案已儲存至 'results' 目錄")
    print("包含CSV檔案和PNG圖表")
    print(f"{'='*60}")
