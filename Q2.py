import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from itertools import combinations
import warnings
import csv
import math
warnings.filterwarnings('ignore')

class ID3StockSelector:
    def __init__(self, data_path='top200.xlsx'):
        """
        初始化ID3股票選擇器
        """
        self.data = self.load_data(data_path)
        self.feature_columns = [
            '市值(百萬元)', '收盤價(元)_年', 'Unknown masked parameter',
            '股價淨值比', '股價營收比', 'M淨值報酬率─稅後', '資產報酬率ROA',
            '營業利益率OPM', '利潤邊際NPM', '負債/淨值比', 'M流動比率',
            'M速動比率', 'M存貨週轉率 (次)', 'M應收帳款週轉次',
            'M營業利益成長率', 'M稅後淨利成長率'
        ]
        print(f"特徵數量: {len(self.feature_columns)}")
        print(f"特徵索引範圍: 0-{len(self.feature_columns)-1}")
        
    def load_data(self, data_path):
        """
        載入股票資料並預處理
        """
        df = pd.read_excel(data_path)
        # 移除200912的資料
        df = df[df['年月'] != 200912]
        # 處理缺失值
        df = df.fillna(df.median(numeric_only=True))
        return df
    
    def get_year_data(self, year):
        """
        取得特定年份的資料
        """
        year_str = f"{year}12"
        return self.data[self.data['年月'] == int(year_str)]
    
    def prepare_features(self, data, feature_subset=None):
        """
        準備特徵資料
        """
        if feature_subset is None:
            feature_subset = self.feature_columns
        
        X = data[feature_subset].values
        y = data['ReturnMean_year_Label'].values
        returns = data['Return'].values
        stock_names = data['簡稱'].values
        
        return X, y, returns, stock_names
    
    def generate_feature_combinations(self):
        """
        生成不同的特徵組合
        """
        feature_combinations = []
        
        # 1. 所有特徵的索引
        all_features = list(range(len(self.feature_columns)))
        feature_combinations.append(all_features)
        
        # 2. 重要財務指標 - 修正索引範圍
        important_indices = [0, 3, 4, 5, 6, 7]  # 確保在有效範圍內
        feature_combinations.append(important_indices)
        
        # 3. 財務比率特徵
        ratio_indices = [3, 4, 5, 6, 7]  # 確保索引在有效範圍內
        feature_combinations.append(ratio_indices)
        
        # 4. 成長性指標
        growth_indices = [14, 15, 5, 6]  # 檢查索引是否有效
        # 過濾掉超出範圍的索引
        growth_indices = [i for i in growth_indices if i < len(self.feature_columns)]
        if growth_indices:  # 確保列表不為空
            feature_combinations.append(growth_indices)
        
        # 5. 流動性指標
        liquidity_indices = [10, 11, 12, 13]
        # 過濾掉超出範圍的索引
        liquidity_indices = [i for i in liquidity_indices if i < len(self.feature_columns)]
        if liquidity_indices:  # 確保列表不為空
            feature_combinations.append(liquidity_indices)
        
        return feature_combinations
    
    def train_and_evaluate_id3(self, X_train, y_train, train_returns, features, 
                              max_depth=10, min_samples_split=5, min_samples_leaf=2):
        """
        訓練ID3模型並評估在訓練集上的表現
        """
        # 使用scikit-learn的決策樹實現ID3算法
        dt = DecisionTreeClassifier(
            criterion='entropy',  # 使用信息熵，類似ID3
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        # 訓練模型
        dt.fit(X_train, y_train)
        
        # 在訓練集上預測
        y_pred = dt.predict(X_train)
        
        # 選擇預測為正類(1)的股票
        positive_indices = np.where(y_pred == 1)[0]
        
        if len(positive_indices) == 0:
            return -np.inf, dt, 0
        
        # 計算這些股票的平均報酬率
        selected_returns = train_returns[positive_indices]
        avg_return = np.mean(selected_returns)
        
        return avg_return, dt, len(positive_indices)
    
    def find_best_parameters(self, train_data, 
                           max_depth_range=[5, 8, 10, 15], 
                           min_samples_split_range=[3, 5, 10],
                           min_samples_leaf_range=[1, 2, 5]):
        """
        尋找最佳的決策樹參數和特徵組合
        """
        feature_combinations = self.generate_feature_combinations()
        
        best_return = -np.inf
        best_params = None
        best_model = None
        
        print(f"  正在測試 {len(feature_combinations)} 種特徵組合和多種參數...")
        
        for i, features in enumerate(feature_combinations):
            print(f"    特徵組合 {i+1}/{len(feature_combinations)}: {len(features)} 個特徵")
            
            # 驗證特徵索引是否有效
            valid_features = [f for f in features if f < len(self.feature_columns)]
            if len(valid_features) != len(features):
                print(f"    警告: 特徵組合包含無效索引，已過濾")
                features = valid_features
            
            if len(features) == 0:
                print(f"    跳過: 沒有有效特徵")
                continue
                
            try:
                X_train, y_train, train_returns, _ = self.prepare_features(
                    train_data, [self.feature_columns[j] for j in features]
                )
                
                for max_depth in max_depth_range:
                    for min_samples_split in min_samples_split_range:
                        for min_samples_leaf in min_samples_leaf_range:
                            avg_return, dt, n_selected = self.train_and_evaluate_id3(
                                X_train, y_train, train_returns, features,
                                max_depth, min_samples_split, min_samples_leaf
                            )
                            
                            # 更新最佳參數
                            if avg_return > best_return:
                                best_return = avg_return
                                best_params = {
                                    'max_depth': max_depth,
                                    'min_samples_split': min_samples_split,
                                    'min_samples_leaf': min_samples_leaf,
                                    'features': features,
                                    'feature_names': [self.feature_columns[j] for j in features],
                                    'avg_return': avg_return,
                                    'n_selected': n_selected
                                }
                                best_model = dt
            except Exception as e:
                print(f"    特徵組合 {i+1} 處理失敗: {e}")
                continue
        
        return best_params, best_model
    
    def select_stocks_with_tree(self, model, test_data, features, top_n_stocks=10):
        """
        使用已訓練的決策樹選擇股票
        """
        X_test, _, test_returns, stock_names = self.prepare_features(
            test_data, [self.feature_columns[j] for j in features]
        )
        
        # 預測測試資料
        y_pred = model.predict(X_test)
        
        # 如果模型支持預測機率，使用機率排序
        try:
            pred_proba = model.predict_proba(X_test)
            has_proba = True
        except:
            has_proba = False
        
        # 選擇預測為正類的股票
        positive_indices = np.where(y_pred == 1)[0]
        
        if len(positive_indices) == 0:
            # 如果沒有預測為正類的股票，選擇報酬率最高的股票
            if has_proba:
                positive_proba = pred_proba[:, 1]
                top_indices = np.argsort(positive_proba)[-top_n_stocks:]
            else:
                top_indices = np.argsort(test_returns)[-top_n_stocks:]
            selected_indices = top_indices
        else:
            # 如果選中的股票太多，選擇最有潛力的前N檔
            if len(positive_indices) > top_n_stocks:
                if has_proba:
                    positive_proba = pred_proba[positive_indices, 1]
                    top_relative_indices = np.argsort(positive_proba)[-top_n_stocks:]
                    selected_indices = positive_indices[top_relative_indices]
                else:
                    positive_returns = test_returns[positive_indices]
                    top_relative_indices = np.argsort(positive_returns)[-top_n_stocks:]
                    selected_indices = positive_indices[top_relative_indices]
            else:
                selected_indices = positive_indices
        
        selected_returns = test_returns[selected_indices]
        selected_stocks = stock_names[selected_indices]
        
        return selected_returns, selected_stocks
    
    def rolling_window_backtest(self, start_year=1997, end_year=2008):
        """
        執行rolling window回測
        """
        results = []
        
        print("開始ID3決策樹股票選擇回測...")
        print("=" * 60)
        
        for test_year in range(start_year + 1, end_year + 1):
            train_year = test_year - 1
            
            print(f"\n【第 {test_year - start_year} 年】訓練年份: {train_year}, 測試年份: {test_year}")
            
            # 取得訓練和測試資料
            train_data = self.get_year_data(train_year)
            test_data = self.get_year_data(test_year)
            
            if len(train_data) == 0 or len(test_data) == 0:
                print(f"  警告: {train_year} 或 {test_year} 年度無資料")
                continue
            
            print(f"  訓練資料: {len(train_data)} 檔股票")
            print(f"  測試資料: {len(test_data)} 檔股票")
            
            # 尋找最佳參數並訓練模型
            best_params, best_model = self.find_best_parameters(train_data)
            
            if best_params is None:
                print(f"  錯誤: 無法找到有效參數")
                continue
            
            print(f"  最佳參數: 深度={best_params['max_depth']}, "
                  f"最小分割={best_params['min_samples_split']}, "
                  f"最小葉節點={best_params['min_samples_leaf']}")
            print(f"  特徵數={len(best_params['features'])}")
            print(f"  訓練集平均報酬率: {best_params['avg_return']:.4f}%")
            print(f"  訓練集選中股票數: {best_params['n_selected']}")
            
            # 使用最佳參數選擇測試年度的股票
            selected_returns, selected_stocks = self.select_stocks_with_tree(
                best_model, test_data, best_params['features']
            )
            
            # 計算測試結果
            test_avg_return = np.mean(selected_returns)
            test_std_return = np.std(selected_returns)
            
            result = {
                'test_year': test_year,
                'train_year': train_year,
                'max_depth': best_params['max_depth'],
                'min_samples_split': best_params['min_samples_split'],
                'min_samples_leaf': best_params['min_samples_leaf'],
                'n_features': len(best_params['features']),
                'feature_names': best_params['feature_names'],
                'train_return': best_params['avg_return'],
                'train_n_selected': best_params['n_selected'],
                'test_return': test_avg_return,
                'test_std': test_std_return,
                'test_n_selected': len(selected_returns),
                'selected_returns': selected_returns,
                'selected_stocks': selected_stocks
            }
            
            results.append(result)
            
            print(f"  測試集平均報酬率: {test_avg_return:.4f}% ± {test_std_return:.4f}%")
            print(f"  測試集選中股票數: {len(selected_returns)}")
            print(f"  選中股票: {', '.join(selected_stocks[:5])}{'...' if len(selected_stocks) > 5 else ''}")
            print("-" * 60)
        
        return results
    
    def save_results_to_csv(self, results, filename='id3_stock_selection_results.csv'):
        """
        將結果儲存為CSV檔案
        """
        if not results:
            print("無結果可儲存")
            return
        
        # 準備主要結果CSV資料
        csv_data = []
        for result in results:
            csv_data.append({
                'test_year': result['test_year'],
                'train_year': result['train_year'],
                'max_depth': result['max_depth'],
                'min_samples_split': result['min_samples_split'],
                'min_samples_leaf': result['min_samples_leaf'],
                'n_features': result['n_features'],
                'train_return': result['train_return'],
                'train_n_selected': result['train_n_selected'],
                'test_return': result['test_return'],
                'test_std': result['test_std'],
                'test_n_selected': result['test_n_selected']
            })
        
        # 寫入主要結果CSV檔案
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['test_year', 'train_year', 'max_depth', 'min_samples_split', 
                         'min_samples_leaf', 'n_features', 'train_return', 'train_n_selected', 
                         'test_return', 'test_std', 'test_n_selected']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        
        print(f"主要結果已儲存至: {filename}")
        
        # 儲存詳細的選股結果
        detail_filename = filename.replace('.csv', '_selected_stocks.csv')
        with open(detail_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['test_year', 'stock_name', 'return']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                for stock, ret in zip(result['selected_stocks'], result['selected_returns']):
                    writer.writerow({
                        'test_year': result['test_year'],
                        'stock_name': stock,
                        'return': ret
                    })
        
        print(f"選股詳細結果已儲存至: {detail_filename}")
        
        # 儲存特徵使用記錄
        feature_filename = filename.replace('.csv', '_features.csv')
        with open(feature_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['test_year', 'feature_name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                for feature in result['feature_names']:
                    writer.writerow({
                        'test_year': result['test_year'],
                        'feature_name': feature
                    })
        
        print(f"特徵使用記錄已儲存至: {feature_filename}")
    
    def analyze_results(self, results):
        """
        分析回測結果
        """
        if not results:
            print("無結果可分析")
            return None
        
        # 計算總體統計
        test_returns = [r['test_return'] for r in results]
        train_returns = [r['train_return'] for r in results]
        
        print("\n" + "=" * 60)
        print("ID3決策樹股票選擇結果分析")
        print("=" * 60)
        print(f"測試期間: {results[0]['test_year']} - {results[-1]['test_year']}")
        print(f"回測次數: {len(results)}")
        print()
        print("【測試集績效】")
        print(f"平均報酬率: {np.mean(test_returns):.4f}%")
        print(f"報酬率標準差: {np.std(test_returns):.4f}%")
        print(f"最佳報酬率: {np.max(test_returns):.4f}%")
        print(f"最差報酬率: {np.min(test_returns):.4f}%")
        print(f"勝率: {sum(1 for r in test_returns if r > 0) / len(test_returns) * 100:.2f}%")
        
        if np.std(test_returns) != 0:
            sharpe_ratio = np.mean(test_returns) / np.std(test_returns)
            print(f"夏普比率: {sharpe_ratio:.4f}")
        
        print()
        print("【訓練集績效】")
        print(f"平均報酬率: {np.mean(train_returns):.4f}%")
        print(f"報酬率標準差: {np.std(train_returns):.4f}%")
        
        # 年度詳細結果
        print()
        print("【年度詳細結果】")
        print("年份   深度  分割  葉節點  特徵數  訓練報酬率  測試報酬率  選股數量")
        print("-" * 70)
        for result in results:
            print(f"{result['test_year']}    {result['max_depth']:2d}    {result['min_samples_split']:2d}     "
                  f"{result['min_samples_leaf']:2d}      {result['n_features']:2d}     "
                  f"{result['train_return']:8.2f}%   {result['test_return']:8.2f}%     {result['test_n_selected']:2d}")
        
        return {
            'avg_test_return': np.mean(test_returns),
            'std_test_return': np.std(test_returns),
            'avg_train_return': np.mean(train_returns),
            'best_test_return': np.max(test_returns),
            'worst_test_return': np.min(test_returns),
            'win_rate': sum(1 for r in test_returns if r > 0) / len(test_returns),
            'sharpe_ratio': np.mean(test_returns) / np.std(test_returns) if np.std(test_returns) != 0 else 0,
            'results': results
        }

def main():
    """
    主函數
    """
    print("ID3決策樹股票選擇系統")
    print("=" * 60)
    
    # 初始化選股器
    try:
        selector = ID3StockSelector('top200.xlsx')
        print(f"成功載入資料: {len(selector.data)} 筆記錄")
        print(f"資料年份範圍: {selector.data['年月'].min()} - {selector.data['年月'].max()}")
        print(f"特徵數量: {len(selector.feature_columns)}")
    except Exception as e:
        print(f"載入資料失敗: {e}")
        return None
    
    # 執行回測
    try:
        results = selector.rolling_window_backtest()
        
        if not results:
            print("回測失敗，無結果產生")
            return None
        
        # 分析結果
        analysis = selector.analyze_results(results)
        
        # 儲存結果到CSV
        selector.save_results_to_csv(results)
        
        return analysis
        
    except Exception as e:
        print(f"回測過程發生錯誤: {e}")
        return None

if __name__ == "__main__":
    analysis = main()
