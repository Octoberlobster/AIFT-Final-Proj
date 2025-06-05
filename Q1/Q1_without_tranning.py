import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from itertools import combinations
import warnings
import csv
warnings.filterwarnings('ignore')

class KNNStockSelector:
    def __init__(self, data_path='top200.xlsx'):
        """
        初始化KNN股票選擇器
        """
        self.data = self.load_data(data_path)
        self.feature_columns = [
            '市值(百萬元)', '收盤價(元)_年', 'Unknown masked parameter',
            '股價淨值比', '股價營收比', 'M淨值報酬率─稅後', '資產報酬率ROA',
            '營業利益率OPM', '利潤邊際NPM', '負債/淨值比', 'M流動比率',
            'M速動比率', 'M存貨週轉率 (次)', 'M應收帳款週轉次',
            'M營業利益成長率', 'M稅後淨利成長率'
        ]
        
    def load_data(self, data_path):
        """
        載入股票資料
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
        
        return X, y, returns
    
    def train_knn_model(self, X_train, y_train, k=3):
        """
        訓練KNN模型
        """
        # 標準化特徵
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 建立並訓練KNN模型
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn.fit(X_train_scaled, y_train)
        
        return knn, scaler
    
    def evaluate_model_performance(self, X_train, y_train, train_returns, k, features):
        """
        評估模型在訓練集上的表現
        """
        # 訓練模型
        knn, scaler = self.train_knn_model(X_train, y_train, k)
        
        # 在訓練集上預測
        X_train_scaled = scaler.transform(X_train)
        y_pred = knn.predict(X_train_scaled)
        
        # 選擇預測為正類(1)的股票
        positive_indices = np.where(y_pred == 1)[0]
        
        if len(positive_indices) == 0:
            return -np.inf, knn, scaler
        
        # 計算這些股票的平均報酬率
        selected_returns = train_returns[positive_indices]
        avg_return = np.mean(selected_returns)
        
        return avg_return, knn, scaler
    
    def find_best_params(self, train_data, k_range=range(3, 21, 2)):
        """
        尋找最佳的K值和特徵組合
        """
        # 定義不同的特徵組合
        feature_combinations = [
            # 所有特徵
            self.feature_columns,
            # 重要財務指標
            ['市值(百萬元)', '股價淨值比', '股價營收比', 'M淨值報酬率─稅後',
             '資產報酬率ROA', '營業利益率OPM', '利潤邊際NPM', '負債/淨值比'],
            # 財務比率特徵
            ['股價淨值比', '股價營收比', 'M淨值報酬率─稅後', '資產報酬率ROA',
             '營業利益率OPM', '利潤邊際NPM'],
            # 成長性指標
            ['M營業利益成長率', 'M稅後淨利成長率', '資產報酬率ROA', 'M淨值報酬率─稅後'],
            # 流動性指標
            ['M流動比率', 'M速動比率', 'M存貨週轉率 (次)', 'M應收帳款週轉次']
        ]
        
        best_return = -np.inf
        best_params = None
        best_model = None
        best_scaler = None
        
        for features in feature_combinations:
            X_train, y_train, train_returns = self.prepare_features(train_data, features)
            
            for k in k_range:
                avg_return, knn, scaler = self.evaluate_model_performance(
                    X_train, y_train, train_returns, k, features
                )
                
                # 更新最佳參數
                if avg_return > best_return:
                    best_return = avg_return
                    best_params = {
                        'k': k,
                        'features': features,
                        'avg_return': avg_return
                    }
                    best_model = knn
                    best_scaler = scaler
        
        return best_params, best_model, best_scaler
    
    def select_stocks_with_trained_model(self, model, scaler, test_data, features, top_n_stocks=10):
        """
        使用已訓練的模型選擇股票
        """
        X_test, _, test_returns = self.prepare_features(test_data, features)
        X_test_scaled = scaler.transform(X_test)
        
        # 預測測試資料
        y_pred = model.predict(X_test_scaled)
        pred_proba = model.predict_proba(X_test_scaled)
        
        # 選擇預測為正類的股票
        positive_indices = np.where(y_pred == 1)[0]
        
        if len(positive_indices) == 0:
            # 如果沒有預測為正類的股票，選擇正類機率最高的股票
            positive_proba = pred_proba[:, 1]
            top_indices = np.argsort(positive_proba)[-top_n_stocks:]
            selected_returns = test_returns[top_indices]
        else:
            # 如果選中的股票太多，根據正類機率選擇前N檔
            if len(positive_indices) > top_n_stocks:
                positive_proba = pred_proba[positive_indices, 1]
                top_relative_indices = np.argsort(positive_proba)[-top_n_stocks:]
                selected_indices = positive_indices[top_relative_indices]
            else:
                selected_indices = positive_indices
            
            selected_returns = test_returns[selected_indices]
        
        return selected_returns
    
    def rolling_window_backtest(self, start_year=1997, end_year=2008):
        """
        執行rolling window回測
        """
        results = []
        
        for test_year in range(start_year + 1, end_year + 1):
            train_year = test_year - 1
            
            print(f"Training on {train_year}, Testing on {test_year}")
            
            # 取得訓練和測試資料
            train_data = self.get_year_data(train_year)
            test_data = self.get_year_data(test_year)
            
            if len(train_data) == 0 or len(test_data) == 0:
                print(f"No data for year {train_year} or {test_year}")
                continue
            
            # 尋找最佳參數並訓練模型
            best_params, best_model, best_scaler = self.find_best_params(train_data)
            
            if best_params is None:
                print(f"No valid parameters found for year {train_year}")
                continue
            
            print(f"Best params: K={best_params['k']}, Features={len(best_params['features'])}, "
                  f"Train Return={best_params['avg_return']:.4f}%")
            
            # 使用訓練好的模型選擇測試年度的股票
            selected_returns = self.select_stocks_with_trained_model(
                best_model, best_scaler, test_data, best_params['features']
            )
            
            # 計算測試結果
            test_avg_return = np.mean(selected_returns)
            test_std_return = np.std(selected_returns)
            
            result = {
                'test_year': test_year,
                'train_year': train_year,
                'best_k': best_params['k'],
                'n_features': len(best_params['features']),
                'train_return': best_params['avg_return'],
                'test_return': test_avg_return,
                'test_std': test_std_return,
                'n_selected': len(selected_returns),
                'selected_returns': selected_returns,
                'feature_names': best_params['features']
            }
            
            results.append(result)
            
            print(f"Test Return: {test_avg_return:.4f}% ± {test_std_return:.4f}%")
            print("-" * 50)
        
        return results
    
    def save_results_to_csv(self, results, filename='knn_stock_selection_results.csv'):
        """
        將結果儲存為CSV檔案
        """
        # 準備CSV資料
        csv_data = []
        for result in results:
            csv_data.append({
                'test_year': result['test_year'],
                'train_year': result['train_year'],
                'best_k': result['best_k'],
                'n_features': result['n_features'],
                'train_return': result['train_return'],
                'test_return': result['test_return'],
                'test_std': result['test_std'],
                'n_selected': result['n_selected']
            })
        
        # 寫入CSV檔案
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['test_year', 'train_year', 'best_k', 'n_features', 
                         'train_return', 'test_return', 'test_std', 'n_selected']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        
        print(f"Results saved to {filename}")
        
        # 也儲存詳細的特徵使用資訊
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
        
        print(f"Feature usage saved to {feature_filename}")
    
    def analyze_results(self, results):
        """
        分析回測結果
        """
        if not results:
            print("No results to analyze")
            return
        
        # 計算總體統計
        test_returns = [r['test_return'] for r in results]
        train_returns = [r['train_return'] for r in results]
        
        print("=== KNN Stock Selection Results ===")
        print(f"Average Test Return: {np.mean(test_returns):.4f}%")
        print(f"Test Return Std: {np.std(test_returns):.4f}%")
        print(f"Average Train Return: {np.mean(train_returns):.4f}%")
        print(f"Best Test Return: {np.max(test_returns):.4f}%")
        print(f"Worst Test Return: {np.min(test_returns):.4f}%")
        print(f"Sharpe Ratio: {np.mean(test_returns)/np.std(test_returns):.4f}")
        
        # 詳細結果
        print("\n=== Yearly Results ===")
        for result in results:
            print(f"{result['test_year']}: {result['test_return']:.4f}% "
                  f"(K={result['best_k']}, Features={result['n_features']}, "
                  f"Selected={result['n_selected']})")
        
        # 特徵使用頻率分析
        print("\n=== Feature Usage Analysis ===")
        feature_usage = {}
        for result in results:
            for feature in result['feature_names']:
                feature_usage[feature] = feature_usage.get(feature, 0) + 1
        
        sorted_features = sorted(feature_usage.items(), key=lambda x: x[1], reverse=True)
        for feature, count in sorted_features:
            print(f"{feature}: {count}/{len(results)} times")
        
        return {
            'avg_test_return': np.mean(test_returns),
            'std_test_return': np.std(test_returns),
            'avg_train_return': np.mean(train_returns),
            'best_test_return': np.max(test_returns),
            'worst_test_return': np.min(test_returns),
            'sharpe_ratio': np.mean(test_returns)/np.std(test_returns),
            'results': results,
            'feature_usage': feature_usage
        }

def main():
    """
    主函數
    """
    print("Starting KNN Stock Selection with Model Training...")
    
    # 初始化選股器
    selector = KNNStockSelector('top200.xlsx')
    
    # 執行回測
    results = selector.rolling_window_backtest()
    
    # 分析結果
    analysis = selector.analyze_results(results)
    
    # 儲存結果到CSV
    selector.save_results_to_csv(results)
    
    return analysis

if __name__ == "__main__":
    analysis = main()
