import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import csv
warnings.filterwarnings('ignore')

class KNNStockSelector:
    def __init__(self, data_path='top200.xlsx'):
        self.data = self.load_data(data_path)
        self.feature_columns = [
            '市值(百萬元)', '收盤價(元)_年', 'Unknown masked parameter',
            '股價淨值比', '股價營收比', 'M淨值報酬率─稅後', '資產報酬率ROA',
            '營業利益率OPM', '利潤邊際NPM', '負債/淨值比', 'M流動比率',
            'M速動比率', 'M存貨週轉率 (次)', 'M應收帳款週轉次',
            'M營業利益成長率', 'M稅後淨利成長率'
        ]
        
    def load_data(self, data_path):
        df = pd.read_excel(data_path)
        df = df[df['年月'] != 200912]
        df = df.fillna(df.median(numeric_only=True))
        return df
    
    def get_year_data(self, year):
        year_str = f"{year}12"
        return self.data[self.data['年月'] == int(year_str)]
    
    def prepare_features(self, data, feature_subset=None):
        if feature_subset is None:
            feature_subset = self.feature_columns
        X = data[feature_subset].values
        y = data['ReturnMean_year_Label'].values
        returns = data['Return'].values
        stock_names = data['簡稱'].values
        return X, y, returns, stock_names
    
    def generate_feature_combinations(self):
        feature_combinations = []
        feature_combinations.append(self.feature_columns)
        important_features = [
            '市值(百萬元)', '股價淨值比', '股價營收比', 'M淨值報酬率─稅後',
            '資產報酬率ROA', '營業利益率OPM', '利潤邊際NPM', '負債/淨值比'
        ]
        feature_combinations.append(important_features)
        financial_ratios = [
            '股價淨值比', '股價營收比', 'M淨值報酬率─稅後', '資產報酬率ROA',
            '營業利益率OPM', '利潤邊際NPM'
        ]
        feature_combinations.append(financial_ratios)
        growth_features = [
            'M營業利益成長率', 'M稅後淨利成長率', '資產報酬率ROA', 'M淨值報酬率─稅後'
        ]
        feature_combinations.append(growth_features)
        liquidity_features = [
            'M流動比率', 'M速動比率', 'M存貨週轉率 (次)', 'M應收帳款週轉次'
        ]
        feature_combinations.append(liquidity_features)
        return feature_combinations
    
    def train_and_evaluate_knn(self, X_train, y_train, train_returns, k, features):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_train_scaled)
        positive_indices = np.where(y_pred == 1)[0]
        if len(positive_indices) == 0:
            return -np.inf, knn, scaler, 0
        selected_returns = train_returns[positive_indices]
        avg_return = np.mean(selected_returns)
        return avg_return, knn, scaler, len(positive_indices)
    
    def find_best_parameters(self, train_data, k_range=range(3, 21, 2)):
        feature_combinations = self.generate_feature_combinations()
        best_return = -np.inf
        best_params = None
        best_model = None
        best_scaler = None
        for i, features in enumerate(feature_combinations):
            try:
                X_train, y_train, train_returns, _ = self.prepare_features(train_data, features)
                for k in k_range:
                    avg_return, knn, scaler, n_selected = self.train_and_evaluate_knn(
                        X_train, y_train, train_returns, k, features
                    )
                    if avg_return > best_return:
                        best_return = avg_return
                        best_params = {
                            'k': k,
                            'features': features,
                            'feature_names': features,
                            'avg_return': avg_return,
                            'n_selected': n_selected
                        }
                        best_model = knn
                        best_scaler = scaler
            except Exception as e:
                continue
        return best_params, best_model, best_scaler

    def select_stocks_with_model(self, model, scaler, test_data, features):
        X_test, _, test_returns, stock_names = self.prepare_features(test_data, features)
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        pred_proba = model.predict_proba(X_test_scaled)
        positive_indices = np.where(y_pred == 1)[0]
        if len(positive_indices) == 0:
            # 若沒有正類，選機率最高的5檔
            positive_proba = pred_proba[:, 1]
            top_indices = np.argsort(positive_proba)[-5:]
            selected_indices = top_indices
        else:
            selected_indices = positive_indices
        selected_returns = test_returns[selected_indices]
        selected_stocks = stock_names[selected_indices]
        return selected_returns, selected_stocks
    
    def rolling_window_backtest(self, start_year=1997, end_year=2008):
        results = []
        print("開始KNN股票選擇回測...")
        print("=" * 60)
        for test_year in range(start_year + 1, end_year + 1):
            train_year = test_year - 1
            print(f"\n【第 {test_year - start_year} 年】訓練年份: {train_year}, 測試年份: {test_year}")
            train_data = self.get_year_data(train_year)
            test_data = self.get_year_data(test_year)
            if len(train_data) == 0 or len(test_data) == 0:
                print(f"  警告: {train_year} 或 {test_year} 年度無資料")
                continue
            best_params, best_model, best_scaler = self.find_best_parameters(train_data)
            if best_params is None:
                print(f"  錯誤: 無法找到有效參數")
                continue
            print(f"  最佳參數: K={best_params['k']}, 特徵數={len(best_params['features'])}")
            print(f"  訓練集平均報酬率: {best_params['avg_return']:.4f}%")
            print(f"  訓練集選中股票數: {best_params['n_selected']}")
            selected_returns, selected_stocks = self.select_stocks_with_model(
                best_model, best_scaler, test_data, best_params['features']
            )
            test_avg_return = np.mean(selected_returns)
            test_std_return = np.std(selected_returns)
            result = {
                'test_year': test_year,
                'train_year': train_year,
                'best_k': best_params['k'],
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

    def save_results_to_csv(self, results, filename='knn_stock_selection_results.csv'):
        if not results:
            print("無結果可儲存")
            return
        csv_data = []
        for result in results:
            csv_data.append({
                'test_year': result['test_year'],
                'train_year': result['train_year'],
                'best_k': result['best_k'],
                'n_features': result['n_features'],
                'train_return': result['train_return'],
                'train_n_selected': result['train_n_selected'],
                'test_return': result['test_return'],
                'test_std': result['test_std'],
                'test_n_selected': result['test_n_selected']
            })
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['test_year', 'train_year', 'best_k', 'n_features', 
                         'train_return', 'train_n_selected', 'test_return', 
                         'test_std', 'test_n_selected']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        print(f"主要結果已儲存至: {filename}")
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

def main():
    print("KNN股票選擇系統")
    print("=" * 60)
    try:
        selector = KNNStockSelector('top200.xlsx')
        print(f"成功載入資料: {len(selector.data)} 筆記錄")
        print(f"資料年份範圍: {selector.data['年月'].min()} - {selector.data['年月'].max()}")
        print(f"特徵數量: {len(selector.feature_columns)}")
    except Exception as e:
        print(f"載入資料失敗: {e}")
        return None
    try:
        results = selector.rolling_window_backtest()
        if not results:
            print("回測失敗，無結果產生")
            return None
        selector.save_results_to_csv(results)
        return results
    except Exception as e:
        print(f"回測過程發生錯誤: {e}")
        return None

if __name__ == "__main__":
    main()
