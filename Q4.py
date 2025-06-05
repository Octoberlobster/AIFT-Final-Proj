import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import warnings
import csv
import random
warnings.filterwarnings('ignore')

class HybridSVRStockSelector:
    def __init__(self, data_path='top200.xlsx'):
        """
        初始化混合SVR股票選擇器
        """
        self.data = self.load_data(data_path)
        self.feature_columns = [
            '市值(百萬元)', '收盤價(元)_年', 'Unknown masked parameter',
            '股價淨值比', '股價營收比', 'M淨值報酬率─稅後', '資產報酬率ROA',
            '營業利益率OPM', '利潤邊際NPM', '負債/淨值比', 'M流動比率',
            'M速動比率', 'M存貨週轉率 (次)', 'M應收帳款週轉次',
            'M營業利益成長率', 'M稅後淨利成長率'
        ]
        
        # 遺傳演算法參數
        self.population_size = 50
        self.generations = 30
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1
        
        # SVR參數範圍
        self.c_range = (0.1, 1000)
        self.gamma_range = (0.001, 1)
        self.epsilon_range = (0.01, 1)
        
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
        y = data['Return'].values  # 使用實際報酬率作為目標變數
        stock_names = data['簡稱'].values
        
        return X, y, stock_names
    
    def decode_chromosome(self, chromosome):
        """
        解碼染色體為SVR參數和特徵選擇
        """
        # 前16位用於特徵選擇
        feature_selection = chromosome[:16]
        
        # 接下來8位用於C參數
        c_bits = chromosome[16:24]
        c_decimal = int(''.join(map(str, c_bits)), 2)
        c_value = self.c_range[0] + (c_decimal / 255) * (self.c_range[1] - self.c_range[0])
        
        # 接下來8位用於gamma參數
        gamma_bits = chromosome[24:32]
        gamma_decimal = int(''.join(map(str, gamma_bits)), 2)
        gamma_value = self.gamma_range[0] + (gamma_decimal / 255) * (self.gamma_range[1] - self.gamma_range[0])
        
        # 最後8位用於epsilon參數
        epsilon_bits = chromosome[32:40]
        epsilon_decimal = int(''.join(map(str, epsilon_bits)), 2)
        epsilon_value = self.epsilon_range[0] + (epsilon_decimal / 255) * (self.epsilon_range[1] - self.epsilon_range[0])
        
        # 選擇的特徵
        selected_features = [self.feature_columns[i] for i, bit in enumerate(feature_selection) if bit == 1]
        
        if len(selected_features) == 0:
            selected_features = [self.feature_columns[0]]  # 至少選擇一個特徵
        
        return {
            'features': selected_features,
            'C': c_value,
            'gamma': gamma_value,
            'epsilon': epsilon_value
        }
    
    def evaluate_chromosome(self, chromosome, train_data):
        """
        評估染色體的適應度
        """
        params = self.decode_chromosome(chromosome)
        
        try:
            # 準備訓練資料
            X_train, y_train, _ = self.prepare_features(train_data, params['features'])
            
            # 標準化特徵
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # 建立SVR模型
            svr = SVR(
                kernel='rbf',
                C=params['C'],
                gamma=params['gamma'],
                epsilon=params['epsilon']
            )
            
            # 使用交叉驗證評估模型
            cv_scores = cross_val_score(svr, X_train_scaled, y_train, cv=3, scoring='neg_mean_squared_error')
            fitness = -np.mean(cv_scores)  # 負的MSE，越小越好
            
            return fitness
            
        except Exception as e:
            return float('inf')  # 返回最差的適應度
    
    def initialize_population(self):
        """
        初始化族群
        """
        population = []
        for _ in range(self.population_size):
            chromosome = [random.randint(0, 1) for _ in range(40)]  # 16特徵 + 8C + 8gamma + 8epsilon
            population.append(chromosome)
        return population
    
    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """
        競賽選擇
        """
        selected = []
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]  # 選擇適應度最好的
            selected.append(population[winner_idx].copy())
        return selected
    
    def crossover(self, parent1, parent2):
        """
        單點交配
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def mutation(self, chromosome):
        """
        突變操作
        """
        mutated = chromosome.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]
        return mutated
    
    def genetic_algorithm_optimization(self, train_data):
        """
        遺傳演算法主要流程
        """
        # 初始化族群
        population = self.initialize_population()
        best_fitness = float('inf')
        best_chromosome = None
        
        for generation in range(self.generations):
            # 計算適應度
            fitness_scores = []
            for chromosome in population:
                fitness = self.evaluate_chromosome(chromosome, train_data)
                fitness_scores.append(fitness)
            
            # 記錄最佳適應度
            current_best_fitness = min(fitness_scores)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_chromosome = population[np.argmin(fitness_scores)].copy()
            
            print(f"    世代 {generation + 1}: 最佳適應度 = {current_best_fitness:.6f}")
            
            # 選擇
            selected_population = self.tournament_selection(population, fitness_scores)
            
            # 交配和突變
            new_population = []
            for i in range(0, len(selected_population), 2):
                parent1 = selected_population[i]
                parent2 = selected_population[(i + 1) % len(selected_population)]
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        return best_chromosome, best_fitness
    
    def train_svr_model(self, train_data, params):
        """
        訓練SVR模型
        """
        X_train, y_train, _ = self.prepare_features(train_data, params['features'])
        
        # 標準化特徵
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 建立並訓練SVR模型
        svr = SVR(
            kernel='rbf',
            C=params['C'],
            gamma=params['gamma'],
            epsilon=params['epsilon']
        )
        svr.fit(X_train_scaled, y_train)
        
        return svr, scaler
    
    def predict_and_select_stocks(self, svr, scaler, test_data, features, top_n_stocks=10):
        """
        預測並選擇股票
        """
        X_test, actual_returns, stock_names = self.prepare_features(test_data, features)
        X_test_scaled = scaler.transform(X_test)
        
        # 預測報酬率
        predicted_returns = svr.predict(X_test_scaled)
        
        # 根據預測報酬率排序並選擇前N檔股票
        sorted_indices = np.argsort(predicted_returns)[::-1]  # 降序排列
        top_indices = sorted_indices[:top_n_stocks]
        
        selected_returns = actual_returns[top_indices]
        selected_stocks = stock_names[top_indices]
        predicted_selected = predicted_returns[top_indices]
        
        return selected_returns, selected_stocks, predicted_selected
    
    def rolling_window_backtest(self, start_year=1997, end_year=2008):
        """
        執行rolling window回測
        """
        results = []
        
        print("開始混合SVR股票選擇回測...")
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
            
            # 使用遺傳演算法尋找最佳參數
            print(f"  開始遺傳演算法最佳化...")
            best_chromosome, best_fitness = self.genetic_algorithm_optimization(train_data)
            best_params = self.decode_chromosome(best_chromosome)
            
            print(f"  最佳參數: C={best_params['C']:.4f}, gamma={best_params['gamma']:.4f}, "
                  f"epsilon={best_params['epsilon']:.4f}")
            print(f"  選中特徵數: {len(best_params['features'])}")
            print(f"  最佳適應度: {best_fitness:.6f}")
            
            # 使用最佳參數訓練SVR模型
            svr, scaler = self.train_svr_model(train_data, best_params)
            
            # 選擇測試年度的股票
            selected_returns, selected_stocks, predicted_returns = self.predict_and_select_stocks(
                svr, scaler, test_data, best_params['features']
            )
            
            # 計算測試結果
            test_avg_return = np.mean(selected_returns)
            test_std_return = np.std(selected_returns)
            
            result = {
                'test_year': test_year,
                'train_year': train_year,
                'best_params': best_params,
                'train_fitness': best_fitness,
                'test_return': test_avg_return,
                'test_std': test_std_return,
                'test_n_selected': len(selected_returns),
                'selected_returns': selected_returns,
                'selected_stocks': selected_stocks,
                'predicted_returns': predicted_returns
            }
            
            results.append(result)
            
            print(f"  測試集平均報酬率: {test_avg_return:.4f}% ± {test_std_return:.4f}%")
            print(f"  測試集選中股票數: {len(selected_returns)}")
            print(f"  選中股票: {', '.join(selected_stocks[:5])}{'...' if len(selected_stocks) > 5 else ''}")
            print("-" * 60)
        
        return results
    
    def save_results_to_csv(self, results, filename='hybrid_svr_stock_selection_results.csv'):
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
                'C': result['best_params']['C'],
                'gamma': result['best_params']['gamma'],
                'epsilon': result['best_params']['epsilon'],
                'n_features': len(result['best_params']['features']),
                'train_fitness': result['train_fitness'],
                'test_return': result['test_return'],
                'test_std': result['test_std'],
                'test_n_selected': result['test_n_selected']
            })
        
        # 寫入主要結果CSV檔案
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['test_year', 'train_year', 'C', 'gamma', 'epsilon', 
                         'n_features', 'train_fitness', 'test_return', 
                         'test_std', 'test_n_selected']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        
        print(f"主要結果已儲存至: {filename}")
        
        # 儲存特徵選擇結果
        feature_filename = filename.replace('.csv', '_features.csv')
        with open(feature_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['test_year', 'feature_name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                for feature in result['best_params']['features']:
                    writer.writerow({
                        'test_year': result['test_year'],
                        'feature_name': feature
                    })
        
        print(f"特徵選擇結果已儲存至: {feature_filename}")
        
        # 儲存詳細的選股結果
        detail_filename = filename.replace('.csv', '_selected_stocks.csv')
        with open(detail_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['test_year', 'stock_name', 'actual_return', 'predicted_return']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                for stock, actual_ret, pred_ret in zip(result['selected_stocks'], 
                                                     result['selected_returns'], 
                                                     result['predicted_returns']):
                    writer.writerow({
                        'test_year': result['test_year'],
                        'stock_name': stock,
                        'actual_return': actual_ret,
                        'predicted_return': pred_ret
                    })
        
        print(f"選股詳細結果已儲存至: {detail_filename}")
    
    def analyze_results(self, results):
        """
        分析回測結果
        """
        if not results:
            print("無結果可分析")
            return None
        
        # 計算總體統計
        test_returns = [r['test_return'] for r in results]
        train_fitness = [r['train_fitness'] for r in results]
        
        print("\n" + "=" * 60)
        print("混合SVR股票選擇結果分析")
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
        print(f"平均適應度: {np.mean(train_fitness):.6f}")
        print(f"適應度標準差: {np.std(train_fitness):.6f}")
        
        # 年度詳細結果
        print()
        print("【年度詳細結果】")
        print("年份    C值      gamma    epsilon  特徵數  訓練適應度    測試報酬率  選股數量")
        print("-" * 80)
        for result in results:
            print(f"{result['test_year']}  {result['best_params']['C']:8.2f}  "
                  f"{result['best_params']['gamma']:8.4f}  {result['best_params']['epsilon']:8.4f}  "
                  f"{len(result['best_params']['features']):4d}     {result['train_fitness']:8.4f}     "
                  f"{result['test_return']:8.2f}%     {result['test_n_selected']:2d}")
        
        # 特徵使用頻率分析
        print()
        print("【特徵使用頻率分析】")
        feature_usage = {}
        for result in results:
            for feature in result['best_params']['features']:
                feature_usage[feature] = feature_usage.get(feature, 0) + 1
        
        sorted_features = sorted(feature_usage.items(), key=lambda x: x[1], reverse=True)
        for feature, count in sorted_features:
            print(f"{feature}: {count}/{len(results)} 次 ({count/len(results)*100:.1f}%)")
        
        return {
            'avg_test_return': np.mean(test_returns),
            'std_test_return': np.std(test_returns),
            'avg_train_fitness': np.mean(train_fitness),
            'best_test_return': np.max(test_returns),
            'worst_test_return': np.min(test_returns),
            'win_rate': sum(1 for r in test_returns if r > 0) / len(test_returns),
            'sharpe_ratio': np.mean(test_returns) / np.std(test_returns) if np.std(test_returns) != 0 else 0,
            'feature_usage': feature_usage,
            'results': results
        }

def main():
    """
    主函數
    """
    print("混合SVR股票選擇系統")
    print("=" * 60)
    
    # 初始化選股器
    try:
        selector = HybridSVRStockSelector('top200.xlsx')
        print(f"成功載入資料: {len(selector.data)} 筆記錄")
        print(f"資料年份範圍: {selector.data['年月'].min()} - {selector.data['年月'].max()}")
        print(f"特徵數量: {len(selector.feature_columns)}")
        print(f"遺傳演算法參數: 族群大小={selector.population_size}, 世代數={selector.generations}")
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
