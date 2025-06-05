import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import csv
import random
warnings.filterwarnings('ignore')

class GeneticAlgorithmStockSelector:
    def __init__(self, data_path='top200.xlsx'):
        """
        初始化遺傳演算法股票選擇器
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
        self.population_size = 100
        self.generations = 50
        self.crossover_rate = 0.7
        self.mutation_rate = 0.05
        self.chromosome_length = len(self.feature_columns)
        
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
    
    def rank_stocks_by_indicators(self, data):
        """
        根據基本分析指標對股票進行排序評分
        """
        # 需要越低越好的指標（排序時分數越高越好）
        lower_is_better = ['股價淨值比', '股價營收比', '負債/淨值比']
        
        # 需要越高越好的指標
        higher_is_better = [col for col in self.feature_columns if col not in lower_is_better]
        
        scores = np.zeros((len(data), len(self.feature_columns)))
        
        for i, feature in enumerate(self.feature_columns):
            feature_values = data[feature].values
            
            if feature in lower_is_better:
                # 值越低排名越高（分數越高）
                ranks = len(data) + 1 - np.argsort(np.argsort(feature_values)) - 1
            else:
                # 值越高排名越高（分數越高）
                ranks = len(data) - np.argsort(np.argsort(feature_values))
            
            scores[:, i] = ranks
        
        return scores
    
    def binary_encode_weights(self, weights, bits_per_weight=8):
        """
        將權重編碼為二進制染色體
        """
        chromosome = []
        for weight in weights:
            # 將0-1的權重轉換為0-255的整數
            int_weight = int(weight * (2**bits_per_weight - 1))
            # 轉換為二進制
            binary = format(int_weight, f'0{bits_per_weight}b')
            chromosome.extend([int(bit) for bit in binary])
        return chromosome
    
    def binary_decode_weights(self, chromosome, bits_per_weight=8):
        """
        將二進制染色體解碼為權重
        """
        weights = []
        for i in range(0, len(chromosome), bits_per_weight):
            # 取出對應的二進制位
            binary_str = ''.join(str(bit) for bit in chromosome[i:i+bits_per_weight])
            # 轉換為整數再正規化到0-1
            int_weight = int(binary_str, 2)
            weight = int_weight / (2**bits_per_weight - 1)
            weights.append(weight)
        return np.array(weights)
    
    def initialize_population(self):
        """
        初始化族群
        """
        population = []
        bits_per_weight = 8
        chromosome_length = len(self.feature_columns) * bits_per_weight
        
        for _ in range(self.population_size):
            chromosome = [random.randint(0, 1) for _ in range(chromosome_length)]
            population.append(chromosome)
        
        return population
    
    def calculate_fitness(self, chromosome, stock_scores, returns, top_n_stocks=10):
        """
        計算適應度函數
        """
        # 解碼權重
        weights = self.binary_decode_weights(chromosome)
        
        # 計算每檔股票的綜合評分
        stock_total_scores = np.dot(stock_scores, weights)
        
        # 選擇評分最高的前N檔股票
        top_indices = np.argsort(stock_total_scores)[-top_n_stocks:]
        selected_returns = returns[top_indices]
        
        # 計算平均報酬率作為適應度
        avg_return = np.mean(selected_returns)
        
        return avg_return
    
    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """
        競賽選擇
        """
        selected = []
        for _ in range(len(population)):
            # 隨機選擇參與競賽的個體
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # 選擇適應度最高的個體
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        return selected
    
    def single_point_crossover(self, parent1, parent2):
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
                mutated[i] = 1 - mutated[i]  # 0變1，1變0
        return mutated
    
    def genetic_algorithm_optimization(self, stock_scores, returns):
        """
        遺傳演算法主要流程
        """
        # 初始化族群
        population = self.initialize_population()
        best_fitness_history = []
        best_chromosome = None
        best_fitness = -np.inf
        
        for generation in range(self.generations):
            # 計算適應度
            fitness_scores = []
            for chromosome in population:
                fitness = self.calculate_fitness(chromosome, stock_scores, returns)
                fitness_scores.append(fitness)
            
            # 記錄最佳適應度
            current_best_fitness = max(fitness_scores)
            best_fitness_history.append(current_best_fitness)
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_chromosome = population[np.argmax(fitness_scores)].copy()
            
            # 選擇
            selected_population = self.tournament_selection(population, fitness_scores)
            
            # 交配和突變
            new_population = []
            for i in range(0, len(selected_population), 2):
                parent1 = selected_population[i]
                parent2 = selected_population[(i + 1) % len(selected_population)]
                
                child1, child2 = self.single_point_crossover(parent1, parent2)
                
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        return best_chromosome, best_fitness, best_fitness_history
    
    def select_stocks_with_weights(self, weights, test_data, top_n_stocks=10):
        """
        使用最佳權重選擇股票
        """
        # 對測試資料進行排序評分
        test_scores = self.rank_stocks_by_indicators(test_data)
        
        # 計算綜合評分
        stock_total_scores = np.dot(test_scores, weights)
        
        # 選擇評分最高的前N檔股票
        top_indices = np.argsort(stock_total_scores)[-top_n_stocks:]
        
        selected_returns = test_data['Return'].values[top_indices]
        selected_stocks = test_data['簡稱'].values[top_indices]
        
        return selected_returns, selected_stocks, weights
    
    def rolling_window_backtest(self, start_year=1997, end_year=2008):
        """
        執行rolling window回測
        """
        results = []
        
        print("開始遺傳演算法股票選擇回測...")
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
            
            # 對訓練資料進行排序評分
            train_scores = self.rank_stocks_by_indicators(train_data)
            train_returns = train_data['Return'].values
            
            print(f"  開始遺傳演算法最佳化...")
            
            # 使用遺傳演算法尋找最佳權重
            best_chromosome, best_fitness, fitness_history = self.genetic_algorithm_optimization(
                train_scores, train_returns
            )
            
            # 解碼最佳權重
            best_weights = self.binary_decode_weights(best_chromosome)
            
            print(f"  訓練完成，最佳適應度: {best_fitness:.4f}%")
            print(f"  最佳權重前5項: {best_weights[:5]}")
            
            # 使用最佳權重選擇測試年度的股票
            selected_returns, selected_stocks, final_weights = self.select_stocks_with_weights(
                best_weights, test_data
            )
            
            # 計算測試結果
            test_avg_return = np.mean(selected_returns)
            test_std_return = np.std(selected_returns)
            
            result = {
                'test_year': test_year,
                'train_year': train_year,
                'best_weights': best_weights,
                'train_fitness': best_fitness,
                'test_return': test_avg_return,
                'test_std': test_std_return,
                'test_n_selected': len(selected_returns),
                'selected_returns': selected_returns,
                'selected_stocks': selected_stocks,
                'fitness_history': fitness_history
            }
            
            results.append(result)
            
            print(f"  測試集平均報酬率: {test_avg_return:.4f}% ± {test_std_return:.4f}%")
            print(f"  測試集選中股票數: {len(selected_returns)}")
            print(f"  選中股票: {', '.join(selected_stocks[:5])}{'...' if len(selected_stocks) > 5 else ''}")
            print("-" * 60)
        
        return results
    
    def save_results_to_csv(self, results, filename='ga_stock_selection_results.csv'):
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
                'train_fitness': result['train_fitness'],
                'test_return': result['test_return'],
                'test_std': result['test_std'],
                'test_n_selected': result['test_n_selected']
            })
        
        # 寫入主要結果CSV檔案
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['test_year', 'train_year', 'train_fitness', 
                         'test_return', 'test_std', 'test_n_selected']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        
        print(f"主要結果已儲存至: {filename}")
        
        # 儲存權重資訊
        weights_filename = filename.replace('.csv', '_weights.csv')
        with open(weights_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['test_year'] + [f'weight_{i}' for i in range(len(self.feature_columns))]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                row = {'test_year': result['test_year']}
                for i, weight in enumerate(result['best_weights']):
                    row[f'weight_{i}'] = weight
                writer.writerow(row)
        
        print(f"權重資訊已儲存至: {weights_filename}")
        
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
        print("遺傳演算法股票選擇結果分析")
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
        print(f"平均適應度: {np.mean(train_fitness):.4f}%")
        print(f"適應度標準差: {np.std(train_fitness):.4f}%")
        
        # 年度詳細結果
        print()
        print("【年度詳細結果】")
        print("年份   訓練適應度  測試報酬率  選股數量")
        print("-" * 40)
        for result in results:
            print(f"{result['test_year']}    {result['train_fitness']:8.2f}%   {result['test_return']:8.2f}%     {result['test_n_selected']:2d}")
        
        # 權重分析
        print()
        print("【平均權重分析】")
        avg_weights = np.mean([r['best_weights'] for r in results], axis=0)
        for i, (feature, weight) in enumerate(zip(self.feature_columns, avg_weights)):
            print(f"{feature}: {weight:.4f}")
        
        return {
            'avg_test_return': np.mean(test_returns),
            'std_test_return': np.std(test_returns),
            'avg_train_fitness': np.mean(train_fitness),
            'best_test_return': np.max(test_returns),
            'worst_test_return': np.min(test_returns),
            'win_rate': sum(1 for r in test_returns if r > 0) / len(test_returns),
            'sharpe_ratio': np.mean(test_returns) / np.std(test_returns) if np.std(test_returns) != 0 else 0,
            'avg_weights': avg_weights,
            'results': results
        }

def main():
    """
    主函數
    """
    print("遺傳演算法股票選擇系統")
    print("=" * 60)
    
    # 初始化選股器
    try:
        selector = GeneticAlgorithmStockSelector('top200.xlsx')
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
