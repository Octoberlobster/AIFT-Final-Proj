import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
import os
import platform
from deap import base, creator, tools, algorithms
import random
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

class Paper3StockSelector:
    def __init__(self, data_path, output_dir='Q4output'):
        """初始化Paper 3股票選股模型（GA-SVR混合方法）"""
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
        
        # GA參數設定
        self.population_size = 50
        self.generations = 30
        self.crossover_prob = 0.8
        self.mutation_prob = 0.1
        
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
    
    def create_enhanced_features(self, data):
        """創建進階特徵（Paper 3的特徵工程）"""
        enhanced_data = data.copy()
        
        # 1. 價格相關比率
        enhanced_data['PE_ratio'] = enhanced_data['收盤價(元)_年'] / (enhanced_data['M稅後淨利成長率'] + 1e-6)
        enhanced_data['PB_ratio'] = enhanced_data['股價淨值比']
        enhanced_data['PS_ratio'] = enhanced_data['股價營收比']
        
        # 2. 獲利能力指標
        enhanced_data['ROE'] = enhanced_data['M淨值報酬率─稅後']
        enhanced_data['ROA'] = enhanced_data['資產報酬率ROA']
        enhanced_data['OPM'] = enhanced_data['營業利益率OPM']
        enhanced_data['NPM'] = enhanced_data['利潤邊際NPM']
        
        # 3. 槓桿比率
        enhanced_data['DE_ratio'] = enhanced_data['負債/淨值比']
        
        # 4. 流動性指標
        enhanced_data['CR'] = enhanced_data['M流動比率']
        enhanced_data['QR'] = enhanced_data['M速動比率']
        
        # 5. 效率指標
        enhanced_data['ITR'] = enhanced_data['M存貨週轉率 (次)']
        enhanced_data['RTR'] = enhanced_data['M應收帳款週轉次']
        
        # 6. 成長性指標
        enhanced_data['OIG'] = enhanced_data['M營業利益成長率']
        enhanced_data['NIG'] = enhanced_data['M稅後淨利成長率']
        
        # 7. 綜合指標
        enhanced_data['profitability_score'] = (enhanced_data['ROA'] + enhanced_data['ROE'] + 
                                               enhanced_data['OPM'] + enhanced_data['NPM']) / 4
        enhanced_data['liquidity_score'] = (enhanced_data['CR'] + enhanced_data['QR']) / 2
        enhanced_data['efficiency_score'] = (enhanced_data['ITR'] + enhanced_data['RTR']) / 2
        enhanced_data['growth_score'] = (enhanced_data['OIG'] + enhanced_data['NIG']) / 2
        
        # 8. 市場估值綜合指標
        enhanced_data['valuation_score'] = (enhanced_data['PB_ratio'] + enhanced_data['PS_ratio']) / 2
        
        return enhanced_data
    
    def setup_ga(self):
        """設定遺傳演算法"""
        # 創建適應度函數和個體類型
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # 特徵選擇部分（二進制編碼）
        n_features = len(self.feature_columns) + 8  # 原始特徵 + 新增特徵
        
        # 註冊遺傳演算法組件
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("attr_float", random.uniform, 0, 1)
        
        # 個體結構：特徵選擇(n_features bits) + C(10 bits) + gamma(10 bits) + epsilon(10 bits)
        def create_individual():
            # 特徵選擇部分
            features = [self.toolbox.attr_bool() for _ in range(n_features)]
            # 確保至少選擇3個特徵
            if sum(features) < 3:
                indices = random.sample(range(n_features), 3)
                for idx in indices:
                    features[idx] = 1
            
            # SVR參數部分（使用浮點數編碼）
            c_param = self.toolbox.attr_float()
            gamma_param = self.toolbox.attr_float()
            epsilon_param = self.toolbox.attr_float()
            
            return features + [c_param, gamma_param, epsilon_param]
        
        self.toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
    def decode_individual(self, individual):
        """解碼個體"""
        n_features = len(self.feature_columns) + 8
        
        # 特徵選擇
        feature_mask = individual[:n_features]
        
        # SVR參數
        c_encoded = individual[n_features]
        gamma_encoded = individual[n_features + 1]
        epsilon_encoded = individual[n_features + 2]
        
        # 參數映射到實際範圍
        C = 0.1 + c_encoded * (1000 - 0.1)
        gamma = 0.001 + gamma_encoded * (10 - 0.001)
        epsilon = 0.01 + epsilon_encoded * (1.0 - 0.01)
        
        return feature_mask, C, gamma, epsilon
    
    def evaluate_individual(self, individual):
        """評估個體適應度"""
        try:
            feature_mask, C, gamma, epsilon = self.decode_individual(individual)
            
            # 獲取選中的特徵
            all_features = self.feature_columns + ['profitability_score', 'liquidity_score', 
                                                 'efficiency_score', 'growth_score', 
                                                 'valuation_score', 'PE_ratio', 'DE_ratio', 'CR']
            selected_features = [feat for i, feat in enumerate(all_features) if feature_mask[i] == 1]
            
            if len(selected_features) < 2:
                return (0.0,)
            
            # 使用當前訓練數據評估
            if not hasattr(self, 'current_train_data'):
                return (0.0,)
            
            train_data = self.current_train_data
            
            # 檢查特徵是否存在
            available_features = [f for f in selected_features if f in train_data.columns]
            if len(available_features) < 2:
                return (0.0,)
            
            X_train = train_data[available_features].values
            y_train = train_data['Return'].values
            
            # 標準化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # 訓練SVR
            svr = SVR(C=C, gamma=gamma, epsilon=epsilon, kernel='rbf')
            svr.fit(X_train_scaled, y_train)
            
            # 預測
            y_pred = svr.predict(X_train_scaled)
            
            # 計算適應度（使用負MSE，因為我們要最大化）
            mse = mean_squared_error(y_train, y_pred)
            fitness = -mse
            
            return (fitness,)
            
        except Exception as e:
            return (0.0,)
    
    def crossover(self, ind1, ind2):
        """交叉操作"""
        n_features = len(self.feature_columns) + 8
        
        # 特徵部分使用單點交叉
        if random.random() < 0.5:
            point = random.randint(1, n_features - 1)
            ind1[:point], ind2[:point] = ind2[:point], ind1[:point]
        
        # 參數部分使用算術交叉
        for i in range(n_features, len(ind1)):
            if random.random() < 0.5:
                alpha = random.random()
                temp1 = alpha * ind1[i] + (1 - alpha) * ind2[i]
                temp2 = alpha * ind2[i] + (1 - alpha) * ind1[i]
                ind1[i] = temp1
                ind2[i] = temp2
        
        return ind1, ind2
    
    def mutate(self, individual):
        """突變操作"""
        n_features = len(self.feature_columns) + 8
        
        # 特徵部分突變
        for i in range(n_features):
            if random.random() < self.mutation_prob:
                individual[i] = 1 - individual[i]
        
        # 確保至少有3個特徵被選中
        if sum(individual[:n_features]) < 3:
            indices = random.sample(range(n_features), 3)
            for idx in indices:
                individual[idx] = 1
        
        # 參數部分突變
        for i in range(n_features, len(individual)):
            if random.random() < self.mutation_prob:
                individual[i] = random.random()
        
        return (individual,)
    
    def train_and_evaluate(self, train_year, test_year):
        """訓練並評估模型"""
        train_data = self.data[self.data['年份'] == train_year].copy()
        test_data = self.data[self.data['年份'] == test_year].copy()
        
        if len(train_data) == 0 or len(test_data) == 0:
            return None
        
        # 創建進階特徵
        train_enhanced = self.create_enhanced_features(train_data)
        test_enhanced = self.create_enhanced_features(test_data)
        
        # 設定當前訓練數據供GA使用
        self.current_train_data = train_enhanced
        
        # 設定GA
        self.setup_ga()
        
        # 創建初始族群
        population = self.toolbox.population(n=self.population_size)
        
        # 執行GA
        algorithms.eaSimple(population, self.toolbox, 
                          cxpb=self.crossover_prob, 
                          mutpb=self.mutation_prob, 
                          ngen=self.generations, 
                          verbose=False)
        
        # 獲取最佳個體
        best_individual = tools.selBest(population, 1)[0]
        feature_mask, best_C, best_gamma, best_epsilon = self.decode_individual(best_individual)
        
        # 獲取最佳特徵組合
        all_features = self.feature_columns + ['profitability_score', 'liquidity_score', 
                                             'efficiency_score', 'growth_score', 
                                             'valuation_score', 'PE_ratio', 'DE_ratio', 'CR']
        best_features = [feat for i, feat in enumerate(all_features) if feature_mask[i] == 1]
        
        # 檢查特徵是否存在
        available_features = [f for f in best_features if f in train_enhanced.columns and f in test_enhanced.columns]
        
        if len(available_features) < 2:
            return None
        
        try:
            # 準備最終訓練數據
            X_train = train_enhanced[available_features].values
            y_train = train_enhanced['Return'].values
            X_test = test_enhanced[available_features].values
            
            # 標準化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 訓練最終SVR模型
            final_svr = SVR(C=best_C, gamma=best_gamma, epsilon=best_epsilon, kernel='rbf')
            final_svr.fit(X_train_scaled, y_train)
            
            # 預測測試集
            y_pred = final_svr.predict(X_test_scaled)
            
            # 根據預測結果排序股票
            test_enhanced['predicted_return'] = y_pred
            test_enhanced_sorted = test_enhanced.sort_values('predicted_return', ascending=False)
            
            # 選擇前10支股票（根據Paper 3的要求）
            top_10_stocks = test_enhanced_sorted.head(10)
            avg_return = top_10_stocks['Return'].mean()
            
            # 也選擇前30支股票用於比較
            top_30_stocks = test_enhanced_sorted.head(30)
            avg_return_30 = top_30_stocks['Return'].mean()
            
            return {
                'train_year': train_year,
                'test_year': test_year,
                'best_return': avg_return,
                'best_return_30': avg_return_30,
                'best_features': available_features,
                'best_C': best_C,
                'best_gamma': best_gamma,
                'best_epsilon': best_epsilon,
                'num_features': len(available_features),
                'num_selected_stocks': len(top_10_stocks),
                'selected_stocks_top10': top_10_stocks,
                'selected_stocks_top30': top_30_stocks,
                'fitness': best_individual.fitness.values[0],
                'all_stocks_with_predictions': test_enhanced_sorted
            }
            
        except Exception as e:
            print(f"Error in train_and_evaluate: {e}")
            return None
    
    def run_rolling_window_analysis(self):
        """執行滾動視窗分析"""
        years = sorted(self.data['年份'].unique())
        
        print(f"開始Paper 3 GA-SVR演算法分析")
        print(f"使用遺傳演算法優化SVR參數和特徵選擇")
        print(f"年份範圍: {years}")
        
        for i in range(len(years) - 1):
            train_year = years[i]
            test_year = years[i + 1]
            
            print(f"\n分析 {train_year} -> {test_year}")
            
            result = self.train_and_evaluate(train_year, test_year)
            
            if result and result['best_return'] is not None:
                self.results.append(result)
                print(f"前10支股票平均報酬率: {result['best_return']:.4f}%")
                print(f"前30支股票平均報酬率: {result['best_return_30']:.4f}%")
                print(f"最佳C: {result['best_C']:.4f}")
                print(f"最佳gamma: {result['best_gamma']:.4f}")
                print(f"最佳epsilon: {result['best_epsilon']:.4f}")
                print(f"特徵數量: {result['num_features']}")
                print(f"適應度: {result['fitness']:.4f}")
            else:
                print("未找到有效結果")
    
    def save_results_to_csv(self):
        """儲存結果到CSV檔案"""
        if not self.results:
            print("沒有結果可儲存")
            return
        
        results_data = []
        top10_stocks_data = []
        top30_stocks_data = []
        all_predictions_data = []
        
        for result in self.results:
            results_data.append({
                '訓練年份': result['train_year'],
                '測試年份': result['test_year'],
                '前10支股票平均報酬率(%)': result['best_return'],
                '前30支股票平均報酬率(%)': result['best_return_30'],
                '最佳C參數': result['best_C'],
                '最佳gamma參數': result['best_gamma'],
                '最佳epsilon參數': result['best_epsilon'],
                '特徵數量': result['num_features'],
                'GA適應度': result['fitness'],
                '最佳特徵組合': ', '.join(result['best_features'])
            })
            
            # 儲存前10支股票詳細資訊
            if result['selected_stocks_top10'] is not None:
                stocks = result['selected_stocks_top10'].copy()
                stocks['訓練年份'] = result['train_year']
                stocks['測試年份'] = result['test_year']
                stocks['使用C參數'] = result['best_C']
                stocks['使用gamma參數'] = result['best_gamma']
                stocks['使用epsilon參數'] = result['best_epsilon']
                stocks['排名'] = range(1, len(stocks) + 1)
                top10_stocks_data.append(stocks)
            
            # 儲存前30支股票詳細資訊
            if result['selected_stocks_top30'] is not None:
                stocks = result['selected_stocks_top30'].copy()
                stocks['訓練年份'] = result['train_year']
                stocks['測試年份'] = result['test_year']
                stocks['使用C參數'] = result['best_C']
                stocks['使用gamma參數'] = result['best_gamma']
                stocks['使用epsilon參數'] = result['best_epsilon']
                stocks['排名'] = range(1, len(stocks) + 1)
                top30_stocks_data.append(stocks)
            
            # 儲存所有股票的預測結果
            if result['all_stocks_with_predictions'] is not None:
                all_stocks = result['all_stocks_with_predictions'].copy()
                all_stocks['訓練年份'] = result['train_year']
                all_stocks['測試年份'] = result['test_year']
                all_stocks['排名'] = range(1, len(all_stocks) + 1)
                all_predictions_data.append(all_stocks)
        
        # 儲存主要結果
        results_df = pd.DataFrame(results_data)
        results_path = os.path.join(self.output_dir, 'paper3_ga_svr_results.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        
        # 儲存前10支股票詳細資訊
        if top10_stocks_data:
            all_top10_stocks = pd.concat(top10_stocks_data, ignore_index=True)
            top10_path = os.path.join(self.output_dir, 'paper3_top10_stocks_details.csv')
            all_top10_stocks.to_csv(top10_path, index=False, encoding='utf-8-sig')
        
        # 儲存前30支股票詳細資訊
        if top30_stocks_data:
            all_top30_stocks = pd.concat(top30_stocks_data, ignore_index=True)
            top30_path = os.path.join(self.output_dir, 'paper3_top30_stocks_details.csv')
            all_top30_stocks.to_csv(top30_path, index=False, encoding='utf-8-sig')
        
        # 儲存所有股票預測結果
        if all_predictions_data:
            all_predictions = pd.concat(all_predictions_data, ignore_index=True)
            predictions_path = os.path.join(self.output_dir, 'paper3_all_stock_predictions.csv')
            all_predictions.to_csv(predictions_path, index=False, encoding='utf-8-sig')
        
        print(f"結果已儲存到 {self.output_dir} 資料夾:")
        print(f"- paper3_ga_svr_results.csv (主要結果)")
        print(f"- paper3_top10_stocks_details.csv (前10支股票詳細資訊)")
        print(f"- paper3_top30_stocks_details.csv (前30支股票詳細資訊)")
        print(f"- paper3_all_stock_predictions.csv (所有股票預測結果)")
        
        return results_df
    
    def create_visualizations(self):
        """創建視覺化圖表"""
        if not self.results:
            print("沒有結果可視覺化")
            return
        
        setup_chinese_font()
        
        years = [r['test_year'] for r in self.results]
        returns_10 = [r['best_return'] for r in self.results]
        returns_30 = [r['best_return_30'] for r in self.results]
        c_params = [r['best_C'] for r in self.results]
        gamma_params = [r['best_gamma'] for r in self.results]
        epsilon_params = [r['best_epsilon'] for r in self.results]
        num_features = [r['num_features'] for r in self.results]
        fitness_scores = [r['fitness'] for r in self.results]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Paper 3 GA-SVR股票選股模型分析結果', fontsize=16, fontweight='bold')
        
        # 1. 年度報酬率趨勢比較（前10 vs 前30）
        axes[0, 0].plot(years, returns_10, marker='o', linewidth=2, markersize=8, color='green', label='前10支股票')
        axes[0, 0].plot(years, returns_30, marker='s', linewidth=2, markersize=6, color='blue', label='前30支股票')
        axes[0, 0].set_title('年度最佳平均報酬率趨勢比較')
        axes[0, 0].set_xlabel('年份')
        axes[0, 0].set_ylabel('平均報酬率 (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].legend()
        
        # 2. SVR參數C的變化
        axes[0, 1].plot(years, c_params, marker='s', linewidth=2, markersize=6, color='blue')
        axes[0, 1].set_title('最佳C參數變化趨勢')
        axes[0, 1].set_xlabel('年份')
        axes[0, 1].set_ylabel('C參數值')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # 3. SVR參數gamma的變化
        axes[0, 2].plot(years, gamma_params, marker='^', linewidth=2, markersize=6, color='red')
        axes[0, 2].set_title('最佳gamma參數變化趨勢')
        axes[0, 2].set_xlabel('年份')
        axes[0, 2].set_ylabel('gamma參數值')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_yscale('log')
        
        # 4. 特徵數量 vs 報酬率（前10支股票）
        scatter = axes[1, 0].scatter(num_features, returns_10, c=years, cmap='viridis', s=100, alpha=0.7)
        axes[1, 0].set_title('特徵數量 vs 前10支股票平均報酬率')
        axes[1, 0].set_xlabel('特徵數量')
        axes[1, 0].set_ylabel('平均報酬率 (%)')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='年份')
        
        # 5. GA適應度變化
        axes[1, 1].plot(years, fitness_scores, marker='d', linewidth=2, markersize=6, color='purple')
        axes[1, 1].set_title('GA適應度變化趨勢')
        axes[1, 1].set_xlabel('年份')
        axes[1, 1].set_ylabel('適應度分數')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 累積報酬率比較
        cumulative_returns_10 = np.cumsum(returns_10)
        cumulative_returns_30 = np.cumsum(returns_30)
        axes[1, 2].plot(years, cumulative_returns_10, marker='o', linewidth=2, markersize=6, color='green', label='前10支股票')
        axes[1, 2].plot(years, cumulative_returns_30, marker='s', linewidth=2, markersize=6, color='blue', label='前30支股票')
        axes[1, 2].fill_between(years, cumulative_returns_10, alpha=0.3, color='green')
        axes[1, 2].fill_between(years, cumulative_returns_30, alpha=0.3, color='blue')
        axes[1, 2].set_title('累積報酬率趨勢比較')
        axes[1, 2].set_xlabel('年份')
        axes[1, 2].set_ylabel('累積報酬率 (%)')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()
        
        plt.tight_layout()
        chart_path = os.path.join(self.output_dir, 'paper3_ga_svr_analysis.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_top10_analysis_chart(self):
        """創建前10支股票專門分析圖表"""
        if not self.results:
            return
        
        setup_chinese_font()
        
        # 分析前10支股票的表現
        years = [r['test_year'] for r in self.results]
        returns_10 = [r['best_return'] for r in self.results]
        
        # 計算年度排名分布
        yearly_rankings = {}
        for result in self.results:
            if result['selected_stocks_top10'] is not None:
                year = result['test_year']
                stocks = result['selected_stocks_top10']
                yearly_rankings[year] = stocks['Return'].tolist()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('前10支股票詳細分析', fontsize=16, fontweight='bold')
        
        # 1. 前10支股票年度報酬率分布
        colors = ['green' if r > 0 else 'red' for r in returns_10]
        axes[0, 0].bar(years, returns_10, color=colors, alpha=0.7)
        axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 0].set_title('前10支股票年度平均報酬率')
        axes[0, 0].set_xlabel('年份')
        axes[0, 0].set_ylabel('平均報酬率 (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 前10支股票報酬率箱型圖
        if yearly_rankings:
            box_data = []
            box_labels = []
            for year in sorted(yearly_rankings.keys()):
                box_data.append(yearly_rankings[year])
                box_labels.append(str(year))
            
            axes[0, 1].boxplot(box_data, labels=box_labels)
            axes[0, 1].set_title('前10支股票個別報酬率分布')
            axes[0, 1].set_xlabel('年份')
            axes[0, 1].set_ylabel('個別股票報酬率 (%)')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 勝率統計
        positive_years = sum(1 for r in returns_10 if r > 0)
        total_years = len(returns_10)
        win_rate = positive_years / total_years * 100
        
        axes[1, 0].pie([positive_years, total_years - positive_years], 
                      labels=[f'正報酬\n({positive_years}年)', f'負報酬\n({total_years - positive_years}年)'],
                      colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title(f'前10支股票勝率統計\n總勝率: {win_rate:.1f}%')
        
        # 4. 累積報酬率與統計指標
        cumulative_returns = np.cumsum(returns_10)
        axes[1, 1].plot(years, cumulative_returns, marker='o', linewidth=2, markersize=6, color='blue')
        axes[1, 1].fill_between(years, cumulative_returns, alpha=0.3, color='blue')
        axes[1, 1].set_title('前10支股票累積報酬率')
        axes[1, 1].set_xlabel('年份')
        axes[1, 1].set_ylabel('累積報酬率 (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加統計信息
        mean_return = np.mean(returns_10)
        std_return = np.std(returns_10)
        max_return = np.max(returns_10)
        min_return = np.min(returns_10)
        
        stats_text = f'平均: {mean_return:.2f}%\n標準差: {std_return:.2f}%\n最高: {max_return:.2f}%\n最低: {min_return:.2f}%'
        axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        top10_chart_path = os.path.join(self.output_dir, 'paper3_top10_analysis.png')
        plt.savefig(top10_chart_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_parameter_analysis_chart(self):
        """創建參數分析圖表"""
        if not self.results:
            return
        
        setup_chinese_font()
        
        c_params = [r['best_C'] for r in self.results]
        gamma_params = [r['best_gamma'] for r in self.results]
        epsilon_params = [r['best_epsilon'] for r in self.results]
        returns = [r['best_return'] for r in self.results]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('GA-SVR參數分析', fontsize=16, fontweight='bold')
        
        # 1. C參數分布
        axes[0, 0].hist(c_params, bins=10, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('C參數分布')
        axes[0, 0].set_xlabel('C參數值')
        axes[0, 0].set_ylabel('頻次')
        axes[0, 0].set_xscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. gamma參數分布
        axes[0, 1].hist(gamma_params, bins=10, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_title('gamma參數分布')
        axes[0, 1].set_xlabel('gamma參數值')
        axes[0, 1].set_ylabel('頻次')
        axes[0, 1].set_xscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. epsilon參數分布
        axes[1, 0].hist(epsilon_params, bins=10, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_title('epsilon參數分布')
        axes[1, 0].set_xlabel('epsilon參數值')
        axes[1, 0].set_ylabel('頻次')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 參數相關性熱圖
        param_data = pd.DataFrame({
            'C': c_params,
            'gamma': gamma_params,
            'epsilon': epsilon_params,
            'return': returns
        })
        correlation_matrix = param_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('參數相關性分析')
        
        plt.tight_layout()
        param_chart_path = os.path.join(self.output_dir, 'ga_svr_parameter_analysis.png')
        plt.savefig(param_chart_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_feature_importance_chart(self):
        """創建特徵重要性圖表"""
        setup_chinese_font()
        
        feature_counts = {}
        for result in self.results:
            for feature in result['best_features']:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        if not feature_counts:
            return
        
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        features, counts = zip(*sorted_features)
        
        plt.figure(figsize=(14, 8))
        bars = plt.barh(range(len(features)), counts, alpha=0.7, color='lightblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('被選中次數')
        plt.title('GA-SVR特徵重要性分析（被選為最佳特徵的次數）')
        plt.grid(True, alpha=0.3)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    str(counts[i]), ha='left', va='center')
        
        plt.tight_layout()
        feature_chart_path = os.path.join(self.output_dir, 'ga_svr_feature_importance.png')
        plt.savefig(feature_chart_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_analysis_report(self):
        """儲存分析報告"""
        if not self.results:
            return
        
        returns_10 = [r['best_return'] for r in self.results]
        returns_30 = [r['best_return_30'] for r in self.results]
        c_params = [r['best_C'] for r in self.results]
        gamma_params = [r['best_gamma'] for r in self.results]
        epsilon_params = [r['best_epsilon'] for r in self.results]
        
        report = f"""
Paper 3 GA-SVR股票選股模型分析報告
{'='*50}

分析概況:
- 分析期間: {self.results[0]['train_year']}-{self.results[-1]['test_year']}
- 總測試年數: {len(self.results)}
- 使用演算法: 遺傳演算法優化的支持向量回歸 (GA-SVR)
- 族群大小: {self.population_size}
- 演化世代: {self.generations}

前10支股票績效統計:
- 平均年報酬率: {np.mean(returns_10):.4f}%
- 報酬率標準差: {np.std(returns_10):.4f}%
- 最高年報酬率: {np.max(returns_10):.4f}%
- 最低年報酬率: {np.min(returns_10):.4f}%
- 正報酬年數: {sum(1 for r in returns_10 if r > 0)}/{len(returns_10)}
- 勝率: {sum(1 for r in returns_10 if r > 0)/len(returns_10)*100:.2f}%

前30支股票績效統計:
- 平均年報酬率: {np.mean(returns_30):.4f}%
- 報酬率標準差: {np.std(returns_30):.4f}%
- 最高年報酬率: {np.max(returns_30):.4f}%
- 最低年報酬率: {np.min(returns_30):.4f}%
- 正報酬年數: {sum(1 for r in returns_30 if r > 0)}/{len(returns_30)}
- 勝率: {sum(1 for r in returns_30 if r > 0)/len(returns_30)*100:.2f}%

SVR參數統計:
- C參數平均值: {np.mean(c_params):.4f}
- C參數標準差: {np.std(c_params):.4f}
- gamma參數平均值: {np.mean(gamma_params):.4f}
- gamma參數標準差: {np.std(gamma_params):.4f}
- epsilon參數平均值: {np.mean(epsilon_params):.4f}
- epsilon參數標準差: {np.std(epsilon_params):.4f}

年度詳細結果:
"""
        
        for result in self.results:
            report += f"""
{result['test_year']}年:
  - 前10支股票報酬率: {result['best_return']:.4f}%
  - 前30支股票報酬率: {result['best_return_30']:.4f}%
  - 最佳C: {result['best_C']:.4f}
  - 最佳gamma: {result['best_gamma']:.4f}
  - 最佳epsilon: {result['best_epsilon']:.4f}
  - 特徵數量: {result['num_features']}
  - GA適應度: {result['fitness']:.4f}
  - 主要特徵: {', '.join(result['best_features'][:5])}{'...' if len(result['best_features']) > 5 else ''}
"""
        
        best_year_idx_10 = np.argmax(returns_10)
        best_result = self.results[best_year_idx_10]
        
        report += f"""

最佳表現年份（前10支股票）: {best_result['test_year']}
  - 報酬率: {best_result['best_return']:.4f}%
  - 使用C: {best_result['best_C']:.4f}
  - 使用gamma: {best_result['best_gamma']:.4f}
  - 使用epsilon: {best_result['best_epsilon']:.4f}
  - 特徵數量: {best_result['num_features']}

風險指標:
- 前10支股票夏普比率: {np.mean(returns_10)/np.std(returns_10):.4f} (假設無風險利率為0)
- 前30支股票夏普比率: {np.mean(returns_30)/np.std(returns_30):.4f} (假設無風險利率為0)
- 前10支股票最大回撤: {self.calculate_max_drawdown(returns_10):.4f}%
- 前30支股票最大回撤: {self.calculate_max_drawdown(returns_30):.4f}%

GA-SVR方法優勢:
- 同時優化特徵選擇和SVR參數，避免局部最優解
- 使用遺傳演算法的全域搜索能力
- SVR的非線性建模能力適合複雜的股票市場
- 自動化參數調整，減少人為偏差
- 強健的回歸預測能力
- 能夠選出每年表現最佳的前10支股票

Paper 3 核心貢獻:
- 結合GA和SVR的混合方法，提供了有效的股票選擇策略
- 透過14個財務比率的綜合分析，捕捉股票的多維特徵
- 基於預測報酬率的排序選擇，提供了量化的投資決策依據
"""
        
        report_path = os.path.join(self.output_dir, 'paper3_ga_svr_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"分析報告已儲存至: {report_path}")
    
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
        
        returns_10 = [r['best_return'] for r in self.results]
        returns_30 = [r['best_return_30'] for r in self.results]
        
        print("\n" + "="*50)
        print("Paper 3 GA-SVR股票選股模型分析摘要")
        print("="*50)
        print(f"分析期間: {self.results[0]['train_year']}-{self.results[-1]['test_year']}")
        print(f"總測試年數: {len(self.results)}")
        
        print(f"\n前10支股票表現:")
        print(f"  平均年報酬率: {np.mean(returns_10):.4f}%")
        print(f"  報酬率標準差: {np.std(returns_10):.4f}%")
        print(f"  最高年報酬率: {np.max(returns_10):.4f}%")
        print(f"  最低年報酬率: {np.min(returns_10):.4f}%")
        print(f"  正報酬年數: {sum(1 for r in returns_10 if r > 0)}/{len(returns_10)}")
        
        print(f"\n前30支股票表現:")
        print(f"  平均年報酬率: {np.mean(returns_30):.4f}%")
        print(f"  報酬率標準差: {np.std(returns_30):.4f}%")
        print(f"  最高年報酬率: {np.max(returns_30):.4f}%")
        print(f"  最低年報酬率: {np.min(returns_30):.4f}%")
        print(f"  正報酬年數: {sum(1 for r in returns_30 if r > 0)}/{len(returns_30)}")
        
        best_year_idx = np.argmax(returns_10)
        best_result = self.results[best_year_idx]
        
        print(f"\n最佳表現年份: {best_result['test_year']}")
        print(f"  前10支股票報酬率: {best_result['best_return']:.4f}%")
        print(f"  前30支股票報酬率: {best_result['best_return_30']:.4f}%")
        print(f"  使用C: {best_result['best_C']:.4f}")
        print(f"  使用gamma: {best_result['best_gamma']:.4f}")
        print(f"  使用epsilon: {best_result['best_epsilon']:.4f}")
        print(f"  特徵數量: {best_result['num_features']}")

def main():
    selector = Paper3StockSelector('top200.xlsx', output_dir='Q4output')
    
    print("開始執行Paper 3 GA-SVR股票選股分析...")
    selector.run_rolling_window_analysis()
    
    selector.save_results_to_csv()
    selector.create_visualizations()
    selector.create_top10_analysis_chart()
    selector.create_parameter_analysis_chart()
    selector.create_feature_importance_chart()
    selector.save_analysis_report()
    selector.print_summary()
    
    print(f"\n所有結果已儲存至 Q4output 資料夾")
    
    return selector

if __name__ == "__main__":
    selector = main()
