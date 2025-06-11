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
        self.backtest_results = []  # 新增：儲存回測結果
        
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

    def calculate_backtest_metrics(self):
        """計算TradingView風格的回測指標"""
        if not self.results:
            return None
        
        returns_top10 = [r['best_return'] for r in self.results]
        returns_top30 = [r['best_return_30'] for r in self.results]
        years = [r['test_year'] for r in self.results]
        
        # 計算累積淨值
        initial_capital = 100000  # 初始資金10萬
        cumulative_top10 = [initial_capital]
        cumulative_top30 = [initial_capital]
        
        for i, (ret_10, ret_30) in enumerate(zip(returns_top10, returns_top30)):
            cumulative_top10.append(cumulative_top10[-1] * (1 + ret_10/100))
            cumulative_top30.append(cumulative_top30[-1] * (1 + ret_30/100))
        
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
        metrics_top30 = calculate_metrics(returns_top30, cumulative_top30)
        
        self.backtest_results = {
            'years': years,
            'returns_top10': returns_top10,
            'returns_top30': returns_top30,
            'cumulative_top10': cumulative_top10[1:],
            'cumulative_top30': cumulative_top30[1:],
            'metrics_top10': metrics_top10,
            'metrics_top30': metrics_top30,
            'initial_capital': initial_capital
        }
        
        return self.backtest_results

    def create_tradingview_style_backtest_chart(self):
        """創建TradingView風格的回測圖表（獨立圖表）"""
        if not self.backtest_results:
            self.calculate_backtest_metrics()
        
        setup_chinese_font()
        
        # 創建獨立的TradingView風格回測圖表
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, height_ratios=[3, 1, 1, 1], width_ratios=[2, 1, 1])
        
        # 主圖：淨值曲線
        ax_main = fig.add_subplot(gs[0, :])
        years = self.backtest_results['years']
        cumulative_top10 = self.backtest_results['cumulative_top10']
        cumulative_top30 = self.backtest_results['cumulative_top30']
        
        ax_main.plot(years, cumulative_top10, linewidth=3, color='#2E8B57', label='前10支股票策略', marker='o', markersize=6)
        ax_main.plot(years, cumulative_top30, linewidth=2, color='#4169E1', label='前30支股票策略', marker='s', markersize=4)
        ax_main.axhline(y=self.backtest_results['initial_capital'], color='gray', linestyle='--', alpha=0.5, label='初始資金')
        
        ax_main.set_title('Paper 3 GA-SVR股票選股策略回測 - 淨值曲線', fontsize=18, fontweight='bold', pad=20)
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
        returns_top30 = self.backtest_results['returns_top30']
        
        x = np.arange(len(years))
        width = 0.35
        
        colors_top10 = ['#2E8B57' if r > 0 else '#DC143C' for r in returns_top10]
        colors_top30 = ['#4169E1' if r > 0 else '#FF6347' for r in returns_top30]
        
        ax_returns.bar(x - width/2, returns_top10, width, label='前10支股票', color=colors_top10, alpha=0.8)
        ax_returns.bar(x + width/2, returns_top30, width, label='前30支股票', color=colors_top30, alpha=0.8)
        
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
        
        # 子圖3：績效指標表格 - 前30支股票
        ax_metrics2 = fig.add_subplot(gs[2, 1])
        ax_metrics2.axis('off')
        
        metrics_top30 = self.backtest_results['metrics_top30']
        metrics_data2 = [
            ['總報酬率', f"{metrics_top30['total_return']:.2f}%"],
            ['年化報酬率', f"{metrics_top30['annual_return']:.2f}%"],
            ['波動率', f"{metrics_top30['volatility']:.2f}%"],
            ['夏普比率', f"{metrics_top30['sharpe_ratio']:.3f}"],
            ['最大回撤', f"{metrics_top30['max_drawdown']:.2f}%"],
            ['勝率', f"{metrics_top30['win_rate']:.1f}%"],
            ['盈利因子', f"{metrics_top30['profit_factor']:.2f}"],
            ['總交易次數', f"{metrics_top30['total_trades']}"]
        ]
        
        table2 = ax_metrics2.table(cellText=metrics_data2,
                                  colLabels=['指標', '前30支股票策略'],
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
        
        ax_metrics2.set_title('前30支股票策略績效', fontsize=12, fontweight='bold')
        
        # 子圖4：策略比較
        ax_comparison = fig.add_subplot(gs[2, 2])
        ax_comparison.axis('off')
        
        comparison_data = [
            ['策略', '前10支股票', '前30支股票'],
            ['總報酬率', f"{metrics_top10['total_return']:.2f}%", f"{metrics_top30['total_return']:.2f}%"],
            ['夏普比率', f"{metrics_top10['sharpe_ratio']:.3f}", f"{metrics_top30['sharpe_ratio']:.3f}"],
            ['最大回撤', f"{metrics_top10['max_drawdown']:.2f}%", f"{metrics_top30['max_drawdown']:.2f}%"],
            ['勝率', f"{metrics_top10['win_rate']:.1f}%", f"{metrics_top30['win_rate']:.1f}%"]
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
        drawdowns_top30 = calculate_drawdown_series(cumulative_top30)
        
        ax_drawdown.fill_between(years, drawdowns_top10, 0, alpha=0.6, color='#2E8B57', label='前10支股票策略')
        ax_drawdown.fill_between(years, drawdowns_top30, 0, alpha=0.4, color='#4169E1', label='前30支股票策略')
        
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
        cumulative_top30 = [self.backtest_results['initial_capital']] + self.backtest_results['cumulative_top30']
        
        for i, year in enumerate(['初始'] + self.backtest_results['years']):
            if i == 0:
                backtest_data.append({
                    '年份': year,
                    '前10支股票_年度報酬率(%)': 0,
                    '前10支股票_累積淨值': cumulative_top10[i],
                    '前30支股票_年度報酬率(%)': 0,
                    '前30支股票_累積淨值': cumulative_top30[i],
                    '前10支股票_回撤(%)': 0,
                    '前30支股票_回撤(%)': 0
                })
            else:
                # 計算回撤
                peak_top10 = max(cumulative_top10[:i+1])
                peak_top30 = max(cumulative_top30[:i+1])
                drawdown_top10 = (peak_top10 - cumulative_top10[i]) / peak_top10 * 100
                drawdown_top30 = (peak_top30 - cumulative_top30[i]) / peak_top30 * 100
                
                backtest_data.append({
                    '年份': year,
                    '前10支股票_年度報酬率(%)': self.backtest_results['returns_top10'][i-1],
                    '前10支股票_累積淨值': cumulative_top10[i],
                    '前30支股票_年度報酬率(%)': self.backtest_results['returns_top30'][i-1],
                    '前30支股票_累積淨值': cumulative_top30[i],
                    '前10支股票_回撤(%)': drawdown_top10,
                    '前30支股票_回撤(%)': drawdown_top30
                })
        
        # 儲存回測資料
        backtest_df = pd.DataFrame(backtest_data)
        backtest_path = os.path.join(self.output_dir, 'tradingview_backtest_data.csv')
        backtest_df.to_csv(backtest_path, index=False, encoding='utf-8-sig')
        
        # 儲存績效指標
        metrics_data = []
        metrics_top10 = self.backtest_results['metrics_top10']
        metrics_top30 = self.backtest_results['metrics_top30']
        
        for key in metrics_top10.keys():
            metrics_data.append({
                '指標': key,
                '前10支股票策略': metrics_top10[key],
                '前30支股票策略': metrics_top30[key]
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_path = os.path.join(self.output_dir, 'tradingview_backtest_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
        
        print(f"回測資料已匯出至:")
        print(f"- {backtest_path}")
        print(f"- {metrics_path}")

    def print_backtest_summary(self):
        """印出TradingView風格的回測摘要"""
        if not self.backtest_results:
            self.calculate_backtest_metrics()
        
        metrics_top10 = self.backtest_results['metrics_top10']
        metrics_top30 = self.backtest_results['metrics_top30']
        
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
        
        print(f"\n【前30支股票策略】")
        print(f"  總報酬率: {metrics_top30['total_return']:.2f}%")
        print(f"  年化報酬率: {metrics_top30['annual_return']:.2f}%")
        print(f"  夏普比率: {metrics_top30['sharpe_ratio']:.3f}")
        print(f"  最大回撤: {metrics_top30['max_drawdown']:.2f}%")
        print(f"  勝率: {metrics_top30['win_rate']:.1f}%")
        print(f"  盈利因子: {metrics_top30['profit_factor']:.2f}")
        print(f"  期末淨值: {self.backtest_results['cumulative_top30'][-1]:,.0f} 元")

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

    def create_top10_individual_returns_chart(self):
        """創建前10支股票各自的年化報酬折線圖"""
        if not self.results:
            return
        
        setup_chinese_font()
        
        # 收集前10支股票的個別報酬率資料
        stock_returns = {}
        years = []
        
        for result in self.results:
            if result['selected_stocks_top10'] is not None:
                year = result['test_year']
                years.append(year)
                stocks = result['selected_stocks_top10']
                
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
        individual_chart_path = os.path.join(self.output_dir, 'paper3_top10_individual_returns.png')
        plt.savefig(individual_chart_path, dpi=300, bbox_inches='tight')
        plt.show()

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

def main():
    selector = Paper3StockSelector('top200.xlsx', output_dir='Q4output')
    
    print("開始執行Paper 3 GA-SVR股票選股分析...")
    selector.run_rolling_window_analysis()
    
    selector.save_results_to_csv()
    selector.create_visualizations()
    selector.create_top10_individual_returns_chart()
    
    # 新增：TradingView風格回測（獨立圖表）
    print("\n開始TradingView風格回測分析...")
    selector.calculate_backtest_metrics()
    selector.create_tradingview_style_backtest_chart()
    selector.export_backtest_data_to_csv()
    selector.print_backtest_summary()
    
    print(f"\n所有結果已儲存至 Q4output 資料夾")
    
    return selector

if __name__ == "__main__":
    selector = main()
