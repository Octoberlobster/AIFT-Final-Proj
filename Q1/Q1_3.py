import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class StockKNNSelector:
    def __init__(self, data_path):
        """
        初始化股票KNN選擇器
        
        Args:
            data_path: Excel檔案路徑
        """
        self.data = pd.read_excel(data_path)
        self.feature_columns = [
            '市值(百萬元)', '收盤價(元)_年', 'Unknown masked parameter',
            '股價淨值比', '股價營收比', 'M淨值報酬率─稅後', '資產報酬率ROA',
            '營業利益率OPM', '利潤邊際NPM', '負債/淨值比', 'M流動比率',
            'M速動比率', 'M存貨週轉率 (次)', 'M應收帳款週轉次',
            'M營業利益成長率', 'M稅後淨利成長率'
        ]
        self.target_return = 'Return'
        self.target_label = 'ReturnMean_year_Label'
        self.scaler = StandardScaler()
        
    def prepare_data(self):
        """
        準備和清理數據
        """
        # 移除200912的數據
        self.data = self.data[self.data['年月'] != 200912]
        
        # 處理缺失值
        self.data = self.data.dropna(subset=self.feature_columns + [self.target_return, self.target_label])
        
        # 創建年份欄位
        self.data['年份'] = self.data['年月'] // 100
        
        print(f"數據準備完成，共有 {len(self.data)} 筆記錄")
        print(f"年份範圍: {self.data['年份'].min()} - {self.data['年份'].max()}")
        
    def get_year_data(self, year):
        """
        獲取特定年份的數據
        
        Args:
            year: 年份
            
        Returns:
            該年份的數據
        """
        return self.data[self.data['年份'] == year].copy()
    
    def select_features(self, X_train, y_train, X_test, max_features=8):
        """
        使用貪婪搜索選擇最佳特徵組合
        
        Args:
            X_train: 訓練特徵
            y_train: 訓練標籤
            X_test: 測試特徵
            max_features: 最大特徵數量
            
        Returns:
            最佳特徵索引列表
        """
        n_features = X_train.shape[1]
        best_features = []
        best_score = -np.inf
        
        # 貪婪搜索
        for i in range(min(max_features, n_features)):
            best_feature_this_round = None
            best_score_this_round = -np.inf
            
            for feature_idx in range(n_features):
                if feature_idx in best_features:
                    continue
                    
                current_features = best_features + [feature_idx]
                
                # 使用KNN回歸評估特徵組合
                knn = KNeighborsRegressor(n_neighbors=5)
                X_train_selected = X_train[:, current_features]
                X_test_selected = X_test[:, current_features]
                
                knn.fit(X_train_selected, y_train)
                y_pred = knn.predict(X_test_selected)
                
                # 計算平均預測回報率作為評估指標
                score = np.mean(y_pred)
                
                if score > best_score_this_round:
                    best_score_this_round = score
                    best_feature_this_round = feature_idx
            
            if best_feature_this_round is not None:
                best_features.append(best_feature_this_round)
                if best_score_this_round > best_score:
                    best_score = best_score_this_round
            else:
                break
                
        return best_features
    
    def find_best_k_and_features(self, train_year, test_year, k_range=range(3, 21, 2)):
        """
        尋找最佳的K值和特徵組合
        
        Args:
            train_year: 訓練年份
            test_year: 測試年份
            k_range: K值範圍
            
        Returns:
            最佳K值、最佳特徵、最佳回報率
        """
        train_data = self.get_year_data(train_year)
        test_data = self.get_year_data(test_year)
        
        if len(train_data) == 0 or len(test_data) == 0:
            return None, None, -np.inf
        
        X_train = train_data[self.feature_columns].values
        y_train_return = train_data[self.target_return].values
        y_train_label = train_data[self.target_label].values
        
        X_test = test_data[self.feature_columns].values
        y_test_return = test_data[self.target_return].values
        
        # 標準化特徵
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 選擇最佳特徵
        best_features = self.select_features(X_train_scaled, y_train_return, X_test_scaled)
        
        if not best_features:
            return None, None, -np.inf
        
        X_train_selected = X_train_scaled[:, best_features]
        X_test_selected = X_test_scaled[:, best_features]
        
        best_k = None
        best_return = -np.inf
        
        # 尋找最佳K值
        for k in k_range:
            if k >= len(X_train_selected):
                continue
                
            # 使用KNN分類器預測標籤（是否優於平均）
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            knn_classifier.fit(X_train_selected, y_train_label)
            predicted_labels = knn_classifier.predict(X_test_selected)
            
            # 選擇預測為正向的股票
            positive_indices = np.where(predicted_labels == 1)[0]
            
            if len(positive_indices) > 0:
                selected_returns = y_test_return[positive_indices]
                avg_return = np.mean(selected_returns)
                
                if avg_return > best_return:
                    best_return = avg_return
                    best_k = k
        
        return best_k, best_features, best_return
    
    def create_feature_matrix_csv(self, start_year=1997, end_year=2007, output_file='feature_selection_matrix.csv'):
        """
        創建特徵選擇矩陣CSV，16個指標為標題，選中的特徵顯示數值
        
        Args:
            start_year: 開始年份
            end_year: 結束年份
            output_file: 輸出檔案名稱
        """
        results = []
        
        for train_year in range(start_year, end_year + 1):
            for test_year in range(train_year + 1, end_year + 2):
                if test_year > 2008:  # 數據只到2008年
                    break
                    
                print(f"處理訓練年份: {train_year}, 測試年份: {test_year}")
                
                # 獲取最佳參數
                best_k, best_features, best_return = self.find_best_k_and_features(
                    train_year, test_year
                )
                
                if best_k is not None and best_features:
                    # 創建一行數據
                    row_data = {
                        'train_year': train_year,
                        'test_year': test_year,
                        'best_k': best_k,
                        'avg_return': best_return,
                        'n_features': len(best_features)
                    }
                    
                    # 為每個特徵創建欄位，選中的顯示重要性分數，未選中的顯示空值
                    train_data = self.get_year_data(train_year)
                    test_data = self.get_year_data(test_year)
                    
                    if len(train_data) > 0 and len(test_data) > 0:
                        # 計算特徵重要性（使用選中特徵在訓練數據中的標準差作為重要性指標）
                        X_train = train_data[self.feature_columns].values
                        X_train_scaled = self.scaler.fit_transform(X_train)
                        
                        for i, feature_name in enumerate(self.feature_columns):
                            if i in best_features:
                                # 計算該特徵的重要性分數（標準差）
                                importance_score = np.std(X_train_scaled[:, i])
                                row_data[feature_name] = round(importance_score, 4)
                            else:
                                row_data[feature_name] = np.nan  # 未選中的特徵顯示空值
                    
                    results.append(row_data)
        
        # 創建DataFrame
        if results:
            df_results = pd.DataFrame(results)
            
            # 重新排列欄位順序：基本信息 + 16個特徵
            basic_columns = ['train_year', 'test_year', 'best_k', 'avg_return', 'n_features']
            feature_columns = self.feature_columns
            column_order = basic_columns + feature_columns
            
            df_results = df_results[column_order]
            
            # 輸出為CSV
            df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n特徵選擇矩陣已輸出至 {output_file}")
            
            # 顯示結果預覽
            print("\n特徵選擇矩陣預覽:")
            print("基本信息:")
            print(df_results[basic_columns].head(10))
            
            print("\n特徵選擇情況（前5個特徵）:")
            print(df_results[basic_columns + feature_columns[:5]].head(10))
            
            # 統計每個特徵被選中的次數
            feature_selection_count = {}
            for feature in self.feature_columns:
                count = df_results[feature].notna().sum()
                feature_selection_count[feature] = count
            
            print("\n各特徵被選中次數統計:")
            for feature, count in sorted(feature_selection_count.items(), key=lambda x: x[1], reverse=True):
                print(f"{feature}: {count}次")
            
            return df_results
        else:
            print("沒有找到有效的結果")
            return None

    def create_detailed_feature_analysis(self, start_year=1997, end_year=2007, output_file='detailed_feature_analysis.csv'):
        """
        創建詳細的特徵分析報告
        
        Args:
            start_year: 開始年份
            end_year: 結束年份
            output_file: 輸出檔案名稱
        """
        results = []
        
        for train_year in range(start_year, end_year + 1):
            for test_year in range(train_year + 1, end_year + 2):
                if test_year > 2008:
                    break
                    
                best_k, best_features, best_return = self.find_best_k_and_features(
                    train_year, test_year
                )
                
                if best_k is not None and best_features:
                    train_data = self.get_year_data(train_year)
                    
                    if len(train_data) > 0:
                        X_train = train_data[self.feature_columns].values
                        X_train_scaled = self.scaler.fit_transform(X_train)
                        
                        # 為每個選中的特徵創建詳細記錄
                        for feature_idx in best_features:
                            feature_name = self.feature_columns[feature_idx]
                            feature_values = X_train_scaled[:, feature_idx]
                            
                            results.append({
                                'train_year': train_year,
                                'test_year': test_year,
                                'best_k': best_k,
                                'avg_return': best_return,
                                'feature_name': feature_name,
                                'feature_index': feature_idx,
                                'importance_score': np.std(feature_values),
                                'mean_value': np.mean(feature_values),
                                'median_value': np.median(feature_values),
                                'min_value': np.min(feature_values),
                                'max_value': np.max(feature_values)
                            })
        
        if results:
            df_detailed = pd.DataFrame(results)
            df_detailed.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n詳細特徵分析已輸出至 {output_file}")
            
            return df_detailed
        else:
            print("沒有找到有效的結果")
            return None

    def create_visualizations(self, feature_matrix_df, detailed_analysis_df):
        """
        創建各種圖表來呈現特徵選擇結果
        
        Args:
            feature_matrix_df: 特徵選擇矩陣DataFrame
            detailed_analysis_df: 詳細分析DataFrame
        """
        # 設定中文字體
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 特徵被選中次數的長條圖
        self.plot_feature_selection_frequency(feature_matrix_df)
        
        # 2. 年度平均回報率趨勢圖
        self.plot_annual_return_trend(feature_matrix_df)
        
        # 3. K值分布圖
        self.plot_k_value_distribution(feature_matrix_df)
        
        # 4. 特徵重要性熱力圖
        self.plot_feature_importance_heatmap(feature_matrix_df)
        
        # 5. 互動式特徵選擇矩陣
        self.create_interactive_feature_matrix(feature_matrix_df)

    def plot_feature_selection_frequency(self, df):
        """繪製特徵被選中次數的長條圖"""
        feature_counts = {}
        for feature in self.feature_columns:
            count = df[feature].notna().sum()
            feature_counts[feature] = count
        
        # 排序
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        features, counts = zip(*sorted_features)
        
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(features)), counts, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.title('各特徵被選中次數統計', fontsize=16, fontweight='bold')
        plt.xlabel('特徵名稱', fontsize=12)
        plt.ylabel('被選中次數', fontsize=12)
        plt.xticks(range(len(features)), features, rotation=45, ha='right')
        
        # 在長條上顯示數值
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('feature_selection_frequency.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_annual_return_trend(self, df):
        """繪製年度平均回報率趨勢圖"""
        annual_returns = df.groupby('test_year')['avg_return'].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        plt.plot(annual_returns['test_year'], annual_returns['avg_return'], 
                 marker='o', linewidth=2, markersize=8, color='green')
        plt.title('年度平均回報率趨勢', fontsize=16, fontweight='bold')
        plt.xlabel('測試年份', fontsize=12)
        plt.ylabel('平均回報率', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加數值標籤
        for x, y in zip(annual_returns['test_year'], annual_returns['avg_return']):
            plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig('annual_return_trend.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_k_value_distribution(self, df):
        """繪製K值分布圖"""
        plt.figure(figsize=(10, 6))
        
        # 直方圖
        plt.subplot(1, 2, 1)
        plt.hist(df['best_k'], bins=range(int(df['best_k'].min()), 
                                         int(df['best_k'].max()) + 2), 
                 color='lightcoral', edgecolor='black', alpha=0.7)
        plt.title('最佳K值分布', fontsize=14, fontweight='bold')
        plt.xlabel('K值', fontsize=12)
        plt.ylabel('頻率', fontsize=12)
        
        # 箱型圖
        plt.subplot(1, 2, 2)
        plt.boxplot(df['best_k'], vert=True)
        plt.title('最佳K值箱型圖', fontsize=14, fontweight='bold')
        plt.ylabel('K值', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('k_value_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance_heatmap(self, df):
        """繪製特徵重要性熱力圖"""
        # 準備熱力圖數據
        heatmap_data = df[self.feature_columns].fillna(0)
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(heatmap_data.T, cmap='YlOrRd', cbar=True, 
                    xticklabels=[f"{row['train_year']}-{row['test_year']}" 
                               for _, row in df.iterrows()],
                    yticklabels=self.feature_columns,
                    linewidths=0.5)
        plt.title('特徵重要性熱力圖', fontsize=16, fontweight='bold')
        plt.xlabel('訓練-測試年份組合', fontsize=12)
        plt.ylabel('特徵名稱', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_interactive_feature_matrix(self, df):
        """創建互動式特徵選擇矩陣"""
        # 準備數據
        feature_data = df[self.feature_columns].fillna(0)
        
        # 創建互動式熱力圖
        fig = go.Figure(data=go.Heatmap(
            z=feature_data.T.values,
            x=[f"{row['train_year']}-{row['test_year']}" for _, row in df.iterrows()],
            y=self.feature_columns,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='年份組合: %{x}<br>特徵: %{y}<br>重要性: %{z:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='互動式特徵選擇矩陣',
            xaxis_title='訓練-測試年份組合',
            yaxis_title='特徵名稱',
            width=1200,
            height=800
        )
        
        fig.write_html('interactive_feature_matrix.html')
        fig.show()

    def create_comprehensive_dashboard(self, df):
        """創建綜合儀表板"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('特徵選擇頻率', '年度回報率趨勢', 'K值分布', '特徵數量分布'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "histogram"}]]
        )
        
        # 特徵選擇頻率
        feature_counts = {feature: df[feature].notna().sum() for feature in self.feature_columns}
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        features, counts = zip(*sorted_features[:10])  # 只顯示前10個
        
        fig.add_trace(
            go.Bar(x=list(features), y=list(counts), name="選擇頻率"),
            row=1, col=1
        )
        
        # 年度回報率趨勢
        annual_returns = df.groupby('test_year')['avg_return'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=annual_returns['test_year'], y=annual_returns['avg_return'],
                      mode='lines+markers', name="平均回報率"),
            row=1, col=2
        )
        
        # K值分布
        fig.add_trace(
            go.Histogram(x=df['best_k'], name="K值分布"),
            row=2, col=1
        )
        
        # 特徵數量分布
        fig.add_trace(
            go.Histogram(x=df['n_features'], name="特徵數量分布"),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="股票特徵選擇綜合分析儀表板")
        fig.write_html('comprehensive_dashboard.html')
        fig.show()

# 使用範例
def main():
    # 初始化模型
    selector = StockKNNSelector('top200.xlsx')
    
    # 準備數據
    selector.prepare_data()
    
    # 創建特徵選擇矩陣（16個指標為標題）
    feature_matrix = selector.create_feature_matrix_csv(
        start_year=1997, 
        end_year=2006, 
        output_file='feature_selection_matrix.csv'
    )
    
    # 創建詳細特徵分析
    detailed_analysis = selector.create_detailed_feature_analysis(
        start_year=1997, 
        end_year=2006, 
        output_file='detailed_feature_analysis.csv'
    )
    
    # 新增：創建視覺化圖表
    if feature_matrix is not None:
        print("\n=== 開始創建視覺化圖表 ===")
        selector.create_visualizations(feature_matrix, detailed_analysis)
        selector.create_comprehensive_dashboard(feature_matrix)
        
        print("\n=== 圖表創建完成 ===")
        print("已生成以下圖表檔案:")
        print("1. feature_selection_frequency.png - 特徵選擇頻率長條圖")
        print("2. annual_return_trend.png - 年度回報率趨勢圖")
        print("3. k_value_distribution.png - K值分布圖")
        print("4. feature_importance_heatmap.png - 特徵重要性熱力圖")
        print("5. interactive_feature_matrix.html - 互動式特徵矩陣")
        print("6. comprehensive_dashboard.html - 綜合分析儀表板")
    
    print("\n=== 處理完成 ===")
    print("已生成以下檔案:")
    print("1. feature_selection_matrix.csv - 16個指標為標題的特徵選擇矩陣")
    print("2. detailed_feature_analysis.csv - 詳細的特徵分析報告")

if __name__ == "__main__":
    main()
