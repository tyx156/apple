import pickle
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import lightgbm as lgb
import numpy as np
from data_processor import load_data, preprocess_data, extract_features, get_data_statistics

def get_chinese_font_path():
    """查找常见中文字体，优先使用项目内字体。"""
    candidate_paths = [
        os.path.join('static', 'fonts', 'simhei.ttf'),
        os.path.join('static', 'fonts', 'SimHei.ttf'),
        'simhei.ttf',
        r'C:\Windows\Fonts\simhei.ttf',
        r'C:\Windows\Fonts\msyh.ttc',
        r'C:\Windows\Fonts\simsun.ttc',
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/arphic/SimHei.ttf',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None

def train_model():
    """训练模型"""
    # 加载数据
    data = load_data()
    if data is None:
        print("数据加载失败，无法训练模型")
        return False
    
    # 数据预处理
    data = preprocess_data(data)
    if data is None:
        print("数据预处理失败")
        return False
    
    # 特征提取
    X, y, vectorizer = extract_features(data)
    if X is None:
        print("特征提取失败")
        return False
    
    # 数据统计信息
    stats = get_data_statistics(data)
    print("数据统计信息：")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    
    # 训练逻辑回归模型
    print("\n训练逻辑回归模型...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # 训练LightGBM模型
    print("\n训练LightGBM模型...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5,
        random_state=42
    )
    lgb_model.fit(X_train, y_train)
    
    # 评估逻辑回归模型
    print("\n逻辑回归模型评估：")
    lr_y_pred = lr_model.predict(X_test)
    print(f"准确率: {accuracy_score(y_test, lr_y_pred):.4f}")
    print("分类报告：")
    print(classification_report(y_test, lr_y_pred))
    
    # 评估LightGBM模型
    print("\nLightGBM模型评估：")
    lgb_y_pred = lgb_model.predict(X_test)
    print(f"准确率: {accuracy_score(y_test, lgb_y_pred):.4f}")
    print("分类报告：")
    print(classification_report(y_test, lgb_y_pred))
    
    # 选择性能更好的模型
    lr_acc = accuracy_score(y_test, lr_y_pred)
    lgb_acc = accuracy_score(y_test, lgb_y_pred)
    
    if lgb_acc > lr_acc:
        best_model = lgb_model
        print("\n选择LightGBM模型作为最佳模型")
    else:
        best_model = lr_model
        print("\n选择逻辑回归模型作为最佳模型")
    
    # 生成可视化结果
    print("\n生成可视化结果...")
    visualize_dir = 'static/visualizations'
    generate_wordcloud(data, visualize_dir)
    generate_feature_importance(best_model, vectorizer, visualize_dir)
    generate_model_performance(y_test, best_model.predict(X_test), visualize_dir)
    print("可视化结果生成完成！")
    
    # 保存模型和向量器
    try:
        with open('model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        print("\n模型和向量器保存成功！")
        return True
    except Exception as e:
        print(f"模型保存失败: {e}")
        return False

def generate_wordcloud(data, save_path):
    """生成词云"""
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)

    font_path = get_chinese_font_path()
    if not font_path:
        print("未找到可用中文字体，跳过词云生成。可将simhei.ttf放到static/fonts目录。")
        return
    
    # 准备词云数据
    comment_col = 'cus_comment' if 'cus_comment' in data.columns else 'comment'
    water_army_text = ' '.join(data[data['label'] == 1][comment_col].dropna().astype(str).tolist())
    real_text = ' '.join(data[data['label'] == 0][comment_col].dropna().astype(str).tolist())
    
    try:
        # 生成水军评论词云
        if water_army_text:
            wc = WordCloud(
                font_path=font_path,
                width=800, height=400,
                background_color='white',
                max_words=200
            )
            wc.generate(water_army_text)
            wc.to_file(os.path.join(save_path, 'water_army_wordcloud.png'))
        
        # 生成真实评论词云
        if real_text:
            wc = WordCloud(
                font_path=font_path,
                width=800, height=400,
                background_color='white',
                max_words=200
            )
            wc.generate(real_text)
            wc.to_file(os.path.join(save_path, 'real_wordcloud.png'))
    except Exception as e:
        print(f"生成词云失败: {e}")

def generate_feature_importance(model, vectorizer, save_path):
    """生成特征重要性图表"""
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # 获取中文字体路径
        font_path = get_chinese_font_path()
        if font_path:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return
        
        # 获取特征名称
        feature_names = vectorizer.get_feature_names_out()
        
        # 排序并取前20个重要特征
        indices = np.argsort(importances)[-20:]
        feature_names = feature_names[indices]
        importances = importances[indices]
        
        # 生成图表
        plt.figure(figsize=(12, 8))
        plt.title('前20个重要特征 (Top 20 Feature Importance)')
        plt.barh(range(len(importances)), importances, align='center')
        plt.yticks(range(len(importances)), feature_names)
        plt.xlabel('重要性 (Importance)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'feature_importance.png'), dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"生成特征重要性图表失败: {e}")

def generate_model_performance(y_test, y_pred, save_path):
    """生成模型性能指标图表"""
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # 获取中文字体路径
        font_path = get_chinese_font_path()
        if font_path:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 计算性能指标
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # 生成混淆矩阵图表
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵 (Confusion Matrix)')
        plt.colorbar()
        tick_marks = [0, 1]
        plt.xticks(tick_marks, ['真实评论', '水军评论'])
        plt.yticks(tick_marks, ['真实评论', '水军评论'])
        
        # 添加数值标签
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha='center', va='center')
        
        plt.tight_layout()
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
        plt.close()
        
        # 生成准确率图表
        plt.figure(figsize=(8, 6))
        plt.bar(['准确率'], [accuracy])
        plt.title('模型准确率 (Model Accuracy)')
        plt.ylim(0, 1)
        plt.ylabel('准确率')
        plt.text(0, accuracy + 0.02, f'{accuracy:.4f}', ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'accuracy.png'))
        plt.close()
    except Exception as e:
        print(f"生成模型性能指标图表失败: {e}")

if __name__ == '__main__':
    train_model()
