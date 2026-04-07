import pandas as pd
import jieba
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(file_path='data.csv'):
    """加载数据"""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

def clean_text(text):
    """文本清洗"""
    text = str(text)
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)  # 只保留中文
    return text

def generate_label(row):
    """弱标签生成"""
    text = row['clean_text']
    length = len(text)
    
    # 尝试获取评分列
    score_cols = ['kouwei', 'huanjing', 'fuwu', 'score', 'rating']
    scores = []
    for col in score_cols:
        if col in row:
            try:
                s = {'差':1,'一般':2,'好':3,'很好':4,'非常好':5}.get(str(row[col]), 4)
                scores.append(s)
            except:
                pass
    
    avg_score = np.mean(scores) if scores else 4
    
    # 高频刷评词
    spam_words = ['五星', '推荐', '棒', '很好', '极力推荐', '必去', '超级好', '服务周到', '环境优美', '味道好']
    
    if length < 15:  # 文本很短
        return 1
    elif avg_score >= 4.5 and length < 25:  # 高评分短评
        return 1
    elif any(w in text for w in spam_words):  # 包含刷评词
        return 1
    else:
        return 0

def preprocess_data(data):
    """数据预处理"""
    if data is None:
        return None, None, None
    
    # 文本清洗
    data['clean_text'] = data['cus_comment'] if 'cus_comment' in data.columns else data['comment']
    data['clean_text'] = data['clean_text'].apply(clean_text)
    
    # 生成标签
    data['label'] = data.apply(generate_label, axis=1)
    
    # 分词处理
    data['tokens'] = data['clean_text'].apply(lambda x: list(jieba.cut(x)))
    data['tokens_str'] = data['tokens'].apply(lambda x: ' '.join(x))
    
    return data

def extract_features(data, max_features=5000):
    """特征提取"""
    if data is None:
        return None, None, None
    
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(data['tokens_str'])
    y = data['label']
    
    return X, y, vectorizer

def get_data_statistics(data):
    """获取数据统计信息"""
    if data is None:
        return {}
    
    stats = {
        'total_comments': len(data),
        'water_army_count': int(data['label'].sum()),
        'real_comment_count': int(len(data) - data['label'].sum()),
        'water_army_ratio': float(data['label'].mean()),
        'average_length': float(data['clean_text'].str.len().mean())
    }
    return stats