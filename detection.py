# 文件名：water_army_demo_v3.py
# 功能：大众点评水军评论识别演示（优化弱标签生成 + 可视化）

import pandas as pd
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

# 1️⃣ 读取数据
data = pd.read_csv('D:\projects\data.csv')

# 2️⃣ 文本清洗
def clean_text(text):
    text = str(text)
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)  # 只保留中文
    return text

data['clean_text'] = data['cus_comment'].apply(clean_text)

# 3️⃣ 弱标签生成（优化规则）
def generate_label(row):
    text = row['clean_text']
    length = len(text)
    score_cols = ['kouwei', 'huanjing', 'fuwu']
    scores = []
    for col in score_cols:
        try:
            s = {'差':1,'一般':2,'好':3,'很好':4,'非常好':5}.get(str(row[col]), 4)
        except:
            s = 4
        scores.append(s)
    avg_score = np.mean(scores)
    
    # 高频刷评词（可自行扩展）
    spam_words = ['五星', '推荐', '棒', '很好', '极力推荐', '必去']
    
    if length < 15:  # 文本很短
        return 1
    elif avg_score >= 4.5 and length < 25:  # 高评分短评
        return 1
    elif any(w in text for w in spam_words):  # 包含刷评词
        return 1
    else:
        return 0

data['label'] = data.apply(generate_label, axis=1)

print("标签分布：")
print(data['label'].value_counts())

# 4️⃣ 分词处理
data['tokens'] = data['clean_text'].apply(lambda x: list(jieba.cut(x)))
data['tokens_str'] = data['tokens'].apply(lambda x: ' '.join(x))

# 5️⃣ TF-IDF 特征提取
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['tokens_str'])
y = data['label']

# 6️⃣ 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 7️⃣ 模型训练（逻辑回归）
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8️⃣ 预测与评估
y_pred = model.predict(X_test)
print("\n分类报告：")
print(classification_report(y_test, y_pred))

# 9️⃣ 可视化重要 TF-IDF 词
feature_names = np.array(vectorizer.get_feature_names_out())
coefficients = model.coef_[0]
top10_idx = np.argsort(coefficients)[-10:]
plt.figure(figsize=(8,5))
plt.barh(feature_names[top10_idx], coefficients[top10_idx])
plt.xlabel('TF-IDF Coefficient')
plt.title('Top 10 Features for Water Army Detection')
plt.tight_layout()
plt.show()

# 🔟 可视化词云
spam_text = ' '.join(data[data['label']==1]['tokens_str'])
real_text = ' '.join(data[data['label']==0]['tokens_str'])

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
WordCloud(font_path='simhei.ttf', width=600, height=400, background_color='white').generate(spam_text)
plt.imshow(WordCloud(font_path='simhei.ttf').generate(spam_text), interpolation='bilinear')
plt.axis('off')
plt.title('Water Army WordCloud')

plt.subplot(1,2,2)
WordCloud(font_path='simhei.ttf', width=600, height=400, background_color='white').generate(real_text)
plt.imshow(WordCloud(font_path='simhei.ttf').generate(real_text), interpolation='bilinear')
plt.axis('off')
plt.title('Real Comment WordCloud')
plt.tight_layout()
plt.show()

# 🔹 评论长度分布可视化
plt.figure(figsize=(8,5))
plt.hist([data[data['label']==1]['clean_text'].str.len(),
          data[data['label']==0]['clean_text'].str.len()],
         bins=30, label=['Water Army', 'Real Comment'], color=['red','blue'], alpha=0.7)
plt.xlabel('Comment Length')
plt.ylabel('Count')
plt.title('Comment Length Distribution')
plt.legend()
plt.show()

print("优化演示完成！")