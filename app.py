from flask import Flask, render_template, request, jsonify
import os
import pickle
import sys
import re
import jieba
import pandas as pd

from scraper import ScrapeError, scrape_dianping_comments

app = Flask(__name__)

# 加载模型和向量器
model = None
vectorizer = None

def load_model():
    global model, vectorizer
    if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return True
    return False

def ensure_model_loaded():
    """确保模型和向量器已加载"""
    if model is None or vectorizer is None:
        load_model()
    return model is not None and vectorizer is not None

def clean_text(text):
    """文本清洗：保持与训练阶段一致，只保留中文"""
    text = str(text)
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    return text

def comment_to_tokens(comment):
    clean_comment = clean_text(comment)
    tokens = list(jieba.cut(clean_comment))
    return ' '.join(tokens)

def predict_comments(comments, preview_limit=20):
    """批量预测评论，并返回整体统计和前若干条明细"""
    if not ensure_model_loaded():
        raise ValueError('模型未加载，请先训练模型')

    valid_comments = [str(comment).strip() for comment in comments if str(comment).strip()]
    if not valid_comments:
        raise ValueError('没有可分析的评论内容')

    tokens = [comment_to_tokens(comment) for comment in valid_comments]
    X = vectorizer.transform(tokens)
    predictions = model.predict(X)

    if not hasattr(model, 'predict_proba'):
        raise ValueError('当前模型不支持置信度输出，请重新训练逻辑回归或LightGBM模型')

    probabilities = model.predict_proba(X)[:, 1]
    details = []

    for index, (comment, prediction, probability) in enumerate(
        zip(valid_comments, predictions, probabilities), start=1
    ):
        result = '水军评论' if int(prediction) == 1 else '真实评论'
        if index <= preview_limit:
            details.append({
                'index': index,
                'comment': comment,
                'result': result,
                'probability': float(probability)
            })

    total = len(valid_comments)
    water_army_count = int(sum(int(item) for item in predictions))
    real_comment_count = total - water_army_count
    water_army_ratio = water_army_count / total if total else 0

    return {
        'total_comments': total,
        'water_army_count': water_army_count,
        'real_comment_count': real_comment_count,
        'water_army_ratio': float(water_army_ratio),
        'preview_limit': preview_limit,
        'results': details
    }

def read_comments_from_csv(file):
    """读取上传CSV中的评论列"""
    last_error = None
    data = None

    for encoding in ('utf-8-sig', 'utf-8', 'gbk'):
        try:
            file.stream.seek(0)
            data = pd.read_csv(file, encoding=encoding)
            break
        except Exception as exc:
            last_error = exc

    if data is None:
        raise ValueError(f'CSV读取失败: {last_error}')

    if 'cus_comment' in data.columns:
        comment_column = 'cus_comment'
    elif 'comment' in data.columns:
        comment_column = 'comment'
    else:
        raise ValueError('CSV文件必须包含cus_comment或comment列')

    comments = data[comment_column].dropna().astype(str).tolist()
    return comments

@app.route('/')
def index():
    model_loaded = load_model()
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(silent=True) or {}
    comment = payload.get('comment', '')
    if not comment:
        return jsonify({'error': '请输入评论内容'}), 400

    try:
        batch_result = predict_comments([comment], preview_limit=1)
        detail = batch_result['results'][0]
        return jsonify({
            'result': detail['result'],
            'probability': detail['probability'],
            'comment': comment
        })
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if 'file' not in request.files:
        return jsonify({'error': '请选择CSV文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '请选择CSV文件'}), 400

    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': '请上传CSV文件'}), 400

    try:
        comments = read_comments_from_csv(file)
        result = predict_comments(comments)
        result['message'] = 'CSV批量分析完成'
        return jsonify(result)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

@app.route('/analyze_shop_url', methods=['POST'])
def analyze_shop_url():
    payload = request.get_json(silent=True) or {}
    url = payload.get('url', '').strip()

    if not url:
        return jsonify({'error': '请输入大众点评店铺或评论页网址'}), 400

    try:
        scraped_data = scrape_dianping_comments(url)
        scraped_data.to_csv('scraped_comments.csv', index=False, encoding='utf-8-sig')
        result = predict_comments(scraped_data['cus_comment'].tolist())
        result['message'] = '网址评论分析完成'
        result['source'] = 'url'
        result['csv_file'] = 'scraped_comments.csv'
        return jsonify(result)
    except ScrapeError as exc:
        return jsonify({
            'error': str(exc),
            'fallback': '当前页面无法自动采集，请上传CSV文件分析。'
        }), 400
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

@app.route('/train', methods=['POST'])
def train():
    # 调用模型训练模块
    import subprocess
    result = subprocess.run([sys.executable, 'model_trainer.py'],
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        load_model()
        return jsonify({'message': '模型训练成功！'})
    else:
        return jsonify({'error': f'训练失败: {result.stderr}'}), 500

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': '请选择文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '请选择文件'}), 400
    
    if file and file.filename.endswith('.csv'):
        file.save('data.csv')
        return jsonify({'message': '文件上传成功！'})
    else:
        return jsonify({'error': '请上传CSV文件'}), 400

@app.route('/visualizations')
def get_visualizations():
    """获取可视化结果"""
    visualize_dir = 'static/visualizations'
    visualizations = []
    
    if os.path.exists(visualize_dir):
        for file in os.listdir(visualize_dir):
            if file.endswith('.png'):
                visualizations.append({
                    'name': file,
                    'url': f'/static/visualizations/{file}'
                })
    
    return jsonify(visualizations)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
