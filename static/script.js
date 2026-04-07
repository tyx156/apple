document.addEventListener('DOMContentLoaded', function() {
    function escapeHtml(value) {
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function renderBatchResult(data) {
        const ratio = (data.water_army_ratio * 100).toFixed(2);
        const rows = data.results.map(item => {
            const probability = (item.probability * 100).toFixed(2);
            const badgeClass = item.result === '水军评论' ? 'bg-danger' : 'bg-success';
            return `
                <tr>
                    <td>${item.index}</td>
                    <td>${escapeHtml(item.comment)}</td>
                    <td><span class="badge ${badgeClass}">${item.result}</span></td>
                    <td>${probability}%</td>
                </tr>
            `;
        }).join('');

        const previewTip = data.total_comments > data.preview_limit
            ? `<p class="text-muted mb-0">仅展示前${data.preview_limit}条明细，统计结果基于全部评论。</p>`
            : '';

        return `
            <div class="alert alert-info">
                <h5>${data.message || '分析完成'}</h5>
                <p><strong>总评论数:</strong> ${data.total_comments}</p>
                <p><strong>疑似水军评论:</strong> ${data.water_army_count}</p>
                <p><strong>真实评论:</strong> ${data.real_comment_count}</p>
                <p><strong>疑似水军率:</strong> ${ratio}%</p>
                ${previewTip}
            </div>
            <div class="table-responsive">
                <table class="table table-sm table-bordered align-middle">
                    <thead>
                        <tr>
                            <th>序号</th>
                            <th>评论内容</th>
                            <th>识别结果</th>
                            <th>置信度</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        `;
    }

    // 单个评论预测
    const predictForm = document.getElementById('predict-form');
    const predictResult = document.getElementById('predict-result');
    
    predictForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const comment = document.getElementById('comment').value.trim();
        
        if (!comment) {
            predictResult.innerHTML = '<div class="alert alert-danger">请输入评论内容</div>';
            return;
        }
        
        predictResult.innerHTML = '<div class="alert alert-info">分析中...</div>';
        
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ comment: comment })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                predictResult.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            } else {
                const probability = (data.probability * 100).toFixed(2);
                const resultClass = data.result === '水军评论' ? 'alert-danger' : 'alert-success';
                predictResult.innerHTML = `
                    <div class="alert ${resultClass}">
                        <h5>分析结果</h5>
                        <p><strong>评论内容:</strong> ${data.comment}</p>
                        <p><strong>识别结果:</strong> ${data.result}</p>
                        <p><strong>置信度:</strong> ${probability}%</p>
                    </div>
                `;
            }
        })
        .catch(error => {
            predictResult.innerHTML = '<div class="alert alert-danger">分析失败，请稍后重试</div>';
            console.error('Error:', error);
        });
    });

    // 店铺网址分析
    const urlForm = document.getElementById('url-form');
    const urlResult = document.getElementById('url-result');

    urlForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const url = document.getElementById('shop-url').value.trim();

        if (!url) {
            urlResult.innerHTML = '<div class="alert alert-danger">请输入大众点评店铺或评论页网址</div>';
            return;
        }

        urlResult.innerHTML = '<div class="alert alert-info">正在尝试采集公开评论并分析...</div>';

        fetch('/analyze_shop_url', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url: url })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                const fallback = data.fallback ? `<p class="mb-0">${data.fallback}</p>` : '';
                urlResult.innerHTML = `<div class="alert alert-warning"><p>${data.error}</p>${fallback}</div>`;
            } else {
                urlResult.innerHTML = renderBatchResult(data);
            }
        })
        .catch(error => {
            urlResult.innerHTML = '<div class="alert alert-danger">网址分析失败，请上传CSV文件分析</div>';
            console.error('Error:', error);
        });
    });
    
    // 文件上传
    const uploadForm = document.getElementById('upload-form');
    const uploadResult = document.getElementById('upload-result');
    
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const fileInput = document.getElementById('file');
        const file = fileInput.files[0];
        
        if (!file) {
            uploadResult.innerHTML = '<div class="alert alert-danger">请选择文件</div>';
            return;
        }
        
        uploadResult.innerHTML = '<div class="alert alert-info">CSV分析中...</div>';
        
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/batch_predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                uploadResult.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            } else {
                uploadResult.innerHTML = renderBatchResult(data);
            }
        })
        .catch(error => {
            uploadResult.innerHTML = '<div class="alert alert-danger">CSV分析失败，请稍后重试</div>';
            console.error('Error:', error);
        });
    });
    
    // 模型训练
    const trainBtn = document.getElementById('train-btn');
    const trainResult = document.getElementById('train-result');
    
    function trainModel() {
        trainBtn.disabled = true;
        trainResult.innerHTML = '<div class="alert alert-info">模型训练中，请耐心等待...</div>';
        
        fetch('/train', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                trainResult.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            } else {
                trainResult.innerHTML = `<div class="alert alert-success">${data.message}</div><p class="mt-2 text-info">系统将在5秒后自动刷新页面...</p>`;
                // 训练成功后刷新页面
                setTimeout(() => {
                    window.location.reload();
                }, 5000);
            }
        })
        .catch(error => {
            trainResult.innerHTML = '<div class="alert alert-danger">训练失败，请稍后重试</div>';
            console.error('Error:', error);
        })
        .finally(() => {
            trainBtn.disabled = false;
        });
    }
    
    trainBtn.addEventListener('click', trainModel);
    
    // 加载可视化结果
    const loadVisualizationsBtn = document.getElementById('load-visualizations');
    const visualizationContainer = document.getElementById('visualization-container');
    
    loadVisualizationsBtn.addEventListener('click', function() {
        visualizationContainer.innerHTML = '<div class="alert alert-info">加载中...</div>';
        
        fetch('/visualizations')
            .then(response => response.json())
            .then(data => {
                if (data.length === 0) {
                    visualizationContainer.innerHTML = '<div class="alert alert-warning">暂无可视化结果，请先训练模型</div>';
                    return;
                }
                
                visualizationContainer.innerHTML = '';
                
                // 为每种可视化结果创建卡片
                data.forEach(visualization => {
                    const card = document.createElement('div');
                    card.className = 'card mb-3';
                    
                    // 根据文件名生成标题
                    let title = '可视化结果';
                    if (visualization.name.includes('water_army_wordcloud')) {
                        title = '水军评论词云';
                    } else if (visualization.name.includes('real_wordcloud')) {
                        title = '真实评论词云';
                    } else if (visualization.name.includes('feature_importance')) {
                        title = '特征重要性';
                    } else if (visualization.name.includes('confusion_matrix')) {
                        title = '混淆矩阵';
                    } else if (visualization.name.includes('accuracy')) {
                        title = '模型准确率';
                    }
                    
                    card.innerHTML = `
                        <div class="card-header">
                            <h5 class="card-title">${title}</h5>
                        </div>
                        <div class="card-body text-center">
                            <img src="${visualization.url}" class="img-fluid rounded" alt="${title}">
                        </div>
                    `;
                    
                    visualizationContainer.appendChild(card);
                });
            })
            .catch(error => {
                visualizationContainer.innerHTML = '<div class="alert alert-danger">加载失败，请稍后重试</div>';
                console.error('Error:', error);
            });
    });
});
