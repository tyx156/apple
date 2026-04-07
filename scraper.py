import re
from datetime import datetime
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd


class ScrapeError(Exception):
    """网址评论采集失败。"""


BLOCK_KEYWORDS = [
    '验证中心',
    '身份核实',
    '登录',
    '验证码',
    'verify.meituan.com',
    'spiderindefence',
    '请进行验证',
]

COMMENT_SELECTORS = [
    '.reviews-items .review-words',
    '.reviews-items .review-words-Hide',
    '.comment-list .content',
    '.comment-list .comment',
    '.review-list .review-words',
    '.main-review .review-words',
    '[class*="review-words"]',
]


def validate_dianping_url(url):
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if parsed.scheme not in {'http', 'https'} or not host:
        raise ScrapeError('请输入完整的大众点评网址，例如 https://www.dianping.com/...')
    if host != 'dianping.com' and not host.endswith('.dianping.com'):
        raise ScrapeError('目前仅支持大众点评 dianping.com 网址')


def fetch_html(url):
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0 Safari/537.36'
        ),
        'Accept-Language': 'zh-CN,zh;q=0.9',
    }
    request = Request(url, headers=headers)

    try:
        with urlopen(request, timeout=10) as response:
            charset = response.headers.get_content_charset() or 'utf-8'
            final_url = response.geturl()
            html = response.read().decode(charset, errors='ignore')
    except HTTPError as exc:
        raise ScrapeError(f'页面请求失败，HTTP状态码: {exc.code}')
    except URLError as exc:
        raise ScrapeError(f'页面请求失败: {exc.reason}')
    except Exception as exc:
        raise ScrapeError(f'页面请求失败: {exc}')

    if any(keyword in final_url or keyword in html for keyword in BLOCK_KEYWORDS):
        raise ScrapeError('当前页面需要验证或登录，无法自动采集')

    return html


def normalize_comment(text):
    return re.sub(r'\s+', ' ', text or '').strip()


def extract_comments(html):
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ScrapeError('缺少BeautifulSoup依赖，无法自动解析网页，请上传CSV文件分析') from exc

    soup = BeautifulSoup(html, 'html.parser')
    comments = []

    for selector in COMMENT_SELECTORS:
        for node in soup.select(selector):
            text = normalize_comment(node.get_text(' ', strip=True))
            if len(text) >= 5 and text not in comments:
                comments.append(text)

    if not comments:
        raise ScrapeError('未从页面中采集到评论内容')

    return comments


def build_comment_dataframe(comments, shop_id=''):
    now = datetime.now()
    rows = []

    for index, comment in enumerate(comments, start=1):
        rows.append({
            'cus_id': f'scraped_{index}',
            'comment_time': '',
            'comment_star': '',
            'cus_comment': comment,
            'kouwei': '',
            'huanjing': '',
            'fuwu': '',
            'shopID': shop_id,
            'stars': '',
            'year': now.year,
            'month': now.month,
            'weekday': now.weekday(),
            'hour': now.hour,
            'comment_len': len(comment),
        })

    return pd.DataFrame(rows)


def scrape_dianping_comments(url):
    validate_dianping_url(url)
    html = fetch_html(url)
    comments = extract_comments(html)
    parsed = urlparse(url)
    shop_id_match = re.search(r'/shop/([^/?#]+)', parsed.path)
    shop_id = shop_id_match.group(1) if shop_id_match else ''
    return build_comment_dataframe(comments, shop_id=shop_id)
