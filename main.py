import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import date
from typing import List, Optional

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from pydantic import BaseModel, HttpUrl

# --- 1. 配置 ---
# 加载 .env 文件中的环境变量
load_dotenv()

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 从环境变量获取缓存时间，如果没有则默认为12小时
CACHE_EXPIRE_SECONDS = int(os.getenv("CACHE_EXPIRE_SECONDS", 43200))


# --- 2. Pydantic 数据模型 ---
class Paper(BaseModel):
    """代表一篇论文信息的模型"""
    title: str
    summary: str
    arxiv_url: HttpUrl
    huggingface_url: HttpUrl
    authors: List[str]
    github_url: Optional[HttpUrl] = None
    upvotes: int = 0
    ai_summary: Optional[str] = None
    ai_keywords: List[str] = []


# --- 3. FastAPI 应用生命周期管理 (lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    管理应用的启动和关闭事件。
    """
    # --- 启动时执行 ---
    logger.info("Application is starting up...")
    # 初始化缓存
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    logger.info(f"Cache initialized with InMemoryBackend. Expire time: {CACHE_EXPIRE_SECONDS} seconds.")

    yield

    # --- 关闭时执行 ---
    logger.info("Application is shutting down...")


# --- 4. FastAPI 应用实例 ---
app = FastAPI(
    title="Hugging Face Daily Papers Scraper",
    description="爬取Hugging Face每日论文。",
    version="0.1.0",
    lifespan=lifespan
)


# --- 5. 辅助函数 ---
def get_author_names(authors_list: list) -> List[str]:
    """从作者对象列表中安全地提取作者姓名"""
    names = []
    if not isinstance(authors_list, list):
        return names
    for author in authors_list:
        if isinstance(author, dict) and author.get("name"):
            names.append(author["name"])
    return names


# --- 6. API 端点 ---
@app.get(
    "/papers",
    response_model=List[Paper],
    summary="获取指定日期的Hugging Face论文",
    description="默认获取当天论文。结果会被缓存以提高性能。日期格式: YYYY-MM-DD。"
)
@cache(expire=CACHE_EXPIRE_SECONDS)
async def get_daily_papers(request_date: Optional[str] = None):
    """
    爬取 Hugging Face Daily Papers。此函数的执行结果将被缓存。
    """
    if request_date:
        try:
            target_date = date.fromisoformat(request_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="日期格式无效，请使用 'YYYY-MM-DD' 格式。")
    else:
        target_date = date.today()

    date_str = target_date.isoformat()
    url = f"https://huggingface.co/papers/date/{date_str}"

    logger.info(f"开始为日期 {date_str} 获取论文，请求URL: {url}")
    logger.info("缓存未命中，正在执行实时爬取。")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=20.0, follow_redirects=True)
            response.raise_for_status()
    except httpx.RequestError as exc:
        logger.error(f"请求Hugging Face服务器失败: {exc}")
        raise HTTPException(status_code=503, detail=f"请求Hugging Face服务器失败: {exc}")
    except httpx.HTTPStatusError as exc:
        logger.error(f"Hugging Face服务器返回错误状态 {exc.response.status_code} for URL {url}")
        if exc.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"未找到日期 {date_str} 的论文页面。")
        else:
            raise HTTPException(status_code=exc.response.status_code, detail=f"Hugging Face服务器返回错误: {exc.response.text}")

    logger.info(f"成功获取页面内容，开始解析HTML...")

    soup = BeautifulSoup(response.text, 'lxml')
    data_div = soup.find('div', attrs={'data-target': 'DailyPapers'})

    if not data_div or 'data-props' not in data_div.attrs:
        logger.warning(f"在页面 {url} 上未找到 'DailyPapers' 数据块。可能当天没有论文或页面结构已更改。")
        return []

    try:
        props_data = json.loads(data_div['data-props'])
        papers_list = props_data.get('dailyPapers', [])
    except (json.JSONDecodeError, KeyError) as e:
        logger.exception("解析页面内嵌JSON数据时发生严重错误。")
        raise HTTPException(status_code=500, detail=f"解析页面内嵌JSON数据失败: {e}")

    results: List[Paper] = []
    for item in papers_list:
        paper_data = item.get('paper', {})
        paper_id = paper_data.get('id')
        title = paper_data.get('title')

        if not paper_id or not title:
            logger.warning(f"跳过一条不完整的论文记录: {item}")
            continue

        try:
            cleaned_title = ' '.join(title.split())
            cleaned_summary = ' '.join(paper_data.get('summary', '').split())

            paper_obj = Paper(
                title=cleaned_title,
                summary=cleaned_summary,
                arxiv_url=f"https://arxiv.org/abs/{paper_id}",
                huggingface_url=f"https://huggingface.co/papers/{paper_id}",
                authors=get_author_names(paper_data.get("authors", [])),
                github_url=paper_data.get("githubRepo"),
                upvotes=paper_data.get("upvotes", 0),
                ai_summary=paper_data.get("ai_summary"),
                ai_keywords=paper_data.get("ai_keywords", [])
            )
            results.append(paper_obj)
        except Exception as e:
            logger.error(f"处理论文 {paper_id} 时出错: {e}. Raw data: {paper_data}")
            continue

    logger.info(f"成功解析并处理了 {len(results)} 篇论文（日期: {date_str}）。")
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)