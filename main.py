import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import date
from typing import List, Optional

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from pydantic import BaseModel, HttpUrl
from openai import AsyncOpenAI, OpenAIError

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

# --- OpenAI 配置 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# 全局 OpenAI 客户端
openai_client: Optional[AsyncOpenAI] = None

# --- LLM Prompt Template ---
PROMPT_TEMPLATE = """
你是一名顶级的AI研究员助手。你的任务是基于论文的标题和摘要，提供一份简洁而深刻的中文解读。

**论文标题：** {title}
**论文摘要：** {summary}

请严格根据以上信息，生成一份包含以下两个部分的Markdown格式解读：
1.  **核心贡献 (Contribution):** 清晰地阐述这篇论文最主要的创新点或发现。它解决了什么新问题，或者提出了什么新方法？
2.  **方法简介 (Method):** 简要介绍论文中使用的核心方法、模型架构或技术路径。请尽量使用易于理解的语言，避免不必要的术语。

请确保你的解读内容完全基于所提供的标题和摘要，并使用流畅的中文进行表述。
"""


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

class PaperList(BaseModel):
    """用于接收论文列表的请求体模型"""
    papers: List[Paper]

class MarkdownResponse(BaseModel):
    """用于返回Markdown报告的响应体模型"""
    markdown_report: str


# --- 3. FastAPI 应用生命周期管理 (lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global openai_client
    logger.info("Application is starting up...")
    
    # 初始化缓存
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    logger.info(f"Cache initialized. Expire time: {CACHE_EXPIRE_SECONDS} seconds.")

    # 初始化 OpenAI 客户端
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not found. The report generation endpoint will not work.")
    else:
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        logger.info(f"OpenAI client initialized for model: {OPENAI_MODEL_NAME}")
    
    yield
    
    logger.info("Application is shutting down...")
    if openai_client:
        await openai_client.close()


# --- 4. FastAPI 应用实例 ---
app = FastAPI(
    title="AI每日论文解读服务",
    description="一个API服务，用于一键爬取Hugging Face每日论文并生成LLM中文解读报告。",
    version="4.0.0-final",
    lifespan=lifespan
)

# --- 5. 核心逻辑函数 (解耦&可复用) ---

# 将缓存装饰器应用在核心爬虫函数上
@cache(expire=CACHE_EXPIRE_SECONDS)
async def scrape_papers_for_date(date_str: str) -> List[Paper]:
    """
    爬取指定日期的论文数据。此函数的结果将被缓存。
    """
    url = f"https://huggingface.co/papers/date/{date_str}"
    logger.info(f"缓存未命中，开始为日期 {date_str} 实时爬取论文: {url}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36", 
        "Accept-Language": "en-US,en;q=0.9"
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
            # 对于爬虫函数，返回空列表比抛出HTTP异常更通用
            return []
        raise HTTPException(status_code=exc.response.status_code, detail=f"Hugging Face服务器返回错误: {exc.response.text}")

    logger.info(f"成功获取页面内容，开始解析HTML...")

    soup = BeautifulSoup(response.text, 'lxml')
    data_div = soup.find('div', attrs={'data-target': 'DailyPapers'})
    if not data_div or 'data-props' not in data_div.attrs:
        logger.warning(f"在页面 {url} 上未找到 'DailyPapers' 数据块。")
        return []
        
    try:
        props_data = json.loads(data_div['data-props'])
        papers_list_json = props_data.get('dailyPapers', [])
    except (json.JSONDecodeError, KeyError) as e:
        logger.exception("解析页面内嵌JSON数据时发生严重错误。")
        raise HTTPException(status_code=500, detail=f"解析页面内嵌JSON数据失败: {e}")

    results: List[Paper] = []
    for item in papers_list_json:
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
                authors=[author.get("name") for author in paper_data.get("authors", []) if author.get("name")],
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


async def interpret_single_paper(paper: Paper) -> str:
    """使用LLM解读单篇论文"""
    if not openai_client: return "错误：OpenAI客户端未配置。"
    prompt = PROMPT_TEMPLATE.format(title=paper.title, summary=paper.summary)
    try:
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL_NAME, 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, 
            max_tokens=8000)
        return response.choices[0].message.content or paper.summary
    except OpenAIError as e:
        logger.error(f"调用OpenAI API失败，论文标题: '{paper.title}': {e}")
        return f"错误：无法从LLM获取解读结果。详情: {e}"
    except Exception as e:
        logger.error(f"解读论文时发生未知错误，论文标题: '{paper.title}': {e}")
        return "错误：发生未知错误。"

# --- 6. API 端点 ---
@app.get(
    "/daily_report",
    response_model=MarkdownResponse,
    summary="一键获取并解读每日热门论文"
)
async def get_daily_report(
    request_date: Optional[str] = Query(None, description="要获取报告的日期 (YYYY-MM-DD)，留空则为今天。"),
    max_papers: int = Query(3, ge=1, le=10, description="解读论文的最大数量，按点赞数排序。")
):
    """
    使用大语言模型进行中文解读，并返回一份完整的Markdown报告。
    """
    if not openai_client:
        raise HTTPException(status_code=503, detail="OpenAI客户端未配置，无法提供解读服务。")

    target_date = date.today()
    if request_date:
        try:
            target_date = date.fromisoformat(request_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="日期格式无效，请使用 'YYYY-MM-DD' 格式。")
    date_str = target_date.isoformat()

    # 1. 获取论文数据 (此步骤会被缓存)
    papers = await scrape_papers_for_date(date_str)
    if not papers:
        return MarkdownResponse(markdown_report=f"# AI每日论文解读 ({date_str})\n\n今天没有找到可以解读的论文。")

    # 2. 排序和筛选
    sorted_papers = sorted(papers, key=lambda p: p.upvotes, reverse=True)
    papers_to_process = sorted_papers[:max_papers]
    logger.info(f"共找到 {len(papers)} 篇论文，将解读热度最高的 {len(papers_to_process)} 篇。")
    
    # 3. 并发调用LLM解读
    tasks = [interpret_single_paper(p) for p in papers_to_process]
    interpretations = await asyncio.gather(*tasks)

    # 4. 构建Markdown报告
    report_parts = [f"# AI每日论文解读 ({date_str})\n"]
    report_parts.append(f"精选Hugging Face上点赞数最高的 {len(papers_to_process)} 篇论文进行解读。\n---")
    for i, paper in enumerate(papers_to_process):
        report_parts.append(f"\n## {i+1}. {paper.title}\n")
        report_parts.append(f"**点赞数:** {paper.upvotes} | **作者:** {', '.join(paper.authors)}\n")
        report_parts.append(f"> [arXiv链接]({paper.arxiv_url}) | [Hugging Face链接]({paper.huggingface_url})\n")
        report_parts.append("### 🤖 LLM 解读\n")
        report_parts.append(interpretations[i])
        report_parts.append("\n---")
    
    return MarkdownResponse(markdown_report="\n".join(report_parts))

@app.get(
    "/papers",
    response_model=List[Paper],
    summary="仅获取指定日期的原始论文数据",
    description="一个辅助接口，用于获取原始、未加工的论文数据列表。"
)
async def get_raw_papers(request_date: Optional[str] = Query(None, description="日期 (YYYY-MM-DD)，留空为今天。")):
    target_date = date.today()
    if request_date:
        try:
            target_date = date.fromisoformat(request_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="日期格式无效，请使用 'YYYY-MM-DD' 格式。")
    return await scrape_papers_for_date(target_date.isoformat())

# --- 7. 用于本地运行的入口 ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)