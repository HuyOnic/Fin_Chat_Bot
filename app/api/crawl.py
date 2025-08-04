from app.news_crawler.crawl_cafef import cafefCrawler
from app.news_crawler.crawl_vietstock import vietstockCrawler


def run_crawler(crawl_source: str, days: int):
    if days < 1:
        return {"error": "Số ngày crawl phải >= 1."}

    global_stocks, market_analysis, macro_economy = [], [], []

    if crawl_source in ["cafef", "all"]:
        global_stocks += cafefCrawler(channelID=18832, news_type=1, topic="tai-chinh-quoc-te", days=days)
        market_analysis += cafefCrawler(channelID=18839, news_type=1, topic="thi-truong", days=days)
        macro_economy += cafefCrawler(channelID=18833, news_type=3, topic="vi-mo-dau-tu", days=days)

    if crawl_source in ["vietstock", "all"]:
        global_stocks += vietstockCrawler(channelID=773, news_type=1, days=days)
        market_analysis += vietstockCrawler(channelID=1636, news_type=1, days=days)
        macro_economy += vietstockCrawler(channelID=761, news_type=3, days=days)

    if crawl_source not in ["cafef", "vietstock", "all"]:
        return {"error": "Nguồn không hợp lệ. Chọn 'cafef', 'vietstock' hoặc 'all'."}

    all_data = global_stocks + market_analysis + macro_economy
    return all_data
