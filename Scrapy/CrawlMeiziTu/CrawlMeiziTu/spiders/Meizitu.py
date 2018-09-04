# -*- coding: utf-8 -*-
import scrapy
from CrawlMeiziTu.items import CrawlmeizituItem


class MeizituSpider(scrapy.Spider):
    name = "Meizitu"
    start_urls = ['http://www.dili360.com/article/p54d1d255619a921.htm']
    #解析首页获取深度1所有网址
    def parse(self, response):
        item = CrawlmeizituItem()
        item['image_urls'] = response.xpath('//div[@class="aImg"]/img/@src').extract()
        names = []
        for i in range(1,1 + len(response.xpath('//div[@class="aImg"]/img/@src').extract())):
            names.append('太原晋祠' + str(i))
            item['name'] = names
        return item

















