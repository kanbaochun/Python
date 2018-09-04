# -*- coding: utf-8 -*-
import scrapy
from zp_info.items import ZpInfoItem

class ZpinfoSpider(scrapy.Spider):
    name = 'zpinfo'
    urls = []
    key = "算法工程师"
    for i in range(1, 91):
        urls.append('http://sou.zhaopin.com/jobs/searchresult.ashx?jl=%E5%85%A8%E5%9B%BD&kw=' + key + '&sm=0&isfilter=0&fl=489&isadv=0&sg=a42341056a684891be11c88d32001737&p=' + str(i))
    start_urls = urls

    def parse(self, response):
        for url in response.xpath('//div[@style="width: 224px;*width: 218px; _width:200px; float: left"]/a/@href').extract():
            yield scrapy.Request(url, callback = self.parse_url)

    def parse_url(self, response):
        item = ZpInfoItem()
        item['name'] = response.xpath('//div[@class="inner-left fl"]/h2/a/text()').extract()[0]
        item['address'] = response.xpath('//li//strong/a/text()').extract()[0]
        item['salary'] = response.xpath('//li/strong/text()').extract()[0][:-1]
        item['position'] = response.xpath('//li/strong/a[@target="_blank"]/text()').extract()[1]
        yield item




