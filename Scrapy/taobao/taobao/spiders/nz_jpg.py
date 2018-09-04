# -*- coding: utf-8 -*-
import scrapy
from taobao.items import TaobaoItem

class NzJpgSpider(scrapy.Spider):
    name = 'nz_jpg'
    start_urls = ['https://list.tmall.com/search_product.htm?spm=a220m.1000858.1000724.4.841d1d5bm9wgHz&q=%B6%CC%C8%B9&sort=d&style=g&from=mallfp..pc_1_suggest&suggest=0_1&active=2&smAreaId=440300#J_Filter']

    def parse(self, response):
        item = TaobaoItem()
        try:
            urls = response.xpath('//div[@class="productImg-wrap"]/a/img').re(r'//.*.jpg')
            for i in urls:
                item['image_urls'].append('http:' + i)
            item['title'] = response.xpath('//p[@class="productTitle"]/a/@title').extract()
        except:
            pass
        return item


