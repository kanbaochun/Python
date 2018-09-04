# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import csv
import json, codecs

class ZpInfoPipeline(object):

    def open_spider(self, spider):
        self.fo = open(r'C:\Users\baochun_kan\Desktop\招聘信息.csv', 'w', newline='')
        self.csvwriter = csv.writer(self.fo)
        self.csvwriter.writerow(['公司名称', '公司地址', '薪资水平', '职位信息'])

    def process_item(self, item, spider):
        line = [item['name'], item['address'], item['salary'], item['position']]
        self.csvwriter.writerow(line)
        return item

    def close_item(self, item, spider):
        self.fo.close()





