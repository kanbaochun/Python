# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import scrapy, os
from scrapy.contrib.pipeline.images import ImagesPipeline
from scrapy.exceptions import DropItem
from scrapy.utils.project import get_project_settings

class CrawlmeizituPipeline(ImagesPipeline):

    IMAGES_STORE = get_project_settings().get('IMAGES_STORE')#获取配置文件中存储路径

    def get_media_requests(self, item, info):
        for image_url in item['image_urls']:
            yield scrapy.Request(image_url)

    def item_completed(self, results, item, info):
        image_paths = [x['path'] for ok, x in results if ok]
        if not image_paths:
            raise DropItem("Item contains no images")
        for i in range(len(item["name"])):
            item['image_paths'] = self.IMAGES_STORE + 'full/' + str(item["name"][i])
            os.rename(self.IMAGES_STORE + image_paths[i], self.IMAGES_STORE + 'full/' + str(item["name"][i]) + ".jpg")
        return item
