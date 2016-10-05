# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class NbascraperItem(scrapy.Item):
	game_date = scrapy.Field()
	home_team = scrapy.Field()
	away_team = scrapy.Field()
	
	home_fg = scrapy.Field()
	home_fga = scrapy.Field()
	home_3p = scrapy.Field()
	home_3pa = scrapy.Field()
	home_ft = scrapy.Field()
	home_fta = scrapy.Field()
	home_orb = scrapy.Field()
	home_drb = scrapy.Field()
	home_ast = scrapy.Field()
	home_stl = scrapy.Field()
	home_blk = scrapy.Field()
	home_tov = scrapy.Field()
	home_pf = scrapy.Field()
	home_ts = scrapy.Field()
	home_efg = scrapy.Field()

	away_fg = scrapy.Field()
	away_fga = scrapy.Field()
	away_3p = scrapy.Field()
	away_3pa = scrapy.Field()
	away_ft = scrapy.Field()
	away_fta = scrapy.Field()
	away_orb = scrapy.Field()
	away_drb = scrapy.Field()
	away_ast = scrapy.Field()
	away_stl = scrapy.Field()
	away_blk = scrapy.Field()
	away_tov = scrapy.Field()
	away_pf = scrapy.Field()
	away_ts = scrapy.Field()
	away_eft = scrapy.Field()

	home_score = scrapy.Field()
	away_score = scrapy.Field()
