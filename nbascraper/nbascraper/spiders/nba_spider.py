# Run this spider in the command line with the following command: scrapy crawl pbr -o command_results.csv

# To use the pipeline simply uncomment the pipeline lines in the settings.py file and just run scrapy crawl pfr, but no headers will populate





import scrapy

from scrapy import Selector
from nbascraper.items import NbascraperItem
from datetime import date

class NBAScraperSpider(scrapy.Spider):
	name = "pbr"
	allowed_domains = ['basketball-reference.com']
	def start_requests(self):
		months = ['october','november']#,'december','january','february','march','april','may','june']
		for year in range(2001,2002):
			for month in months:
				yield scrapy.Request("http://www.basketball-reference.com/leagues/NBA_"+str(year)+"_games-"+month+".html", self.parse)
		# yield scrapy.Request("http://www.pro-football-reference.com/years/2015/games.htm", self.parse)
		# yield scrapy.Request("http://www.pro-football-reference.com/years/2014/games.htm", self.parse)
		# yield scrapy.Request("http://www.pro-football-reference.com/years/2013/games.htm", self.parse)
		# yield scrapy.Request("http://www.pro-football-reference.com/years/2012/games.htm", self.parse)
		# yield scrapy.Request("http://www.pro-football-reference.com/years/2011/games.htm", self.parse)
		# yield scrapy.Request("http://www.pro-football-reference.com/years/2010/games.htm", self.parse)
		# yield scrapy.Request("http://www.pro-football-reference.com/years/2009/games.htm", self.parse)
		# yield scrapy.Request("http://www.pro-football-reference.com/years/2008/games.htm", self.parse)
		# yield scrapy.Request("http://www.pro-football-reference.com/years/2007/games.htm", self.parse)
		# yield scrapy.Request("http://www.pro-football-reference.com/years/2006/games.htm", self.parse)
		# yield scrapy.Request("http://www.pro-football-reference.com/years/2005/games.htm", self.parse)
		# yield scrapy.Request("http://www.pro-football-reference.com/years/2004/games.htm", self.parse)
		# yield scrapy.Request("http://www.pro-football-reference.com/years/2003/games.htm", self.parse)
		# yield scrapy.Request("http://www.pro-football-reference.com/years/2002/games.htm", self.parse)
		# yield scrapy.Request("http://www.pro-football-reference.com/years/2001/games.htm", self.parse)
		# yield scrapy.Request("http://www.pro-football-reference.com/years/2000/games.htm", self.parse)

	def parse(self,response):
		for href in response.xpath('//a[contains(text(),"Box Score")]/@href'):
			item = NbascraperItem()
			url = response.urljoin(href.extract())
			request = scrapy.Request(url, callback=self.parse_dir_contents)
			request.meta['item'] = item
			yield request

	def parse_dir_contents(self,response):
		item = response.meta['item']

		# Line score is a JS table, so information from there needs to be extracted
		line_score_text = response.xpath('//div[@id="all_line_score"]//comment()').extract()[0]
		line_score_selector = Selector(text=line_score_text[4:-3].strip())

		home_team = line_score_selector.xpath('//*[@id="line_score"]//a/text()').extract()[1]
		away_team = line_score_selector.xpath('//*[@id="line_score"]//a/text()').extract()[0]
		item['home_team'] = home_team
		item['away_team'] = away_team

		item['home_score'] = line_score_selector.xpath('//*[@id="line_score"]/tr[last()]//td/strong/text()').extract()[0].strip()
		item['away_score'] = line_score_selector.xpath('//*[@id="line_score"]/tr[last()-1]//td/strong/text()').extract()[0].strip()

		# Need to create lower values for home and away team due to the custom div id for the tables on each page
		hteam = home_team.lower()
		ateam = away_team.lower()


		DateDict = {"January" : 1, "February" : 2, "March" : 3, "April" : 4, "May" : 5, "June" : 6, "July" : 7, "August" : 8, "Septemper" : 9, "October" : 10, "November" : 11, "December" : 12}
		gamedate = response.xpath('//*[@id="content"]/div[2]/div[3]/div[1]/text()').extract()[0].strip()
		month = DateDict[gamedate.split(', ')[1].split(' ')[0]]
		day = int(gamedate.split(', ')[1].split(' ')[1])
		year = int(gamedate.split(', ')[2])
		item['game_date'] = date(year,month,day)

		item['home_fg'] = response.xpath('//*[@id="box_'+hteam+'_basic"]/tfoot/tr/td[2]/text()').extract()[0].strip()
		item['home_fga'] = response.xpath('//*[@id="box_'+hteam+'_basic"]/tfoot/tr/td[3]/text()').extract()[0].strip()
		item['home_3p'] = response.xpath('//*[@id="box_'+hteam+'_basic"]/tfoot/tr/td[5]/text()').extract()[0].strip()
		item['home_3pa'] = response.xpath('//*[@id="box_'+hteam+'_basic"]/tfoot/tr/td[6]/text()').extract()[0].strip()
		item['home_ft'] = response.xpath('//*[@id="box_'+hteam+'_basic"]/tfoot/tr/td[8]/text()').extract()[0].strip()
		item['home_fta'] = response.xpath('//*[@id="box_'+hteam+'_basic"]/tfoot/tr/td[9]/text()').extract()[0].strip()
		item['home_orb'] = response.xpath('//*[@id="box_'+hteam+'_basic"]/tfoot/tr/td[11]/text()').extract()[0].strip()
		item['home_drb'] = response.xpath('//*[@id="box_'+hteam+'_basic"]/tfoot/tr/td[12]/text()').extract()[0].strip()
		item['home_ast'] = response.xpath('//*[@id="box_'+hteam+'_basic"]/tfoot/tr/td[13]/text()').extract()[0].strip()
		item['home_stl'] = response.xpath('//*[@id="box_'+hteam+'_basic"]/tfoot/tr/td[14]/text()').extract()[0].strip()
		item['home_blk'] = response.xpath('//*[@id="box_'+hteam+'_basic"]/tfoot/tr/td[15]/text()').extract()[0].strip()
		item['home_tov'] = response.xpath('//*[@id="box_'+hteam+'_basic"]/tfoot/tr/td[16]/text()').extract()[0].strip()
		item['home_pf'] = response.xpath('//*[@id="box_'+hteam+'_basic"]/tfoot/tr/td[17]/text()').extract()[0].strip()
		item['home_ts'] = response.xpath('//*[@id="box_'+hteam+'_advanced"]/tfoot/tr/td[2]/text()').extract()[0].strip()
		item['home_efg'] = response.xpath('//*[@id="box_'+hteam+'_advanced"]/tfoot/tr/td[3]/text()').extract()[0].strip()

		item['away_fg'] = response.xpath('//*[@id="box_'+ateam+'_basic"]/tfoot/tr/td[2]/text()').extract()[0].strip()
		item['away_fga'] = response.xpath('//*[@id="box_'+ateam+'_basic"]/tfoot/tr/td[3]/text()').extract()[0].strip()
		item['away_3p'] = response.xpath('//*[@id="box_'+ateam+'_basic"]/tfoot/tr/td[5]/text()').extract()[0].strip()
		item['away_3pa'] = response.xpath('//*[@id="box_'+ateam+'_basic"]/tfoot/tr/td[6]/text()').extract()[0].strip()
		item['away_ft'] = response.xpath('//*[@id="box_'+ateam+'_basic"]/tfoot/tr/td[8]/text()').extract()[0].strip()
		item['away_fta'] = response.xpath('//*[@id="box_'+ateam+'_basic"]/tfoot/tr/td[9]/text()').extract()[0].strip()
		item['away_orb'] = response.xpath('//*[@id="box_'+ateam+'_basic"]/tfoot/tr/td[11]/text()').extract()[0].strip()
		item['away_drb'] = response.xpath('//*[@id="box_'+ateam+'_basic"]/tfoot/tr/td[12]/text()').extract()[0].strip()
		item['away_ast'] = response.xpath('//*[@id="box_'+ateam+'_basic"]/tfoot/tr/td[13]/text()').extract()[0].strip()
		item['away_stl'] = response.xpath('//*[@id="box_'+ateam+'_basic"]/tfoot/tr/td[14]/text()').extract()[0].strip()
		item['away_blk'] = response.xpath('//*[@id="box_'+ateam+'_basic"]/tfoot/tr/td[15]/text()').extract()[0].strip()
		item['away_tov'] = response.xpath('//*[@id="box_'+ateam+'_basic"]/tfoot/tr/td[16]/text()').extract()[0].strip()
		item['away_pf'] = response.xpath('//*[@id="box_'+ateam+'_basic"]/tfoot/tr/td[17]/text()').extract()[0].strip()
		item['away_ts'] = response.xpath('//*[@id="box_'+ateam+'_advanced"]/tfoot/tr/td[2]/text()').extract()[0].strip()
		item['away_efg'] = response.xpath('//*[@id="box_'+ateam+'_advanced"]/tfoot/tr/td[3]/text()').extract()[0].strip()

		yield item