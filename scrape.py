import time
import uuid
import random
import logging
import os
from dotenv import load_dotenv

import requests
from tqdm import tqdm
import pandas as pd

load_dotenv()
MAX_RETRIES = 5
PROXY_API = os.environ.get('SCRAPER_API')
SCRAPING_URL = 'https://www.car.gr/api/clsfds/search/?category=15001&pg={}&per-page=24&only-results=1'
HEADERS = {'User-Agent' : 'Mozilla/5.0'}
PROXIES = {"http": f"http://scraperapi:{PROXY_API}@proxy-server.scraperapi.com:8001", "https": f"http://scraperapi:{PROXY_API}@proxy-server.scraperapi.com:8001"
}

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

#Scraping function:
def scraping_car_ads():
    """
    :return: a dictionary of car ads with uuid as key
    """
    is_last_page = False
    i = 1
    car_ads = {}
    with tqdm() as pbar:
        while not is_last_page:
            print(f"Crawling page {i}")
            url = SCRAPING_URL.format(i)
            for attempt in range(MAX_RETRIES):
                logging.info(f"Attempt number {attempt} at requesting from the url")
                try:
                    req = requests.get(url=url, headers=HEADERS,proxies=PROXIES, verify = False, timeout=10)
                    if req.status_code == 200:
                        break

                except Exception as e:
                    logging.error(e)
                    logging.info(f"Encountered a problem while crawling page {i}. Saving pages {1} to {i-1}")
                time.sleep(0.5)

            if req.status_code != 200:
                backup_df = pd.DataFrame.from_dict(car_ads, orient='index')
                backup_df.to_csv(f'data/car_ads_backup_{i}.csv')
                return backup_df

            is_last_page = req.json()['data']['results']['pagination']['is_last_page']
            c = 0
            for car in req.json()['data']['results']['rows']:
                uuid_car = uuid.uuid4()
                car_ads[uuid_car] = car
                c+=1
            print(f"Crawled {c} cars out of page{i}")
            i+=1
            pbar.update(1)
            time.sleep(0.5)
        logging.info(f"Finished crawling all pages \n Saving car ads to dataframe")
        car_ads_df = pd.DataFrame.from_dict(car_ads, orient='index')
        car_ads_df.to_csv('data/car_ads.csv')

    return car_ads_df
if __name__ == '__main__':

    result = scraping_car_ads()
