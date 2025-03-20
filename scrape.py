import base64
import glob
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
MAX_RETRIES = 7
PROXY_API = os.environ.get("SCRAPER_API")
SCRAPING_URL = os.environ.get("SCRAPING_URL")
AD_URL = os.environ.get("CAR_API_URL")

OXYLAB_USERNAME = os.environ.get("username_oxylabs")
OXYLAB_PASSWORD = os.environ.get("password_oxylabs")


PROXIES = {"https": f"{os.environ.get('proxy')}"}
credentials = f"{OXYLAB_USERNAME}:{OXYLAB_PASSWORD}"
encoded_credentials = base64.b64encode(credentials.encode()).decode()
HEADERS = {"User-Agent": "Mozilla/5.0"}

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])


# Scraping function:
def scraping_car_ads():
    """
    :return: a dictionary of car ads with uuid as key
    """
    is_last_page = False
    i = 4780
    car_ads = {}
    with tqdm() as pbar:
        while not is_last_page:
            print(f"Crawling page {i}")
            url = SCRAPING_URL.format(i)
            for attempt in range(MAX_RETRIES):
                logging.info(f"Attempt number {attempt} at requesting from the url")
                try:
                    req = requests.get(
                        url=url,
                        headers=HEADERS,
                        proxies=PROXIES,
                        verify=False,
                        timeout=10,
                    )
                    if req.status_code == 200:
                        break

                except Exception as e:
                    logging.error(e)
                    logging.info(
                        f"Encountered a problem while crawling page {i}. Saving pages {1} to {i-1}"
                    )
                time.sleep(1)

            if req.status_code != 200:
                backup_df = pd.DataFrame.from_dict(car_ads, orient="index")
                backup_df.to_csv(f"data/car_ads_backup_{i}_v4.csv")
                return backup_df

            is_last_page = req.json()["data"]["results"]["pagination"]["is_last_page"]
            c = 0
            for car in req.json()["data"]["results"]["rows"]:
                uuid_car = uuid.uuid4()
                car_ads[uuid_car] = car
                c += 1
            print(f"Crawled {c} cars out of page{i}")
            i += 1
            pbar.update(1)
            time.sleep(0.5)
        logging.info(f"Finished crawling all pages \n Saving car ads to dataframe")
        car_ads_df = pd.DataFrame.from_dict(car_ads, orient="index")
        car_ads_df.to_csv("data/car_ads.csv")

    return car_ads_df


def ad_request(id):
    url = AD_URL + str(id)
    req = None
    for attempt in range(MAX_RETRIES):
        logging.info(f"Attempt number {attempt} at requesting from the url")
        try:
            req = requests.get(
                url=url, headers=HEADERS, proxies=PROXIES, verify=False, timeout=10
            )
            if req.status_code == 200:
                break

        except Exception as e:
            logging.error(e)
            logging.info(f"Encountered a problem while crawling ad number {id}")

    if req.status_code != 200:
        return id
    else:
        try:
            response = req.json()["data"]["classified"]
        except:
            print(f"Couldn't access json of ad {id}")
            return id
        with open(f"data/ads/{id}.json", "w") as file:
            json.dump(response, file, indent=4, sort_keys=True)
        return None


def scrape_ads_one_by_one():
    ids = (
        pd.read_csv("data/processed_ads_df.csv", usecols=["id"])["id"]
        .astype("str")
        .tolist()
    )
    scraped_ads = [
        os.path.splitext(os.path.basename(x))[0] for x in glob.glob("data/ads/*.json")
    ]
    ids_to_scrape = list(set(ids) - set(scraped_ads))
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(ad_request, ids_to_scrape))

    failed_ads = [ads for ads in results if ads is not None]

    if failed_ads:
        np.save(
            arr=np.array(failed_ads), file="data/ads/failed_ads_crawl/failed_ads.npy"
        )

    return failed_ads


if __name__ == "__main__":
    unscraped_ads = scrape_ads_one_by_one()

# result = scraping_car_ads()
