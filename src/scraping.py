from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from time import sleep
from concurrent.futures import ThreadPoolExecutor
import requests
import random
import io
from PIL import Image
import numpy as np

chromedriver_path = '/home/ilolio/Documents/chromedriver'

options = Options()
# options.headless = True


# Scraping
def download_img(img_url):
    sleep((random.random() + 0.5) * 3)
    r = requests.get(img_url)
    return r.content


def parse_page(url):
    driver = webdriver.Chrome(executable_path=chromedriver_path, options=options)

    driver.get(url=url)

    for j in range(1):  # Scrolling the page

        try:
            button = driver.find_element_by_class_name('tCibT')  # "Show more" button
            button.click()
        except Exception as e:
            pass

        sleep((random.random() + 0.5) * 3)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    sleep((random.random() + 0.5) * 2)

    images = driver.find_elements_by_class_name('KL4Bh > img')  # Image tags
    image_urls = [image.get_attribute('src') for image in images]

    with ThreadPoolExecutor() as executor:
        image_binaries = list(executor.map(download_img, image_urls))  # Extracting binary of the images

    # Converting img binaries to np arrays
    img_arrays = []
    for j, image_binary in enumerate(image_binaries):
        if j >= 20:
            break
        img = Image.open(io.BytesIO(image_binary))
        arr = np.asarray(img)
        img_arrays.append(arr)

    print(len(img_arrays), 'imgs are scraped.')

    return img_arrays
