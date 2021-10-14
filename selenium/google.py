'''
가상환경 설정
터미널 창
python -m venv selenium
cd selenium\Scripts
activate

'''

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request


driver = webdriver.Chrome(executable_path='C:/Users/AI-00/PycharmProjects/yolov5-master/selenium/chromedriver.exe')
driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")

#html 내부 검색창을 찾음
elem = driver.find_element_by_name("q")

# 입력값 전송
elem.send_keys("헤드램프")

# Enter Key 전송
elem.send_keys(Keys.RETURN)

SCROLL_PAUSE_TIME = 1

# 스크롤 내리기
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        try:
            driver.find_element_by_css_selector('.mye4qd').click()
        except:
            break
    last_height = new_height

images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")
count = 1
for image in images:
    try:
        image.click()
        time.sleep(2)
        imgUrl = driver.find_element_by_css_selector(".n3VNCb").get_attribute("src")
        # imgUrl = driver.find_element_by_xpath("/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div[1]/a/img").get_attribute("src")
        urllib.request.urlretrieve(imgUrl, f'./img/{count}.jpg')
        print(f'{count} 저장되었습니다.')
        count += 1
    except:
        pass
driver.close()

