from selenium import webdriver
from time import sleep


def web(username,password):
        driver = webdriver.Chrome('C:\\Users\\LALIT\\Desktop\\chromedriver.exe')
        driver.get("https://instagram.com")
        sleep(2)
        driver.find_element_by_xpath("//input[@name=\"username\"]").send_keys(username)
        sleep(2)
        driver.find_element_by_xpath("//input[@name=\"password\"]").send_keys(password)
        sleep(2)
        driver.find_element_by_xpath('//*[@id="react-root"]/section/main/article/div[2]/div[1]/div/form/div[4]/button/div').click()
        sleep(20000)

def passed():
        web("username","password")