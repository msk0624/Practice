
# 0.
f = open("MLdemofile.txt", "r")
print(f.read())
print(f.read(5))

f = open("MLdemofile.txt","a")
f.write("Now the file has more content!")
f.close()
f = open("MLdemofile.txt", "r")
print(f.read())

# 1.
import numpy
arr = numpy.array([1,2,3,4,5])
print(arr)

import numpy as np
arr = np.array([1,2,3,4,5])
print(type(arr))
print(np.__version__)
print(arr)

arr = np.array([[1,2,3],[4,5,6]])
arr = np.array([[[1,2,3],[4,5,6]], [[1,2,3],[4,5,6]]])
arr[0][0][1]

arr = np.array([1,2,3,4], ndmin=5)
print(arr)
print('number of dimensions :', arr.ndim)

arr = np.array([1,2,3,4,5])
x = arr.copy()
x = arr.view() #얘는 메모리똑같이 할당이네
arr[0] = 42

print(arr)
print(x) 

list_ex = [1,2,3]
x = list_ex
list_ex[0] = 41
print(list_ex)
print(x) #list는 이런식으로 하면 한꺼번에 다같이 바뀜

arr = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
newarr = arr.reshape(4,3) #이야 이거 신기하노
print(newarr)

arr = np.array([1,2,3,4,5,4,4])
x = np.where(arr == 4)
print(x) #인덱스 출력

# 2.
from numpy import random, source

x = random.randint(100)
print(x)

x = random.choice([3,5,7,9])
print(x)

x = random.choice([3,5,7,9], size=(3,5))

# 3. scrapping
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from urllib.request import urlopen

html = urlopen('http://pythonscraping.com/pages/page1.html')
print(html.read())

from bs4 import BeautifulSoup
bs = BeautifulSoup(html.read(), 'html.parser')
print(bs.h1)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from urllib.request import urlopen
from urllib.error import HTTPError
from urllib.error import URLError

try:
    html = urlopen("https://pythonscrapingthisurldoesnotexist.com")
except HTTPError as e:
    print("The server returned an HTTP error")
except URLError as e:
    print("The server could not be found!")
else:
    print(html.read())
    
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from urllib.request import urlopen
from bs4 import BeautifulSoup
html = urlopen('http://www.pythonscraping.com/pages/warandpeace.html')
bs = BeautifulSoup(html, "html.parser")
print(bs.h2)

nameList = bs.findAll('span', {'class': 'green'})
for name in nameList:
    print(name.get_text())
    
titles = bs.find_all(['h1', 'h2','h3','h4','h5','h6'])
print([title for title in titles])