# 김혜성 / 201821487 / ghtn2638@ajou.ac.kr 
#library
from math import sqrt
import re

# 1. convertCelsius
def convertCelsius():
    C_x = float(input("Enter a temperature degree in Celsius. "))
    F_x = C_x * (9 / 5) + 32
    print(F_x, "degrees Fahrenheit.")
convertCelsius()
    
# 2. getLength
def getLength():
    Len_x = input()
    print("The Length of the input is", len(Len_x), "characters.")
getLength()

# 3. quadraticRoot
def quadraticRoot():
    abc = input()
    abc_list = re.findall("\d+", abc)
    print(abc)
    a = int(abc_list[0])
    b = int(abc_list[1])
    c = int(abc_list[2])
    Xplus = (-b + sqrt(b**2-4*a*c))/2 
    Xminus = (-b - sqrt(b**2-4*a*c))/2
    print("The solutions include ",Xplus, "and", Xminus,".")
quadraticRoot()

# 4. calcDistance
def calcDistance(x1,y1,x2,y2):
    d = sqrt((x2-x1)**2 + (y2 - y1)**2)
    return(float(d))
calcDistance(-1,15,7,-3)