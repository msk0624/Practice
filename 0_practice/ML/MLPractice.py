import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([0,6])
ypoints = np.array([0,260])

plt.plot(xpoints, ypoints)
plt.show()

xpoints = np.array([1,9])
ypoints = np.array([3,10])
plt.plot(xpoints, ypoints,'v') #저 v가 점 형태인데 여러개가 있다.
plt.show()

xpoints = np.array([1,2,6,8])
ypoints = np.array([3,8,1,10])
plt.xlabel("PracticeX")
plt.ylabel("PracticeY")

plt.scatter(xpoints,ypoints) #점
plt.plot(xpoints, ypoints) #선
plt.bar(xpoints,ypoints) #바

mylabels = ["x","y","z","w"]
plt.pie(xpoints, labels=mylabels) #파이차트

plt.show()

