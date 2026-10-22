import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"OpenCV version: {cv2.__version__}")

x = np.array([0, 1, 2, 3, 4])
y = x ** 2

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("y = x^2")
plt.savefig("setup_check.png")
plt.show()
