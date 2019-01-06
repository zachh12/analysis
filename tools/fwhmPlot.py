import numpy as np
import matplotlib.pyplot as plt
def main():
    x = np.linspace(0, 5000)
    #x = x * 1000
    y = np.sqrt((x/.3) + (x))
    y = y/1000
    plt.plot(x, y)
    plt.show()
if __name__=="__main__":
    main()