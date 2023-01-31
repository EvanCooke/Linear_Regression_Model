import pandas as pd
import matplotlib.pyplot as plt

# read in data from data.csv to pandas dataframe
data = pd.read_csv('data.csv')

# Data Visualization
print(data)
plt.scatter(data.x, data.y)
plt.xlabel("Attendance")
plt.ylabel("Score Points")
plt.show()


# calculate mean squared error manually
def loss_function(m, b, dataPoints):
    error_sum = 0
    for i in range(len(dataPoints)):
        x = dataPoints.iloc[i].x
        y = dataPoints.iloc[i].y
        error_sum += (y - (m * x + b)) ** 2
    error_sum / float(len(dataPoints))


def gradient_descent(current_m, current_b, dataPoints, learningRate):
    m_gradient = 0
    b_gradient = 0

    n = len(dataPoints)

    for i in range(n):
        x = dataPoints.iloc[i].x
        y = dataPoints.iloc[i].y

        m_gradient += -(2 / n) * x * (y - (current_m * x + current_b))
        b_gradient += -(2 / n) * (y - (current_m * x + current_b))

    m = current_m - m_gradient * learningRate
    b = current_b - b_gradient * learningRate
    return m, b


m = 0
b = 0
learningRate = 0.0001
epochs = 500

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, data, learningRate)

print("m: ", m, ", b: ", b)

plt.scatter(data.x, data.y, color="black")
plt.plot(list(range(0, 50)), [m * x + b for x in range(0, 50)], color="red")
plt.show()
