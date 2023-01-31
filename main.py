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


def gradient_descent(current_m, current_b, data_points, learning_rate):
    m_gradient = 0
    b_gradient = 0

    n = len(data_points)

    for i in range(n):
        x = data_points.iloc[i].x
        y = data_points.iloc[i].y

        m_gradient += -(2 / n) * x * (y - (current_m * x + current_b))
        b_gradient += -(2 / n) * (y - (current_m * x + current_b))

    m = current_m - m_gradient * learning_rate
    b = current_b - b_gradient * learning_rate
    return m, b


def find_y_from_x(m, b, x):
    return m * x + b


def find_x_from_y(m, b, y):
    return (y - b) / m


# Driver code
m = 0
b = 0
learning_rate = 0.0001
epochs = 1000

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, data, learning_rate)

# QUESTION 1
print("\n-----------------QUESTION 1-----------------")
print("Regression line:")
print("y = ", m, "* x + ", b)
print("m: ", m, ", b: ", b)

# QUESTION 2
print("\n-----------------QUESTION 2-----------------")
question2result1 = find_y_from_x(m, b, 9)
question2result2 = find_y_from_x(m, b, 14)

if question2result1 < 40:
    print("With an attendance of 9, it is unlikely that he/she will "
          "pass the course. (predicted score = ", question2result1, ")")
else:
    print("With an attendance of 9, it is likely that he/she will "
          "pass the course. (predicted score = ", question2result1, ")")

if question2result2 < 40:
    print("With an attendance of 14, it is unlikely that he/she "
          "will pass the course. (predicted score = ", question2result2, ")")
else:
    print("With an attendance of 14, it is likely that he/she "
          "will pass the course. (predicted score = ", question2result2, ")")


# QUESTION 3
print("\n-----------------QUESTION 3-----------------")
minimumAttendanceToPass = find_x_from_y(m, b, 40)
print("Predicted minimum attendance to pass the class: ", minimumAttendanceToPass)


# show regression line on plotted data graph
plt.scatter(data.x, data.y, color="black")
plt.plot(list(range(0, 50)), [m * x + b for x in range(0, 50)], color="red")
plt.show()


