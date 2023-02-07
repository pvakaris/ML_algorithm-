def get_gradient_at_b(x_points, y_points, m, b):
    # Sum of all differences between y and predicted y
    sum_of_differences = 0
    n = len(x_points)
    for i in range(n):
        sum_of_differences += y_points[i] - (m*x_points[i] + b)

    b_gradient = -2/n * sum_of_differences
    return b_gradient

def get_gradient_at_m(x_points, y_points, m, b):
    # Sum of all differences between y and predicted y
    sum_of_differences = 0
    n = len(x_points)
    for i in range(n):
        sum_of_differences += x_points[i] * (y_points[i] - (m*x_points[i] + b))

    m_gradient = -2/n * sum_of_differences
    return m_gradient

def step_gradient(x_points, y_points, b_current, m_current, learning_rate):
  b_gradient = get_gradient_at_b(x_points, y_points, b_current, m_current)
  m_gradient = get_gradient_at_m(x_points, y_points, b_current, m_current)
  
  b = b_current - (learning_rate * b_gradient)
  m = m_current - (learning_rate * m_gradient)

  return (b, m)

def gradient_descent(x_points, y_points, learning_rate, iterations):
    b = 0
    m = 0
    for i in range(iterations):
      new_values = step_gradient(b, m, x_points, y_points, learning_rate)
      b = new_values[0]
      m = new_values[1]
    return (b, m)


import matplotlib.pyplot as plt
x_points = []
y_points = []
learning_rate = 0.001
iterations = 1000

# Find the best b and m values that fit the data
b, m = gradient_descent(x_points, y_points, 0.0001, 1000)
y_predictions = [x*m + b for x in x_points ]

# Plot original points and their predictions according to the m and b values
plt.plot(x_points, y_points, 'o')
plt.plot(x_points, y_predictions, 'o')
plt.show()