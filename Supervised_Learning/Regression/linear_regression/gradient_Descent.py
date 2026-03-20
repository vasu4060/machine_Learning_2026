import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



def gradient_descent(x,y,learning_rate,iterations):
           w_curr = 0
           b_curr = 0
           m = len(x)
           cost_history = []


           for i in range(iterations):
                    y_pred = w_curr * x + b_curr
          
                    cost = (1/(2*m)) * (np.sum((y_pred-y)**2))   
                    cost_history.append(cost)

                    w_d = (1/m) * np.sum(x*(y_pred-y))
                    b_d = (1/m) * np.sum(y_pred-y)

                    w_curr = w_curr - learning_rate * w_d
                    b_curr = b_curr - learning_rate * b_d

                    print(f"Iteration {i}: m={w_curr:.4f}, b={b_curr:.4f}, cost={cost:.4f}")

                    

           plt.plot(range(iterations),cost_history)
           plt.show()
           return w_curr, b_curr, cost_history


X = np.round(np.random.randn(100,1),3)
print("X: ", X)

y = 4+(3*X)+ np.round(np.random.randn(100,1),3)
print("Y: ",y)

learning_rate = 0.01
iterations = 1000

final_w, final_b,cost_history = gradient_descent(X,y,learning_rate,iterations)   

print(f"Final parameters: m={final_w:.4f}, b={final_b:.4f}")

model = LinearRegression()
model.fit(X, y)

print("Sklearn slope:", model.coef_)
print("Sklearn intercept:", model.intercept_)





