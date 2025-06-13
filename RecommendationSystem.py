# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def num_of_val(file_path, target_column_index):
    max_value = float('-inf')
    with open(file_path, 'r') as file:
        for line in file:
            columns = line.strip().split('\t')
            if len(columns) > target_column_index:
                try:
                    value = float(columns[target_column_index])
                    if value > max_value:
                        max_value = value
                except ValueError:
                    pass
    if max_value == float('-inf'):
        raise ValueError(f"No valid values at column index {target_column_index} in {file_path}")
    return int(max_value)

# File path
file_path = 'train.txt'

# Define matrix dimensions
m = num_of_val(file_path, 0) + 1  # Number of users
n = num_of_val(file_path, 1) + 1  # Number of movies
k = 25  # Number of latent factors

# Initialize latent factor matrices P (users) and Q (movies)
P = np.random.uniform(0, np.sqrt(5 / k), size=(m, k))
Q = np.random.uniform(0, np.sqrt(5 / k), size=(n, k))

# Hyperparameters
learning_rate = 0.005  # Reduced learning rate for stability
regularization = 0.1
num_iterations = 50
clip_value = 10  # Clipping threshold for P and Q

def cal_error(file_path, P, Q, regularization):
    total_error = 0
    with open(file_path, 'r') as file:
        for line in file:
            user, movie, rating, _ = map(int, line.strip().split('\t'))
            p_x = P[user, :]
            q_i = Q[movie, :]
            error_term = (rating - np.dot(p_x, q_i)) ** 2
            reg_term = regularization * (np.sum(p_x ** 2) + np.sum(q_i ** 2))
            total_error += error_term + reg_term
    return total_error

def gradient_descent(file_path, P, Q, regularization, learning_rate, num_iterations, flag):
    errors = []
    for iteration in range(num_iterations):
        with open(file_path, 'r') as file:
            for line in file:
                user, movie, rating, _ = map(int, line.strip().split('\t'))
                prediction = np.dot(P[user], Q[movie])
                error = rating - prediction
                
                # Gradient update
                P[user] += learning_rate * (error * Q[movie] - regularization * P[user])
                Q[movie] += learning_rate * (error * P[user] - regularization * Q[movie])
                
                # Clip values to avoid overflow
                np.clip(P[user], -clip_value, clip_value, out=P[user])
                np.clip(Q[movie], -clip_value, clip_value, out=Q[movie])
                
                # Check for NaNs and break early if found
                if np.isnan(P[user]).any() or np.isnan(Q[movie]).any():
                    print(f"NaN detected at iteration {iteration}, user {user}, movie {movie}")
                    return P, Q, errors
        
        if flag:
            err = cal_error(file_path, P, Q, regularization)
            print(f"Iteration {iteration+1}/{num_iterations} - Error: {err:.4f}")
            errors.append(err)
    return P, Q, errors

# Train the model
P, Q, errors = gradient_descent(file_path, P, Q, regularization, learning_rate, num_iterations, 1)

# Save matrices
np.save('P.npy', P)
np.save('Q.npy', Q)

# Plot: Error vs Iterations
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(errors) + 1), errors, marker='o')
plt.xlabel('Number of Iterations')
plt.ylabel('Error (Objective Function)')
plt.title('Error vs. Number of Iterations')
plt.grid(True)
plt.show()

# Evaluate error for different latent factor sizes k
def cal_error_k(file_path, min_k, max_k, num_iterations, regularization, m, n, learning_rate):
    e_values = []
    for k_val in range(min_k, max_k + 1, 2):
        P_k = np.random.uniform(0, np.sqrt(5 / k_val), size=(m, k_val))
        Q_k = np.random.uniform(0, np.sqrt(5 / k_val), size=(n, k_val))
        P_k, Q_k, _ = gradient_descent(file_path, P_k, Q_k, regularization, learning_rate, num_iterations, 0)
        e_values.append(cal_error(file_path, P_k, Q_k, regularization))
    return e_values

min_k, max_k = 20, 40
k_values = list(range(min_k, max_k + 1, 2))
e_values = cal_error_k(file_path, min_k, max_k, num_iterations, regularization, m, n, learning_rate)

plt.figure(figsize=(10, 5))
plt.bar(k_values, e_values)
plt.xlabel('Number of Latent Factors (k)')
plt.ylabel('Objective Function (E)')
plt.title('Objective Function E vs. Number of Latent Factors')
plt.show()

# Fine tuning: regularization
def cal_error_r(file_path):
    r = 0.0
    errors_r = []
    for i in range(10):
        r += 0.1
        P_new = np.random.uniform(0, np.sqrt(5 / k), size=(m, k))
        Q_new = np.random.uniform(0, np.sqrt(5 / k), size=(n, k))
        P_new, Q_new, _ = gradient_descent(file_path, P_new, Q_new, r, learning_rate, num_iterations, 0)
        errors_r.append(cal_error(file_path, P_new, Q_new, r))
    return errors_r

errors_r = cal_error_r(file_path)

# Fine tuning: learning rate
def cal_error_lr(file_path):
    lr = 0.005  # start lower since learning rate was reduced
    errors_lr = []
    for i in range(10):
        P_new = np.random.uniform(0, np.sqrt(5 / k), size=(m, k))
        Q_new = np.random.uniform(0, np.sqrt(5 / k), size=(n, k))
        P_new, Q_new, _ = gradient_descent(file_path, P_new, Q_new, regularization, lr, num_iterations, 0)
        errors_lr.append(cal_error(file_path, P_new, Q_new, regularization))
        lr += 0.01
    return errors_lr

errors_lr = cal_error_lr(file_path)

# Model evaluation function for RMSE
def model_eval(eval_data_path, P, Q):
    nv = num_of_val(eval_data_path, 1) + 1
    if Q.shape[0] < nv:
        Q = np.random.uniform(0, np.sqrt(5 / k), size=(nv, k))
        P, Q, _ = gradient_descent(eval_data_path, P, Q, regularization, learning_rate, num_iterations + 7, 0)

    squared_errors = []
    with open(eval_data_path, 'r') as eval_file:
        for line in eval_file:
            user_id, item_id, rating, _ = map(int, line.strip().split('\t'))
            if user_id < P.shape[0] and item_id < Q.shape[0]:
                prediction = np.dot(P[user_id], Q[item_id])
                error = rating - prediction
                squared_errors.append(error ** 2)
    return round(np.sqrt(np.mean(squared_errors)), 4)

# RMSE Evaluation
print("Train RMSE:", model_eval(file_path, P, Q))
print("Test RMSE:", model_eval('test.txt', P, Q))

# Save final matrices again
np.save('P.npy', P)
np.save('Q.npy', Q)
