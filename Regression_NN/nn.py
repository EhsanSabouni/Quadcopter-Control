import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from Quadrotor import Quad
import matplotlib.pyplot as plt
import csv
import re


# Define the neural network
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Generate dummy data
input_size = 23 #19 + 4
hidden_size = 50 
output_size = 19
batch_size = 10 
num_epochs = 1000

# Dummy input and target data
X_train = np.random.rand(batch_size, input_size).astype(np.float32)

contionus = 0  # 1 for contionus, 0 for discrete
if contionus == 0:
    discretesteps = 10

method = 'DOP853'
M = 0.468
g = 9.81
Ixx, Iyy, Izz = 4.856 * 10 ** -3, 4.856 * 10 ** -3, 8.801 * 10 ** -3 
b0, b1 = 1.14 * 10 ** -7, 280.19
Ax, Ay, Az = 0.25, 0.25, 0.25
beta0, beta1, beta2 = 189.63, 6.0612, 0.0122
Ir = 3.357 * 10 ** -5
l = 0.225
k = 2.98 * 10 ** -6
k1 = 0.5
p1 = 5 * 10 ** -7
stepsize = 0.1

Q1 = Quad(M, Ixx, Iyy, Izz, Ir, l, k, b0, b1, beta0, beta1, beta2, k1, p1, Ax, Ay, Az, g, method)

# Simulation parameters

directory_path = r'C:\Users\sabbir92\Desktop\Research\Quadcopter-Control-master\Dataset'

# Get a list of all files in the directory
files = os.listdir(directory_path)

# Filter out only the CSV files
csv_files = [file for file in files if file.endswith('.csv')]

# Create the neural network instance
model = FeedForwardNN(input_size, hidden_size, output_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

#for file in csv_files:
# Open the CSV file and read its content into the 'data' list
#    data = []
t = 0
    ##with open(os.path.join(directory_path, file), 'r') as file:
    #    for row in csv.reader(file):
            # Split each row by commas
            #num_epochs = int(len(row)/batch_size)    
            #for row_index in range(len(row)):
            #    temp = row[row_index].split(' ')
            #    listToStr = ''.join([str(elem) for elem in row[row_index]]).split('    ')
            #    data.append(temp)

        # Read the CSV file into a pandas DataFrame
        #data = pd.read_csv(os.path.join(directory_path, file),delim_whitespace=True)
        # Display the first few rows of the DataFrame to verify the data
        #print(data.head())

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    outputs = model(torch.Tensor(X_train))
    y_train = np.array([]).reshape(0, output_size)

    for index in range(X_train.shape[0]):
        y_train = np.vstack((y_train, Q1.rk4(Q1.dynamics,[0,stepsize],X_train[index][0:19],X_train[index][19:23],n=1)))
    loss = criterion(outputs, torch.Tensor(y_train))

    optimizer.zero_grad()  # Zero the gradients
    loss.backward()  # Perform backward pass
    optimizer.step()  # Update weights

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    t = t+ stepsize

# Example usage after training
example_input = torch.randn(1, input_size)  # Example input tensor
model.eval()  # Set the model to evaluation mode
output = model(example_input)
t_output = Q1.rk4(Q1.dynamics,[t,t+stepsize],X_train[index][0:19],X_train[index][19:23],n=1)
print("Output:", output)
print("true Output:", t_output)
