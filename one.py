import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Google_Stock_Price.csv")
data = df[['Close']].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

time_step = 5
X, y = [], []
for i in range(len(scaled_data) - time_step):
    X.append(scaled_data[i:i + time_step, 0])
    y.append(scaled_data[i + time_step, 0])

X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)

class StockRNN(nn.Module):
    def __init__(self):
        super(StockRNN, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

model = StockRNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(30):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

model.eval()
last_sequence = torch.tensor(scaled_data[-time_step:], dtype=torch.float32).unsqueeze(0)
predicted_scaled = model(last_sequence).detach().numpy()
predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]
current_price = data[-1][0]

print(f"Current Google Stock Price: {current_price:.2f}")
print(f"Predicted Next Day Google Stock Price: {predicted_price:.2f}")

if predicted_price > current_price:
    print("Trend Analysis: INCREASING trend of stock price for the next day.")
else:
    print("Trend Analysis: DECREASING trend of stock price for the next day.")
