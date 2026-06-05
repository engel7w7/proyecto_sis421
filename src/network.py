import torch
import torch.nn as nn

class CheatingLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)      
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        ultimo_cuadro = out[:, -1, :]
        prediccion = self.fc(ultimo_cuadro)
        return self.sigmoid(prediccion)