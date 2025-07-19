import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models



# --- TRANSFORMAÇÕES ---
transformacao = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# --- DATASETS ---
caminho_treino = 'treino'
caminho_val = 'validacao'

dataset_treino = datasets.ImageFolder(root=caminho_treino, transform=transformacao)
dataset_val = datasets.ImageFolder(root=caminho_val, transform=transformacao)

loader_treino = DataLoader(dataset_treino, batch_size=32, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=32)

# --- MODELO CNN ---
class MinhaCNN(nn.Module):
    def __init__(self):
        super(MinhaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, len(dataset_treino.classes))  # classes detectadas

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
  

# --- TREINAMENTO ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = MinhaCNN().to(device)

criterio = nn.CrossEntropyLoss()
otimizador = optim.Adam(modelo.parameters(), lr=0.001)

for epoca in range(5):
    modelo.train()
    total_loss = 0
    for imagens, rotulos in loader_treino:
        imagens, rotulos = imagens.to(device), rotulos.to(device)
        otimizador.zero_grad()
        saida = modelo(imagens)
        loss = criterio(saida, rotulos)
        loss.backward()
        otimizador.step()
        total_loss += loss.item()
    print(f"Época {epoca+1} - Loss: {total_loss:.4f}")
    

torch.save(modelo.state_dict(),'modelo.pth')