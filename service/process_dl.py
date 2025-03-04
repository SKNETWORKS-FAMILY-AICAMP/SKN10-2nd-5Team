import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class MultiModel(nn.Module):
    def __init__(self, input_size, out_size, hidden_size=32):
        super().__init__()
        self.lieanr_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, out_size)
        )

    def forward(self, x):
        return self.lieanr_stack(x)


# 4. Early Stopping 클래스
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 5
            if self.counter >= self.patience:
                self.early_stop = True  #  조기 종료

#  5. Training Loop (Validation Accuracy 기준으로 모델 저장)
def train_loop(model, train_loader, val_loader, criterion, optimizer, epochs=20, device='cpu', patience=5):
    model.train()
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_val_acc = 0.0  #  가장 높은 Validation Accuracy 저장
    best_model_path = "model/best_model.pth"  #  최적 모델 저장 경로
    early_stopping = EarlyStopping(patience=patience)

    for epoch in tqdm(range(epochs), desc="Training Progress", leave=True):
        total_loss, correct, total = 0, 0, 0

        for batch_X, batch_y in train_loader :
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 정확도 계산
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total * 100
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_accuracy)

        #  검증 데이터 평가 (Validation)
        model.eval()
        val_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                # 정확도 계산
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total * 100
        val_loss_history.append(avg_val_loss)
        val_acc_history.append(val_accuracy)

        tqdm.write(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        #  Validation Accuracy가 최고라면 모델 저장
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), best_model_path)  #  최적 모델 저장
            print(f" 모델 저장: Epoch {epoch+1}, Best Val Acc: {best_val_acc:.2f}%")

        #  Early Stopping 체크
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return train_loss_history, val_loss_history, train_acc_history, val_acc_history

#  8. Loss 및 Accuracy 그래프 출력
def plot_metrics(train_loss, val_loss, train_acc, val_acc):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(train_loss, label='Train Loss', linestyle="-", marker="o")
    axs[0].plot(val_loss, label='Val Loss', linestyle="--", marker="s")
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Train vs Validation Loss')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(train_acc, label='Train Accuracy', linestyle="-", marker="o")
    axs[1].plot(val_acc, label='Val Accuracy', linestyle="--", marker="s")
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].set_title('Train vs Validation Accuracy')
    axs[1].legend()
    axs[1].grid(True)

    plt.show()

def test_loop(model, X_test, y_test, criterion, device='cpu'):
    best_model_path = "model/best_model.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        loss = criterion(outputs, y_test_tensor)

        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor) * 100

    print(f" 최적 Validation 모델을 사용한 Test Accuracy: {accuracy:.2f}%")
    return loss.item(), accuracy

def predict(model, X_new, device='cpu'):
    best_model_path = "model/best_model.pth"

    #  가중치만 로드하도록 변경 (보안 권장)
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()

    X_new_tensor = torch.tensor(X_new, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X_new_tensor)
        _, predictions = torch.max(outputs, 1)

    return predictions.cpu().numpy()

def analyze_predictions(y_true, y_pred):
    # DataFrame으로 정리하여 보기 쉽게 출력
    df_results = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Correct': (y_true == y_pred)  # 맞으면 True, 틀리면 False
    })

    # 정확도 계산
    accuracy = np.mean(df_results['Correct']) * 100  # 퍼센트 변환
    correct_count = df_results['Correct'].sum()
    total_count = len(df_results)

    print(f"\n예측 정확도: {accuracy:.2f}% ({correct_count}/{total_count} 개 맞음)")