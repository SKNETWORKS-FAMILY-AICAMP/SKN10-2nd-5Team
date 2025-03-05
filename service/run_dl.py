from data import load_data
from preprocess_dl import *
from process_dl import *
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

dl_set_seed()
# 1. 데이터 불러오기
df = load_data()
print("데이터 로드 후 : ",df.shape)
print(df.columns)
df = create_custom_features(df)
print("특성 추가 후 : ",df.shape)
print(df.columns)
df = cleaning_data(df)
print("클리닝 후 : ",df.shape) 
print(df.columns)
df = encode_data(df)
print("원핫인코딩 후 : ",df.shape)
print(df.columns)
X,y = smote_data(df)
print("smote 적용 후 : ",df.shape)
print(df.columns)

# 2. 데이터 전처리
# 모델 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. 모델 초기화
model = MultiModel(X.shape[1],len(set(y))).to(device)
print(X.shape[1],len(set(y)))

# 4. 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 데이터셋을 훈련(train)과 검증(validation)으로 분할
batch_size = 32
dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
train_size = int(0.8 * len(dataset))  # 80% 훈련 데이터
val_size = len(dataset) - train_size  # 20% 검증 데이터
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 6. 모델 학습
epochs = 50  # Early Stopping으로 조기 종료 가능
train_loss, val_loss, train_acc, val_acc = train_loop(model, train_loader, val_loader, criterion, optimizer, epochs, device=device, patience=7)

# 7. Loss 및 Accuracy 그래프 출력
plot_metrics(train_loss, val_loss, train_acc, val_acc)

# 8. Test Accuracy 확인
test_loss, test_acc = test_loop(model, X, y, criterion, device=device)

# 9. 결과 출력력
X_new_sample = X[:100]
predictions = predict(model, X_new_sample)
print("예측 결과:", predictions)

#  실제값과 예측값 비교 실행
analyze_predictions(y[:100], predictions)