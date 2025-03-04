from sklearn.model_selection import train_test_split
from utils import reset_seeds  # 수정된 부분: utils.py에서 reset_seeds 가져오기
import lightgbm as lgb
from lightgbm import LGBMClassifier, plot_importance
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc

# 정확도, 정밀도, 재현율, f1_score 출력
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def dataset_split(df):
    y = df['Churn']
    x = df.drop(['Churn'], axis=1)

    reset_seeds()  # 수정된 부분: 시드 고정 적용
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.2, stratify=y)  # 분류형 일때 필수 : 데이터를 나눌 때 비율을 맞춰주는 것

    return x_tr, x_te, y_tr, y_te


def train_model(x_tr, x_te, y_tr, y_te):
    model = lgb.LGBMClassifier(
        random_state=42,
        subsample=0.9,
        reg_lambda=0.0,
        reg_alpha=0.2, 
        num_leaves=100,
        n_estimators=100,
        max_depth=3, 
        learning_rate=0.05
    )
    print(f'{x_tr.shape} / {y_tr.shape}')

    # 교차 검증 수행 (10개의 폴드)
    scores = cross_val_score(model, x_tr, y_tr, cv=10)
    print(f'교차 검증 정확도: {scores}')
    print(f'평균 정확도: {scores.mean():.4f}')

    model.fit(x_tr, y_tr)
    score_tr = model.score(x_tr, y_tr)
    score_te = model.score(x_te, y_te)
    print(f"Train Score : {score_tr}, Test Score : {score_te}")
    return model


def model_evaluation(model, x_te, y_te):
    y_pred = model.predict_proba(x_te)[:,1]
    y_pred_class = (y_pred >= 0.5).astype(int)  # 0.5 이상은 1, 미만은 0
    fpr, tpr, thresholds = roc_curve(y_te, y_pred)

    # auc, accuracy, precision, recall, f1_score 출력
    auc_te = auc(fpr, tpr)
    score1 = accuracy_score(y_te, y_pred_class)
    score2 = precision_score(y_te, y_pred_class)
    score3 = recall_score(y_te, y_pred_class)
    score4 = f1_score(y_te, y_pred_class)
    print(f'auc: {auc_te}, accuracy: {score1}, precision: {score2}, recall: {score3}, f1_score: {score4}')
    
    # confusion matrix 출력
    conf_mx = confusion_matrix(y_te, y_pred_class, normalize="true")
    plt.figure(figsize=(7,5))
    sns.heatmap(conf_mx, annot=True, cmap="coolwarm", linewidth=0.5)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # 특성 중요도 출력
    plot_importance(model)
    plt.show()
