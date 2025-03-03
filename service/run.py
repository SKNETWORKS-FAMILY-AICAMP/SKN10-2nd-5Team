from data import load_data
from preprocess import preprocess_dataset
from process import run_training

import joblib
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, roc_auc_score,
    classification_report
)


def main():
    # 원본 데이터셋 가져오기
    raw_data = load_data()

    # 전처리 데이터셋 가져오기
    X_train, X_test, y_train, y_test = preprocess_dataset(raw_data)

    # 모델 학습 및 저장
    best_model = run_training(X_train, y_train)
    print("✅ 모델 학습 및 저장 완료!")

    # 저장된 모델 로드
    loaded_model = joblib.load("models/best_model.pkl")
    print("🚀 모델을 로드했습니다.")

    # 테스트 데이터 평가
    test_preds = loaded_model.predict(X_test)

    # 🔹 다양한 성능 지표 계산
    test_f1 = f1_score(y_test, test_preds)
    test_accuracy = accuracy_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds)
    test_recall = recall_score(y_test, test_preds)
    test_roc_auc = roc_auc_score(y_test, test_preds)

    # 🔹 출력
    print("\n🎯 최종 테스트 데이터 성능 평가")
    print(f"📌 정확도 (Accuracy): {test_accuracy:.4f}")
    print(f"📌 정밀도 (Precision): {test_precision:.4f}")
    print(f"📌 재현율 (Recall): {test_recall:.4f}")
    print(f"📌 F1-score: {test_f1:.4f}")
    print(f"📌 ROC AUC Score: {test_roc_auc:.4f}")

    # 🔹 자세한 분류 보고서 출력
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, test_preds))


if __name__ == "__main__":
    main()
