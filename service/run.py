from data import load_data
from preprocess import preprocess_dataset
from process import train_model

def main():
    total_data = load_data()
    print('Data Load')

    X_train,X_test,y_train,y_test = preprocess_dataset(total_data)
    print('Data Preprocessing')

    accuracy = train_model(X_train, X_test, y_train, y_test)
    print(f'Accuracy: {accuracy}')

    return accuracy

if __name__ == "__main__":
    main()