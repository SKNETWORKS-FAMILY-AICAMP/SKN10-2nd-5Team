from data import load_data
from preprocess import preprocess_dataset

def main():
    total_data = load_data()
    (x_train,x_test,y_train,y_test) = preprocess_dataset(total_data)
    





if __name__ == "__main__":
    main()
