from data import load_data
from preprocess import preprocess_dataset
from process import process
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
def main():
    total_data = load_data()
    (x_train,x_test,y_train,y_test) = preprocess_dataset(total_data)
    model = process(x_train,y_train)


    pre=model.predict(x_test)
    print(f"Accuracy {accuracy_score(y_test,pre)}")
    print(f"f1score {f1_score(y_test,pre)}")
    cm = confusion_matrix(y_test, pre, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
    disp.plot()
    plt.show()



if __name__ == "__main__":
    main()
