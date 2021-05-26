
import joblib
import pandas as pd
import numpy as np
import sys

def movementDetection(data):
    loaded_model = joblib.load('model.pkl')
    predictions = loaded_model.predict(data)
    print(predictions[0])


    
def main():
    # testing = (pd.read_csv("./Dataset/testdata.csv"))
    data = sys.argv[1].split(',')
    # my_array = np.array([[2.29, 9.09, 0.59, -0.06, 0.08 ,-0.11]])
    my_array = np.array([data])
    testing = pd.DataFrame(my_array, columns = ['ACC X','ACC Y','ACC Z','GYRO X','GYRO Y','GYRO Z'])
    # print(testing)
    movementDetection(testing)


if __name__ == "__main__":
    main()