import tensorflow;
from tensorflow import keras;
import cv2;

dataset = keras.datasets.mnist;
(xTrain, yTrain), (xTest, yTest) = dataset.load_data();
print(f"XTrain : {xTrain},\n YTrain {yTrain},\n XTest : {xTest},\n YTest : {yTest};");
xTrain, xTest = xTrain / 255.0, xTest / 255.0;

image = xTrain[0];

for i in range(10) : 
    image = xTrain[i];
    cv2.imshow(f"{yTrain[i]};", image);

if(cv2.waitKey(0) == ord("x")) : 
    cv2.destroyAllWindows();
    print("Closed!");

model = tensorflow.keras.models.Sequential([
    tensorflow.keras.layers.Flatten(input_shape=(28, 28)), 
    tensorflow.keras.layers.Dense(128, activation="relu"), 
    tensorflow.keras.layers.Dropout(0.2), 
    tensorflow.keras.layers.Dense(10, activation="softmax")
]);


#model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]);
'''model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]);
model.fit(xTrain, yTrain, epochs=5);
model.evaluate(xTest, yTest);'''