from mediapipe_model_maker import gesture_recognizer;

data = gesture_recognizer.Dataset.from_folder(
    dirname = "Models", 
    hparams = gesture_recognizer.HandDataPreprocessingParams()
);

train_data, test_data = data.split(0.8);
validation, test = test_data.split(0.5);

hparams = gesture_recognizer.HParams(export_dir="gesture_recognizer_test_model");
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams);
model = gesture_recognizer.GestureRecognizer.create(
    train_data = train_data, 
    validation_data = validation, 
    options = options
);

loss, accuracy = model.evaluate(test, batch_size = 1);
model.export_model();
