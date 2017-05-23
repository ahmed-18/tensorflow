# Description
Modification of tf.contrib.learn.Estimator to prevent reloading model on each predict call.

# Usage

1. Take backup of python2.7/dist-packages/tensorflow/contrib/learn/python/learn/estimators/estimator.py
2. Replace original estimator.py with the provided file.
3. Everything is same except a new function (predict_2) is added to speed up classification and it works exactly the same as the original predict function.
4. It's just a workaround, use at your own risk. The function was written to prevent reloading model.
5. The modified funtion just loads the model at first call and uses the cached version at next calls. It was originally written to get real time detections on webcam videos i.e. capturing a frame using webcam and classifying it in real time.
6. It was tested with TensorFlow 1.0.1 

# Example
'''
 #provide your model_fn and model_dir to initialize Estimator
 classifier = tf.contrib.learn.Estimator(
        model_fn=cnn_model_fn, model_dir=model_path)
 #the function assumes x has only 1 test sample
 predictions = classifier.predict_2(x, as_iterable=False)
 cls = predictions['classes'][0]
 pred_class = mapping[cls]
 prob = np.max(predictions['probabilities'][0][cls])
'''
