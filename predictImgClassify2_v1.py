import PIL
from PIL import Image
import numpy as np
pixel = 30
numberOfClasses=4

newImage = Image.open('C://Anaconda3_500_ProgramFiles/envs/imageClassifyEnv/Images_dataset/predict/apples/apple3.jpg')
newImage = newImage.resize((pixel, pixel), PIL.Image.ANTIALIAS)
newImageArr = np.array(newImage)
newImageArr.shape
newImageArr = newImageArr.reshape(1, pixel, pixel, 3)
#print(newImageArr)


# In[51]:
def _load_label_names():
    """
    Load the label names from file
    """
    #### number of classes should be equal to length of list
    
    return ['nothing', 'apple', 'banana', 'Lemon']

#newImage
save_model_path = './image_classification_2'


# In[59]:
import pickle

test_features, test_labels = pickle.load(open('C://Anaconda3_500_ProgramFiles/envs/imageClassifyEnv/preprocess_training_2.p', mode='rb'))
#print(test_labels[0])
newdata=test_features[62]
newlabel=test_labels[0]
#print(newdata)
#print(newlabel)
        
print('reshaped data')
newdata = newdata.reshape(1, pixel, pixel, 3)
newlabel = newlabel.reshape(1, numberOfClasses)
#print(newdata.shape)
#print(newlabel.shape)
print(newlabel)
import tensorflow as tf
newdata = newImageArr
top_n_predictions = numberOfClasses #display top 4 classes only
loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        
        print(sess.run(loaded_acc, feed_dict={loaded_x: test_features, loaded_y: test_labels, loaded_keep_prob: 1.0}))
    
        predictions = sess.run(tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: newdata, loaded_y: newlabel, loaded_keep_prob: 1.0})
        
        print(predictions.indices, predictions.values)
        label_id=predictions.indices[0][0]
        #print('label ID: ',label_id)
        label_names = _load_label_names()  
        correct_name = label_names[label_id]
        print('predicted image is: ', correct_name)
        #print(sess.run(loaded_y, feed_dict={loaded_x:newdata}))
