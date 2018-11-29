import PIL
from PIL import Image
import numpy as np
dim = 32
newImage = Image.open('C://Users/583175/Pictures/test/sp7.jpg')
newImage = newImage.resize((dim, dim), PIL.Image.ANTIALIAS)
newImageArr = np.array(newImage)
newImageArr.shape
newImageArr = newImageArr.reshape(1, 32, 32, 3)
newImageArr.shape


# In[51]:
def _load_label_names():
    """
    Load the label names from file
    """
    
   # 'aquatic mammals',  'fish',  'flowers',  'food containers',  'fruit and vegetables',  
#'household electrical devices',  
    #'household furniture',  'insects',  'large carnivores',  'large man-made outdoor things',  
    #'large natural outdoor scenes',  
    #'large omnivores and herbivores',  'medium-sized mammals',  'non-insect invertebrates', 
    #'people',  'reptiles',  'small mammals',  'trees',  'vehicles 1',  'vehicles 2'
    
    
    return ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm']

#newImage
save_model_path = './image_classification_100'


# In[59]:
import pickle

test_features, test_labels = pickle.load(open('C://Anaconda3_500_ProgramFiles/envs/imageClassifyEnv/preprocess_training_100.p', mode='rb'))
print(test_labels[0])
newdata=test_features[100]
newlabel=test_labels[100]
#print(newdata)
#print(newlabel)
        
print('reshaped data')
newdata = newdata.reshape(1, 32, 32, 3)
newlabel = newlabel.reshape(1, 100)
#print(newdata.shape)
#print(newlabel.shape)
#print(newlabel)
import tensorflow as tf
newdata = newImageArr
top_n_predictions = 3
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
        
        #print(sess.run(loaded_acc, feed_dict={loaded_x: newdata, loaded_y: newlabel, loaded_keep_prob: 1.0}))
    
        predictions = sess.run(tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: newdata, loaded_y: newlabel, loaded_keep_prob: 1.0})
        
        print(predictions.indices, predictions.values)
        label_id=predictions.indices[0][0]
        #print('label ID: ',label_id)
        label_names = _load_label_names()  
        correct_name = label_names[label_id]
        print('predicted image is: ', correct_name)
        #print(sess.run(loaded_y, feed_dict={loaded_x:newdata}))
