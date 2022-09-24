import numpy as np
import streamlit as st 
import pickle
import tensorflow as tf
from keras.preprocessing import image
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
st.set_option('deprecation.showfileUploaderEncoding', False)
# Loading saved model from Drive.
from keras.models import load_model
model = load_model('FDPCNN1.h5')

html_temp = """
   <div class="" style="background-color:#33332a;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">CNN(convolutional neural network)</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Cat and Dog classification using CNN</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.title("""
        Image Classification Cat/ Dog
         """
         )
file= st.file_uploader("Please upload image", type=("jpg", "png"))

import cv2
from  PIL import Image, ImageOps
def import_and_predict(image_data):
  #x = cv2.resize(image_data, (48, 48)) 
  #img = image.load_img(image_data, target_size=(48, 48))
  #x = image.img_to_array(img)
  size=(64, 64)
  image=ImageOps.fit(image_data, size, Image.ANTIALIAS)
  img=np.asarray(image)
  img_reshape=np.expand_dims(img, axis=1)
  img_reshape=img[np.newaxis,...]
  result = model.predict(img_reshape)
  print(result)
  #training_set.class_indices
  if result[0][0] == 1:
    prediction = "Dog" 
    print(prediction)
  else:
    prediction = 'Cat'
    print(prediction)#x = np.expand_dims(x, axis=1)
  
  
  return prediction
if file is None:
  st.text("Please upload an Image of Cat/ Dog")
else:
  image=Image.open(file)
  #image=np.array(image)
  #file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
  #image = cv2.imdecode(file_bytes, 1)
  st.image(image,caption='Uploaded Image.', use_column_width=True)
    
if st.button("Predict Cat/Dog"):
  result=import_and_predict(image)
  st.success('Model has predicted the image  is  of  {}'.format(result))
if st.button("About"):
  st.header("Nivesh Soni")
  st.subheader("Email: niveshsoni9251@gmail.com")

  
html_temp = """
   <div class="" style="background-color:#33332a;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Image Classification</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
