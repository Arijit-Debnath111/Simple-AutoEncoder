#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


# In[2]:


(x_train,_),(x_test, _) = mnist.load_data()


# In[3]:


x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)


# In[4]:


model = Sequential()
model.add(Input(shape = (784,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(784))


# In[5]:


model.compile(optimizer="adam",loss="mean_squared_error",metrics  = "mean_squared_error")


# In[6]:


model.fit(x=x_train,y=x_train,epochs=20,batch_size=1000,validation_data = (x_test,x_test))


# In[7]:


train_loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
train_accuracy = model.history.history['mean_squared_error']
validation_accuracy = model.history.history['val_mean_squared_error']


# In[8]:


plt.plot(train_loss, label='Train')
plt.plot(val_loss,label='Validation')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.legend()
plt.grid()
plt.show()


# In[9]:


model.summary()


# In[10]:


model.layers[:-3]


# In[11]:


# Encoder


# In[12]:


encoder = Sequential()
for layer in model.layers[:-3]:
    encoder.add(layer)


# In[13]:


encoder.build(input_shape=(None,784))
encoder.summary()


# In[14]:


# Decoder


# In[15]:


decoder = Sequential()
for layer in model.layers[-3:]:
    decoder.add(layer)


# In[16]:


decoder.build(input_shape=(None,32))
decoder.summary()


# In[17]:


# Example


# In[18]:


plt.imshow(x_test[0].reshape(28,28),cmap='gray') # original image
plt.show()


# In[19]:


code = encoder.predict(x_test) # Compressed representation (32 units)
code[0]


# In[20]:


pred = decoder.predict(code)[0] # reconstructed image 
plt.imshow(pred.reshape(28,28),cmap='gray') # original image
plt.show()


# In[ ]:




