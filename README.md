# Realtime-rock-paper-scissor

Realtime Rock Paper Scissor(RPS) Game that you can actually play with the computer. Using your front-camera it detects and predicts RPS from your hand.

# Requirements:-
  1. OpenCV - Used for realtime detection of object
  2. Tensorflow 
  3. Keras - For training the model
  4. Numpy
  5. os - For working with the files in directory (inbuild library with python)
  6. sys - For command line arguments (inbuild library with python)
  7. random - Used for randomly generating numbers (inbuild library with python)

### Use the package manager [pip](https://pypi.org/project/pip/) to install any of the above modules.
  `pip install module_name`
  
# Python Implementation:-
  1. Network Used - Convolutional Neural Network(CNN) using tranfer learning
  
# Procedure:-
  1. First you have to create RPS dataset. For that, open `data.py`. When you run this file has 3 command line inputs. 
      1. Label -> This is the name of your gesture(rock/paper/scissor)
      2. Image-counter -> This counts the number of image to give each image a unique name(eg : if earlier you took 30 images of a label, then set this as 31 to start counting from that number, Note this is to ensure labeling of image is unique otherwise the photo might overlap because of same name)
      3. max_img -> The amount of image you want to take(take sufficient amount of pictures of each label and try taking in some different angles to prevent your model from overfitting the data)
  2. After the command line inputs, a window will open where you have to capture images. You have to press *space bar* for capturing an image.
  3. Repeat this all the 3 features
  4. For training the model, run `train.py` file. Here I've used tranfer learning on DenseNet121 and applied CNN on it. (Note this is will time, be patient )
  5. After training the model is stored in `rps-model.h5` file in the same directory. Also it is dumped in a json file.
  6. And now finally, for playing the game run `play.py` file. here I have set the game to 1st with 3 points wins.( You can comment and uncomment some lines(mentioned in the code) if you want the game to be of some fixed number of rounds).