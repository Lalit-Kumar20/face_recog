# face_recog
Finally done 
A simple Face Recognizer Based on LBPH algorithm and used selenium python module to interact with web and to open Instagram.
LBPH is simple 

Step-by-Step

Now that we know a little more about face recognition and the LBPH, let’s go further and see the steps of the algorithm:

    Parameters: the LBPH uses 4 parameters:

    Radius: the radius is used to build the circular local binary pattern and represents the radius around the central pixel. It is usually set to 1.
    Neighbors: the number of sample points to build the circular local binary pattern. Keep in mind: the more sample points you include, the higher the computational cost. It is usually set to 8.
    Grid X: the number of cells in the horizontal direction. The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector. It is usually set to 8.
    Grid Y: the number of cells in the vertical direction. The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector. It is usually set to 8.

Don’t worry about the parameters right now, you will understand them after reading the next steps.

2. Training the Algorithm: First, we need to train the algorithm. To do so, we need to use a dataset with the facial images of the people we want to recognize. We need to also set an ID (it may be a number or the name of the person) for each image, so the algorithm will use this information to recognize an input image and give you an output. Images of the same person must have the same ID. With the training set already constructed, let’s see the LBPH computational steps.

3. Applying the LBP operation: The first computational step of the LBPH is to create an intermediate image that describes the original image in a better way, by highlighting the facial characteristics. To do so, the algorithm uses a concept of a sliding window, based on the parameters radius and neighbors.


Then after recognizing the face , it opens your instagram by using selenium python module.
