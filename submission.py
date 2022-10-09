# DO NOT RENAME THIS FILE
# This file enables automated judging
# This file should stay named as `submission.py`

# Import Python Libraries
import numpy as np
from glob import glob
from PIL import Image
import cv2
import tensorflow as tf

# Import helper functions from utils.py
import utils

class Predictor:
    """
    DO NOT RENAME THIS CLASS
    This class enables automated judging
    This class should stay named as `Predictor`
    """

    def __init__(self):
        """
        Initializes any variables to be used when making predictions
        """
        self.interpreter = tf.lite.Interpreter(model_path="model_v2.tflite", num_threads = 5)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.interpreter.allocate_tensors()

        self.categories = ['0213', '1230', '0123', '3120', '1302', '1032', '2301', '3021', '0312', '3210', '0321', '2031', '1203', '3012', '2130', '2013', '0231', '1320', '3102', '2310', '0132', '2103', '1023', '3201']

    def make_prediction(self, img_path):
        """
        DO NOT RENAME THIS FUNCTION
        This function enables automated judging
        This function should stay named as `make_prediction(self, img_path)`

        INPUT:
            img_path: 
                A string representing the path to an RGB image with dimensions 128x128
                example: `example_images/1.png`
        
        OUTPUT:
            A 4-character string representing how to re-arrange the input image to solve the puzzle
            example: `3120`
        """

        # Load the image
        img = cv2.imread(img_path)

        #Resize the image
        new_img = cv2.resize(img, (224, 224))

        #Run the model
        self.interpreter.set_tensor(self.input_details[0]['index'], [new_img])
        self.interpreter.invoke()

        #Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        return self.categories[np.argmax(output_data)]



# Example main function for testing/development
# Run this file using `python3 submission.py`
if __name__ == '__main__':

    for img_name in glob('example_images/*'):
        # Open an example image using the PIL library
        example_image = Image.open(img_name)

        # Use instance of the Predictor class to predict the correct order of the current example image
        predictor = Predictor()
        prediction = predictor.make_prediction(img_name)
        # Example images are all shuffled in the "3120" order
        print(prediction)

        # Visualize the image
        pieces = utils.get_uniform_rectangular_split(np.asarray(example_image), 2, 2)
        # Example images are all shuffled in the "3120" order
        final_image = Image.fromarray(np.vstack((np.hstack((pieces[3],pieces[1])),np.hstack((pieces[2],pieces[0])))))
        final_image.show()