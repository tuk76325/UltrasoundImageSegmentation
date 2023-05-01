# Ultrasound Image Segmentation

In its entirety, this project introduced me to deep learning, image segmentation, and their applications in the biomedical field. 
The project began with an outline of how I would approach the problem of image segmentation in ultrasound images, such that a specific structure could be 
identified. Beginning with research on image segmentation, I found that it was necessary to utilize deep learning (via TensorFlow and Keras) to fulfill my goal. 
However, there were many problems that arose from this, beginning with my lack of familiarity on the topic.
	The first step towards completing this project was generating separate lists filled with the training images, training masks, and test images. This 
required the use of the os library. Following this, each of the image files were appended to two pandas dataframes, one for training and the other for testing. 
Next, the image file and the mask file were randomly selected at the same index to ensure that the training set of images corresponded to the mask file. 
Afterwards, the images needed to be preprocessed for insertion in the model. The model required images to be square, and as a result, the library Pillow was 
used to add white bars and vertically center the images and masks. Furthermore, the images also needed to be in the size of 128 x 128 x 3 to be able to be read 
into the convolutional neural network. To accomplish this, the images were converted to a 2D array using CV2 and their dimensions were expanded by use of numpy. 
Finally the data was ready to be inserted into the model. After coding the CNN layers, there ended up being 1,941,105 trainable parameters, and they were 
analyzed over 10 epochs with a batch size of 32, and a validation split of 0.1. It was also worthy to note that the activation used was relu, the loss function 
with binary cross entropy, and the optimizer chosen was adam. These parameters were chosen because they were what had been used in papers and media which had 
done similar projects. 
	After training was completed, the model accuracy and loss were visualized in the results section of this report. Then, the model predicted the results 
of the test dataset.
	In summation, this project was the culmination of many hours of work towards learning a new data science skill. From outlining to the actual 
implementation, this project familiarized myself with many new libraries and gave me a greater understanding of how projects should be undertaken. This topic 
itself seems very applicable to industry and my career as it allowed me to witness the creation and finish of a project in real time.

