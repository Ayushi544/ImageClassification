# ImageClassification
This project aims to build a basic image classification model to classify images of Indian ships into two categories:

Class 1: Army ships
Class 2: Commercial ships
Using data augmentation and pre-trained models like ResNet50, I have trained the model to recognize and classify these two types of ships. This project is designed for beginners to help me learn the basics of image classification, data preprocessing, and model evaluation in deep learning.

Key Concepts
Data Augmentation: To make the model more robust, I applied data augmentation techniques such as rotation, zoom, and flipping to artificially increase the size and diversity of the dataset.
Transfer Learning: The project uses a pre-trained ResNet50 model (trained on ImageNet) as a base to save time and computational resources while achieving better accuracy.
Model Evaluation: I employed accuracy, precision, recall, and F1-score to evaluate the performance of the model. Confusion matrix is also used to understand the model's performance on individual classes.

Data Preprocessing
The dataset consists of images of ships, divided into two categories:

Class 1 (Army ships): Contains images of various army ships.
Class 2 (Commercial ships): Contains images of different commercial ships.
The data is split into train and validation sets with an equal number of images in both classes.

Data augmentation is applied to the images to:

Increase the variety of images available for training by applying transformations like:
Rotation
Zoom
Flip
Rescale

Model Architecture
The project uses the ResNet50 model as a base for transfer learning:

Base Model: ResNet50 pre-trained on ImageNet.
Added Layers:
GlobalAveragePooling2D: To reduce dimensionality.
Dense: Fully connected layer with 256 units and ReLU activation.
Dropout: To prevent overfitting.
Dense (Output): A sigmoid layer for binary classification.
Model Training
The model is trained using binary cross-entropy loss and Adam optimizer. I ran the training for 6 epochs on a small dataset, which showed fluctuating results in accuracy. Further improvements can be made by adjusting the epochs, class balancing, and fine-tuning the model.

Model Evaluation
After training the model, I evaluated its performance using various metrics:

Accuracy: The overall accuracy of the model.
Precision and Recall: Used to understand the model's performance on individual classes, especially the Class 1 (army ships) category, which had fewer samples.
F1-Score: The harmonic mean of precision and recall, used to get a balanced measure of the model's performance.
The confusion matrix was used to visualize:

True Positives (TP): Correctly predicted images of each class.
False Positives (FP): Misclassified images from Class 1 as Class 2, and vice versa.
False Negatives (FN): Missed predictions for both classes.
Results
The model initially showed accuracy fluctuating during training, with validation accuracy remaining constant at around 60%. 


Future Improvements
Fine-tuning: Unfreeze some layers of the ResNet50 model to allow it to learn more specific features.
Class Balancing: Use techniques like oversampling, undersampling, or class weights to address class imbalance.
Additional Augmentation: Explore more augmentation techniques to further diversify the training dataset.
Hyperparameter Tuning: Adjust learning rates, epochs, and other parameters to improve performance.
Conclusion
This project gave me a hands-on introduction to deep learning with image classification. While the initial results were basic, I learned the importance of data preprocessing, model evaluation, and experimentation with different techniques to improve the model. I look forward to continuing my journey and further refining the model!

Acknowledgements
Dataset: Indian ships dataset from Kaggle.
Libraries: TensorFlow, Keras, Matplotlib.
