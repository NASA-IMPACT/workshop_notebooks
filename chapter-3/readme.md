# Chapter-3: Train model using SageMaker
The objectives you complete during the course of this chapter introduce you to the process of implementing the SageMaker model training tool. You will engage in this process by completing the following objectives:
Create a neural network to detect high latitude dust events by reviewing and executing the HLD SageMaker codebase.
Implement  the attributes required by the HLD SageMaker codebase.
Review and execute the code needed to set up the HLD SageMaker

As with chapter 2, a subset of High Latitude Dust data is pre-prepared and uploaded into a s3 bucket (impact-datashare). The details can be found here. Once again, the code contains details on the necessary import elements and specifies which variables need to be assigned predetermined values. Note that the image_url function must be used to create a valid url, if the shapefile generation was not done in Aqua, TrueColor.

The code again assumes the notebookAccessRole role that was created using the Amazon Web Services Cloud Development Kit (AWS CDK). Note the specific constant variables that are required for this process as well as the specific values returned.

Four of the helper agents described in the chapter 2 Readme file are created:
- mkdir
- delete_folder
- create_split
- prepare_splits

The code instantiates TensorFlow from SageMaker. It then uploads the training, test, and validation images into SageMaker’s TensorFlow. The fit method of TensorFlow is run on the training, test, and validation images.
