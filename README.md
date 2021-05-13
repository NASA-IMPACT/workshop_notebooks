# End-to-end ML using ImageLabeler and AWS SageMaker
The goal of the workshop is to familiarize you with the ImageLabeler and AWS SageMaker tool sets. In the process of completing this workshop you will have built an end-to-end machine learning (ML) solution: One that lets you label your own Earth Science phenomena, preprocess, train and deploy ML models within the confines of jupyter notebooks.

## ImageLabeler
[Image Labeler](https://impact.earthdata.nasa.gov/labeler/) facilitates the creation and management of labeled Earth science images for use in machine learning models. Using Image Labeler, you can label images to indicate the presence or absence of a target Earth science phenomena.
Additional features include: extraction of images from satellite imagery, drawing bounding boxes, and assigning teams for labeling.

## AWS SageMaker
Amazon SageMaker helps data scientists and developers to prepare, build, train, and deploy high-quality machine learning (ML) models quickly by bringing together a broad set of capabilities purpose-built for ML.
Simply put, It is a set of cloud based (specifically, AWS) apps that focus on labeling, training, testing and deploying models. Among other things, It provides access to provisioned jupyter notebooks to train models and easy and accessible model deployment.

## Overall workflow
1. Deploy a sagemaker instance using cdk. This process uses AWS Python CDK to deploy AWS services (Role, SageMaker instance) within the familiarity of python.
2. label earth science phenomena using imagelabeler. we will be using example of High-Latitude dust for demo purposes.
3. Get labeled data within sagemaker's jupyter notebook instance, preprocess it for ML use.
4. Train a model and observe training metrics
5. deploy the model and infer from it.

The overall process has been divided into five discrete chapters. Each chapter will have its own learning goal and objectives. The materials for each chapter are contained in a corresponding directory in this repository. Listed below are the sections of the workshop split into chapters.


1. Chapter-0: Creating Sagemaker instance with AWS Cloud Development Kit
2. Chapter-1: Demonstrate the usage of ImageLabeler to create Labels
3. Chapter-2: Prepare training data
4. Chapter-3: Train model using SageMaker
5. Chapter-4: Deploy model and Infer Predictions

There are readme files for each of these chapters that gives an overview of the contents in the chapter. Each chapter has a `.ipynb` jupyter notebook file that walks through the process and the codebase, enabling you to set up a cloud based End-to-end pipeline for training and deploying ML models, one chapter at a time.
