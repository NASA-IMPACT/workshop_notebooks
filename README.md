# End-to-end ML using ImageLabeler and AWS SageMaker
The goal of the workshop is to familiarize you with the ImageLabeler and AWS SageMaker tool sets. In the process of completing this workshop you will have built an end-to-end machine learning (ML) solution: one that lets you label your own Earth science phenomena, as well as preprocess, train and deploy ML models within the confines of Jupyter notebooks.

## ImageLabeler
[ImageLabeler](https://impact.earthdata.nasa.gov/labeler/) facilitates the creation and management of labeled Earth science images for use in machine learning models. Using ImageLabeler, you can label images to indicate the presence or absence of a target Earth science phenomena.
Additional features include: extraction of images from satellite imagery, drawing bounding boxes, and assigning teams for labeling.

## Amazon Web Services (AWS) SageMaker
Amazon SageMaker helps data scientists and developers to prepare, build, train, and deploy high-quality ML models quickly by bringing together a broad set of capabilities purpose-built for ML. Simply put, it is a set of cloud-based (specifically, AWS) apps that focus on labeling, training, testing, and deploying models. Among other things, It provides access to provisioned Jupyter notebooks to train models and easily accessible model deployment.

## Overall workflow
1. Deploy a SageMaker instance using cdk. This process uses AWS Python CDK to deploy AWS services (Role, SageMaker instance) within the familiarity of Python.
2. Label Earth science phenomena using ImageLabeler. We will be using an example of high-latitude dust for demonstration purposes.
3. Get labeled data within SageMaker's Jupyter notebook instance and preprocess it for ML use.
4. Train a model and observe training metrics.
5. Deploy the model and infer from it.

The overall process has been divided into five chapters. Each chapter will have its own learning goal and objectives. The materials for each chapter are contained in a corresponding directory in this repository. Listed below are the sections of the workshop split into chapters.


1. Chapter-0: Creating Sagemaker instance with the AWS Cloud Development Kit
2. Chapter-1: Using ImageLabeler to create labels
3. Chapter-2: Preparing training data
4. Chapter-3: Training a model using SageMaker
5. Chapter-4: Deploying a model and inferring predictions

There are readme files for each of these chapters that gives an overview of the contents in the chapter. Each chapter has a `.ipynb` Jupyter notebook file that walks through the process and the codebase, enabling you to set up a cloud-based, end-to-end pipeline for training and deploying ML models, one chapter at a time.
