# IEEE-GRSS Summer School on Scaling Machine Learning for Remote Sensing using Cloud Computing Environment
### Using  Imagelabeler labeling tool and AWS Sagemaker
The AWS Cloud Development Kit (AWS CDK) is a framework that lets users define and provision AWS cloud resources using languages like Python, Java, etc.

This module (chapter-0) familiarizes users with the AWS CDK by deploying a SageMaker instance using the AWS CDK.

## Step 1: Redeem your aws credits
You will be provided [AWS credits](https://aws.amazon.com/awscredits/) required for running the services that are used in this course. Please reach out to instructors over slack for your personal credits code.

## Step 2: Configure AWS command-line interface (CLI) on your machine

Install the AWS CLI using steps shown [here](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html).

Configure AWS with your credentials and region:

```
aws configure
```
Provide access key ID, secret access key, and the region information (default: `us-east-1`) provided to you by the instructors.

## Step 3: Install AWS

First install [node + npm](https://nodejs.org/en/). 

Then install the AWS CDK Toolkit using npm.
```
npm install -g aws-cdk
```

Install required [Python 3.6+](https://www.python.org/downloads/) and [virtualenv packages](https://virtualenv.pypa.io/en/latest/installation.html#via-pip). Activate the virtualenv:
```
cd chapter-0/
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade virtualenv
python3 -m venv .venv
source .venv/bin/activate
```

Install required AWS CDK packages in requirements.txt:
```
pip install -r requirements.txt
```

## Step 4: Deploy the AWS CDK stack
AWS CDK apps are composed of building blocks known as constructs, which are composed together to form stacks and apps.
Stacks in AWS CDK apps extend the Stack base class, as shown in ``deploy/stacks/iam_stack.py``. This is a common pattern when creating a stack within your AWS CDK app: a) extend the Stack class, b) define a constructor that accepts scope, id, and props, and c) invoke the base class constructor via super with the received scope, id, and props. (from https://docs.aws.amazon.com/cdk/latest/guide/constructs.html)

Once we have defined a stack, we can populate it with resources by instantiating constructs. After we instantiate a construct, the construct object exposes a set of methods and properties that enable us to interact with the construct and pass it around as a reference to other parts of the system.  All constructs that represent AWS resources must be defined, directly or indirectly, within the scope of a stack construct.

For example, in line 11 of ``deploy/stacks/iam_stack.py``, a `aws_iam.Role` construct is instantiated with a role name and the service that needs to assume this role.

To define the stacks within the scope of an application, we use the App construct, as shown in `deploy/app.py`. This file instantiates the defined stacks within the scope of the application and links them together in one context. Once all the stacks are linked to the app, `app.synth()` is called to synthesize a cloud assembly.

Furthermore, the AWS CDK Toolkit needs to know how to execute the AWS CDK app. For Python, we create a `cdk.json` file that includes the necessesary command for the CDK to initialiize the app.

cdk.json:
```
{
    "app": "python app.py"
}
```
Finally, the CDK pipeline can be deployed using the following command:

```
cdk deploy --all
```
