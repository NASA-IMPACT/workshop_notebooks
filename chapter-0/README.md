# Creating Sagemaker instance with AWS Cloud Development Kit
AWS Cloud Development Kit (AWS CDK) is a framework that lets the users define and provision AWS cloud resources using languages like Python, Java, etc.

This Module (chapter-0) familiarizes users with the CDK by deploying a sagemaker instance using CDK.

## Step 1: Configure AWS CLI in your machine

Install AWS command-line interface using steps shown [here](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html).

Configure AWS with your credentials and region:

```
aws configure
```
Provide Access key ID, Secret Access Key, and the region information.

## Step 2: Install CDK

Install AWS CDK Toolkit using npm.
```
npm install -g aws-cdk
```

Install required Python 3.6+, pip and virtualenv packages. activate the virtualenv:
```
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install --upgrade virtualenv
source .venv/bin/activate
```

Install required CDK packages in requirements.txt:
```
pip install -r requirements.txt
```

## Step 3: Deploy the CDK stack
AWS CDK apps are composed of building blocks known as Constructs, which are composed together to form stacks and apps.
Stacks in AWS CDK apps extend the Stack base class, as shown in ``deploy/stacks/iam_stack.py``. This is a common pattern when creating a stack within your AWS CDK app: extend the Stack class, define a constructor that accepts scope, id, and props, and invoke the base class constructor via super with the received scope, id, and props. (from https://docs.aws.amazon.com/cdk/latest/guide/constructs.html)

Once we have defined a stack, we can populate it with resources by instantiating constructs. After we instantiate a construct, the construct object exposes a set of methods and properties that enables us to interact with the construct and pass it around as a reference to other parts of the system.  All constructs that represent AWS resources must be defined, directly or indirectly, within the scope of a Stack construct.

For example, in line 11 of ``deploy/stacks/iam_stack.py``, an `aws_iam.Role` construct is instantiated with role name and the service that needs to assume this role.

To define the stacks within the scope of an application, we use the App construct, as shown in `deploy/app.py`. This file instantiates the defined stacks within the scope of the application, and links them together in one context. Once all the stacks are linked to the app, `app.synth()` is called to synthesize a cloud assembly.

Furthermore, CDK Toolkit needs to know how to execute the AWS CDK app. for Python, we create a `cdk.json` file that includes the necessesary command for the CDK to initialiize the app.

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