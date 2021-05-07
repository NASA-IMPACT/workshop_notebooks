#!/usr/bin/env python3
import os

from aws_cdk import core

from stacks.vpc_stack import VpcStack
from stacks.iam_stack import IamStack
from stacks.sagemaker_stack import SageMakerStack

enviroment = core.Environment(
    account=os.environ["CDK_DEFAULT_ACCOUNT"],
    region=os.environ["CDK_DEFAULT_REGION"]
)
app = core.App()
vpc_stack = VpcStack(app, "VpcStack", env=enviroment)
iam_stack = IamStack(app, "IamStack", env=enviroment)
sagemaker_stack = SageMakerStack(
    app,
    "SagemakerStack",
    vpc=vpc_stack.vpc,
    role=iam_stack.notebook_role,
    subnet=vpc_stack.private_subnet,
    env=enviroment
)

app.synth()
