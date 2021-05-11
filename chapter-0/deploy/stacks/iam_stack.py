from aws_cdk import core
from aws_cdk import (
    aws_iam as iam_,
    aws_ec2 as ec2
)

class IamStack(core.Stack):
    def __init__(self, scope: core.Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self._notebook_role = iam_.Role(
            self,
            "notebookAccessRole",
            role_name="notebookAccessRole",
            assumed_by=iam_.ServicePrincipal('sagemaker')
        )

        self._notebook_policy = iam_.Policy(
            self,
            "notebookAccessPolicy",
            policy_name="notebookAccessPolicy",
            statements=[
                iam_.PolicyStatement(
                    actions = ['s3:*'],
                    resources=['*']
                ),
                iam_.PolicyStatement(
                    actions = ["logs:*"],
                    resources=['*']
                ),
                iam_.PolicyStatement(
                    actions = ["sagemaker:*"],
                    resources=['*']
                ),
                iam_.PolicyStatement(
                    actions = ["ecr:*"],
                    resources=['*']
                ),
                iam_.PolicyStatement(
                    actions = [
                        "iam:GetRole",
                        "iam:PassRole",
                        "sts:GetSessionToken",
                        "sts:GetAccessKeyInfo",
                        "sts:GetCallerIdentity",
                        "sts:GetServiceBearerToken",
                        "sts:DecodeAuthorizationMessage",
                        "sts:AssumeRole"
                    ],
                    resources=['*']
                )
            ]
        ).attach_to_role(self._notebook_role)


    @property
    def notebook_role(self):
        return self._notebook_role


    @property
    def notebook_policy(self):
        return self._notebook_policy
