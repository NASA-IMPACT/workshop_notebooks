from aws_cdk import core
from aws_cdk import (aws_ec2 as ec2, aws_sagemaker as sm, aws_iam as iam_)
from glob import glob


class SageMakerStack(core.Stack):

    def __init__(
        self,
        scope: core.Construct,
        construct_id: str,
        vpc: ec2.Vpc,
        role: iam_.Role,
        subnet: ec2.PrivateSubnet,
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.security_group = ec2.SecurityGroup(
            self,
            "WorkshopGroup",
            vpc=vpc
        )

        instance_id = "Workshop"
        sm.CfnNotebookInstance(
            self,
            instance_id,
            instance_type='ml.t2.medium',
            volume_size_in_gb=20,
            security_group_ids=[self.security_group.security_group_id],
            subnet_id=subnet.subnet_id,
            notebook_instance_name=instance_id,
            role_arn=role.role_arn,
            direct_internet_access='Enabled',
            root_access='Enabled',
            default_code_repository="https://github.com/NASA-IMPACT/workshop_notebooks"
        )
