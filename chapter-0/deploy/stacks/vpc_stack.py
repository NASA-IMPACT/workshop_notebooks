from aws_cdk import core
from aws_cdk import (
    aws_ec2 as ec2
)

class VpcStack(core.Stack):
    def __init__(self, scope: core.Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self._vpc = ec2.Vpc.from_lookup(
            self,
            "VPC",
            is_default=True
        )

        ip = self._vpc.vpc_cidr_block.split('/')[0].split('.')
        ip[2] = '160'
        ip = ".".join(ip)

        self._private_subnet = ec2.PrivateSubnet(
            self,
            "PrivateSubnet",
            availability_zone="us-east-1a",
            cidr_block=f"{ip}/28",
            vpc_id=self._vpc.vpc_id
        )

    @property
    def vpc(self):
        return self._vpc

    @property
    def private_subnet(self):
        return self._private_subnet
