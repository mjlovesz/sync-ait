from dataclasses import dataclass


ATC_ONLY = 0
AOE_ONLY = 1
ATC_AOE = 2


@dataclass
class ConvertArgs:
    arg_name: str = '',
    arg_des: str = '',
    arg_type: int = 0
