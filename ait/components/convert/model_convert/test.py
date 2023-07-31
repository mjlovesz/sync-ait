import argparse
import yaml


if __name__ == "__main__":
    conf_file = "aoe/aoe_args_map.yml"
    with open(conf_file, 'r', encoding='utf-8') as f:
        args_conf = yaml.load(f, yaml.Loader)
    print(type(args_conf))

    parser = argparse.ArgumentParser()
    args = args_conf.get('aoe_args')
    for arg in args:
        # print("{} : {}".format(arg.get('name'), arg.get('desc')))
        abbr_name = arg.get('abbr_name') if arg.get('abbr_name') else ""
        is_required = arg.get('is_required') if arg.get('is_required') else "False"

        if abbr_name:
            parser.add_argument(abbr_name, arg.get('name'), required=is_required, help=arg.get('desc'))
        else:
            parser.add_argument(arg.get('name'), required=is_required, help=arg.get('desc'))
    # args = parser.parse_args()
    print(parser.print_help())
