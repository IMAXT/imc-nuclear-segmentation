import sys

import pkg_resources


def print_config():
    try:
        conf = pkg_resources.resource_stream(
            'imc_pipeline', '../conf/imc_pipeline.yaml'
        )
    except FileNotFoundError:
        conf = open(f'{sys.prefix}/conf/imc_pipeline.yaml', 'rb')
    print(conf.read().decode())


if __name__ == '__main__':  # pragma: nocover
    if 'config' in sys.argv[1:]:
        print_config()
