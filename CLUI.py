import click
from src.Karyotype_retriever import Environement

@click.group()
def main_command_group():
    pass


@click.command()
@click.argument('path_to_file_location')
@click.argument('file_name')
def run_pipeline(path_to_file_location, file_name):
    """
    Executes the pipeline once provided path to a specific location and filename

    :param path_to_file_location:
    :param file_name:
    :return:
    """
    env = Environement(path_to_file_location, file_name)
    return env.compute_all_karyotypes()

main_command_group.add_command(run_pipeline)


if __name__ == '__main__':
    main_command_group()