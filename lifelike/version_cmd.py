import click


def version():
    """Show the version of LifeLike CLI."""
    click.echo("LifeLike CLI version 0.1.0")


@click.command()
def version_cmd():
    version()
