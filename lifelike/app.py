import click


@click.group()
def cli():
    """LifeLike AI Avatar CLI - Generate AI-powered videos with synthetic avatars and backgrounds."""
    pass


@cli.command()
def version():
    """Show the version of LifeLike CLI."""
    click.echo("LifeLike CLI version 0.1.0")


if __name__ == "__main__":
    cli()
