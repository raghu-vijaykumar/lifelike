import click
from lifelike.generate_person import generate_person
from lifelike.generate_dreamboothset import generate_dreamboothset
from lifelike.generate_variations import generate_variations
from lifelike.generate_image import generate_image
from lifelike.version_cmd import version_cmd
from lifelike.enhance_images import enhance_images
from lifelike.animate_image import animate_image


@click.group()
def cli():
    """LifeLike AI Avatar CLI - Generate AI-powered videos with synthetic avatars and backgrounds."""
    pass


cli.add_command(generate_person)
cli.add_command(generate_dreamboothset)
cli.add_command(generate_variations)
cli.add_command(generate_image)
cli.add_command(version_cmd)
cli.add_command(enhance_images)
cli.add_command(animate_image)

if __name__ == "__main__":
    cli()
