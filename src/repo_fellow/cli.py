import click
import asyncio
from .__main__ import main

@click.group()
def cli():
    """RepoFellow CLI - Analyze GitHub repositories with AI"""
    pass

@cli.command()
@click.argument('repo_url', required=False)
@click.option('--query', '-q', default='analyze', help='Query to run on the repository')
def analyze(repo_url, query):
    """Analyze a GitHub repository"""
    result = asyncio.run(main(repo_url, query))
    click.echo(result.get("response", result.get("error", "No response generated")))

if __name__ == '__main__':
    cli() 