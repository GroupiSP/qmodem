# AGENTS.md

This document contains informations for the agents about general coding guidelines employed in this project.

## Interactions with the users

- Keep a polite and professional tone.
- Keep your replies concise.
- Where possible, provide examples to illustrate your points.
- Where possible, provide schematics, diagrams, code snippets, tables or other visual aids to illustrate your points.

## Coding style

- Favour functions and data container classes over wrapping everything in a single class.
- Avoid god functions/classes. When reviewing code for quality, flag when an entity has too many responsibilities.
- Always use type hints for functions/classes.
- Favour composition over inheritance. Avoid deep inheritance hierarchies (one level at most).
- Try to abstract behaviour into interfaces and define the interfaces via `Protocols` (from `typing` module) rather than abstract base classes.
- Identify the possible introduction of design patterns, when reviewing code quality. Flag when a design pattern could be used to improve code quality.
- Write code for testing and think about which properties of a piece of code can be tested.

## Documentation

- Always add docstrings following the Google style guide (include `Args`, `Returns`, and `Raises` sections).

## Testing

- Add tests using pytest and arrange tests in modules that mirror the structure of the main codebase.
- Use reusable test fixtures as much as possible. Feel free to introduce a `conftest.py` file, if relevant.
- Privilege property-based testing over example-based testing, when possible. Favour mathematical properties, such as invariances, symmetries, monotonicity, etc.

## Running the code

Before presenting the changes for review,

- Always run `pre-commit run --all-files`.
- Smoke test the files in `scripts/`, except for those that are not maintained per repo instructions.
- Always run the tests with `pytest` and ensure that all tests pass.
- Use the mlflow MCP server to run the `scripts/` for a very few epochs and ensure that the losses are decreasing.
