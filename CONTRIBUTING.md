# Contributing to Research Literature Knowledge Graph System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect differing viewpoints and experiences

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in Issues
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots if applicable
   - Environment details (OS, Python version, Node version)

### Suggesting Enhancements

1. Check if the enhancement has been suggested
2. Create an issue describing:
   - The problem you're trying to solve
   - Your proposed solution
   - Alternative solutions considered
   - Any additional context

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following our coding standards
4. Write or update tests as needed
5. Update documentation
6. Commit with clear messages
7. Push to your fork
8. Open a Pull Request

## Development Setup

### Backend

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Frontend

```bash
cd frontend
npm install
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your credentials.

## Coding Standards

### Python

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for functions and classes
- Keep functions focused and small
- Use meaningful variable names

### JavaScript/React

- Use functional components with hooks
- Follow ESLint configuration
- Use meaningful component and variable names
- Keep components focused and reusable
- Write JSDoc comments for complex functions

### Git Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Start with a type: feat, fix, docs, style, refactor, test, chore
- Keep first line under 72 characters
- Reference issues and PRs when applicable

Examples:
```
feat: Add user authentication with Supabase
fix: Resolve citation serialization error
docs: Update README with installation steps
refactor: Simplify chat service logic
test: Add unit tests for document service
```

## Testing

### Run Backend Tests

```bash
pytest
pytest --cov=src tests/  # With coverage
```

### Run Frontend Tests

```bash
cd frontend
npm test
```

## Documentation

- Update README.md for user-facing changes
- Update API documentation for endpoint changes
- Add inline comments for complex logic
- Update CHANGELOG.md

## Questions?

Feel free to open an issue with the "question" label or reach out to the maintainers.

Thank you for contributing! 🎉
