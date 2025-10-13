# Contributing Guide


## Git Pull Request

Please follow these steps to open a pull request:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## CI/CD and GitHub Actions

For information about expanding GitHub Actions runner capacity and optimizing CI/CD workflows, please refer to our [GitHub Actions Runner Guide](./docs/source/github_actions_runners.md).

## Requirements

First of all, please install the required packages specific for developers by running:

```bash
pip install -r requirements.txt
pip install -r requirements_dev.txt
```


### Unit Tests

We use [PyTest](https://docs.pytest.org/en/latest/) to execute tests. You can install pytest by `pip install pytest`. Then, simply run:

```bash
pytest tests/
```

> [!NOTE]
> As some of the tests require initialization of the distributed backend, GPUs are needed to execute these tests.


### Code Style

We have some static checks when you commit your code change, please make sure you can pass all the tests and make sure the coding style follows our requirements. We use pre-commit hooks to make sure the code is aligned with the writing standard. To set up the code style checking, you need to follow the steps below.

* clang-format:

```bash
# Install clang-format with llvm for csrc code format
bash ./scripts/install_clang_format.sh
```

* pre-commit:

```bash
# Install pre-commit
pip install pre-commit

# Set up the hook, which may take some time
# Luckily, this step only needs to be performed once
pre-commit install

# Then each time before you run `git commit`,
# please run pre-commit to polish your code
pre-commit run -a

# For more usage about pre-commit,
# you may check: https://pre-commit.com/
```

> [!NOTE]
> Code format checking will be automatically executed when you commit your changes.
