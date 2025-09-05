.PHONY: refresh build install build_dist json release test clean format

refresh: clean build install

# Format code
format:
	@echo "Formatting code..."
	@find magi_attention/csrc -type f \( -name "*.h" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" \) ! -path "magi_attention/csrc/cutlass/*" | xargs clang-format-21 -i
	@echo "Code formatting completed!"

# Check code format (does not modify files, only checks)
format-check:
	@echo "Checking code format..."
	@find magi_attention/csrc -type f \( -name "*.h" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" \) ! -path "magi_attention/csrc/cutlass/*" | xargs clang-format-21 --dry-run --Werror
	@echo "Code format check completed!"

build:
	git submodule update --init --recursive
	python -m build --wheel --no-isolation

install:
	git submodule update --init --recursive
	pip install --no-build-isolation .

uninstall:
	pip uninstall -y magi_attention

build_dist:
	make clean
	make build
	pip install dist/*.whl
	make test

release:
	python -m twine upload dist/*

test:
	pytest -q -s tests/

coverage:
	rm -f .coverage
	rm -f .coverage.*
	coverage run -m pytest tests/
	coverage combine
	coverage report
	coverage html

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf build
	rm -rf dist
	rm -rf magi_attention/_version.py
	rm -rf magi_attention.egg-info
	make uninstall
