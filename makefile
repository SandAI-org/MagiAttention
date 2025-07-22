.PHONY: refresh build install build_dist json release test clean format

refresh: clean build install

# Format code
format:
	@echo "Formatting code..."
	@find magi_attention/csrc/flexible_flash_attention -name "*.h" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" | xargs clang-format -i
	@find magi_attention/csrc/common -name "*.h" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" | xargs clang-format -i
	@echo "Code formatting completed!"

# Check code format (does not modify files, only checks)
format-check:
	@echo "Checking code format..."
	@find magi_attention/csrc/flexible_flash_attention -name "*.h" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" | xargs clang-format --dry-run --Werror
	@find magi_attention/csrc/common -name "*.h" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" | xargs clang-format --dry-run --Werror
	@echo "Code format check completed!"

build:
	git submodule update --init --recursive
	python -m build --wheel --no-isolation

install:
	git submodule update --init --recursive
	pip install --no-build-isolation .

build_dist:
	make clean
	git submodule update --init --recursive
	python -m build --wheel --no-isolation
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
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf magi_attention/__pycache__
	rm -rf magi_attention/_version.py
	rm -rf build
	rm -rf dist
	rm -rf magi_attention.egg-info
	rm -rf src/magi_attention.egg-info
	pip uninstall -y magi_attention
