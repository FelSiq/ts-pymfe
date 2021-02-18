PACKAGE := pymfe

code-check: ## Execute the code check with flake8, pylint, mypy.
	flake8 $(PACKAGE)
	pylint $(PACKAGE) -j 0 -d 'C0103, R0913, R0902, R0914, C0302, R0904, R0801, E1101, C0330, E1136'
	mypy $(PACKAGE) --ignore-missing-imports

make-package:
	python -m pip install -U setuptools wheel
	python setup.py sdist bdist_wheel
	python -m twine upload dist/*

format: ## format all the package using black
	@black --line-length 79 pymfe/
