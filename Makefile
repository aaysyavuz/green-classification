PROJECT_NAME := "green_classification"

format:
	@poetry run autoflake --in-place --remove-all-unused-imports --recursive --remove-unused-variables \
		--ignore-init-module-imports ${PROJECT_NAME}
	@poetry run isort .
	@poetry run black .

lint:
	@poetry run pylint ${PROJECT_NAME}
	@poetry run pylint tests

type-check:
	@poetry run mypy ${PROJECT_NAME}
	@poetry run mypy tests

compile: format lint type-check
	@clear
	@echo -e "\x1B[32mcompile success"
