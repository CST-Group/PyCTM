.PHONY: kafka-stop
kafka-stop:
	docker-compose down

.PHONY: kafka-start
kafka-start:
	docker-compose up -d

.PHONY: execute-test
execute-test:
	python setup.py pytest

.PHONY: test
test: kafka-start execute-test kafka-stop

.PHONY: build
build:
	python setup.py bdist_wheel

.PHONY: publish
publish:
	twine upload dist/*