.PHONY: kafka-stop
kafka-stop:
	docker-compose down

.PHONY: kafka-start
kafka-start:
	docker-compose up -d

.PHONY: execute-test
execute-test:
	python3 setup.py pytest

.PHONY: test
test: kafka-start execute-test kafka-stop

.PHONY: build
build:
	python3 setup.py sdist bdist_wheel

.PHONY: install
install:
	python3 setup.py install

.PHONY: publish
publish:
	python3 -m twine upload dist/*