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

.PHONY: publish_token
publish_token:
	python3.10 -m twine upload --username __token__ --password pypi-AgEIcHlwaS5vcmcCJGZjNmUwYjVhLTYyMjMtNDU4Mi04MWM4LTUxMWMxZGU4Yzk3ZQACKlszLCI1OWM5N2E3ZS0xNjA4LTRjNzUtYTEyZi1iZjgyOWNiNmE5OTUiXQAABiCcQkI3YN0CnBfEuiVXSYblKs7DvFlOMvVrDBHXtMVY-Q dist/*
