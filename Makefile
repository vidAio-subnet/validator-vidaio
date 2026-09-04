# VidAIO — install/test/build helpers for this release mirror. `make help` lists everything.

PYTEST := $(if $(wildcard .venv/bin/pytest),.venv/bin/pytest,pytest)

.DEFAULT_GOAL := help
.PHONY: help install test docker-build

help: ## Show this help
	@echo "VidAIO release-mirror targets:"
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

install: ## Install this package with the real Bittensor chain adapter
	pip install -e '.[chain]'

test: ## The in-process unit/integration suite (chainless; needs the dev extra)
	$(PYTEST) -q

docker-build: ## Build the release image from this tree
	docker build -t vidaio-next:latest .
