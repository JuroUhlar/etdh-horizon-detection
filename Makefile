ATTEMPT ?= attempts/attempt-2-rotation-invariant

# Build the base Docker image
build:
	docker compose build

# Run the evaluator against an attempt.
# Usage: make eval ATTEMPT=attempts/attempt-3-top-n-ransac
eval: build
	docker compose run --rm horizon sh -c \
		"pip install -q --root-user-action=ignore -r docker/requirements.txt && \
		 python -u tools/evaluate.py $(ATTEMPT)"

.PHONY: build eval
