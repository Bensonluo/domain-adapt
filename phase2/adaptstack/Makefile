.PHONY: test compile doctor dry-run

test:
	python -m unittest discover -s tests -v

compile:
	python -m compileall -q src scripts demo tests

doctor:
	python scripts/train.py --doctor

dry-run:
	python scripts/train.py --config configs/medical.yaml --dry-run
