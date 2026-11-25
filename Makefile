# Default target runs the CPU rules suite, the Rust FFI tests, a compile
# check of the Python extension against the CPU backend, and the host
# bookkeeping tests. None of these need a GPU or the CUDA toolkit.

.PHONY: test cpu-test

test: cpu-test
	cargo test --manifest-path rust/Cargo.toml -p dojo-ref --no-default-features --features cpu
	cargo check --manifest-path rust/Cargo.toml -p dojo-ref-py --no-default-features --features cpu
	python3 -m unittest discover -s tests -v

cpu-test:
	$(MAKE) -C cuda/engine cpu-test
