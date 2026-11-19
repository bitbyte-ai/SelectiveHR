# Third-party components

The public API is project code. The archived research scripts are copied from the project with file provenance recorded in `PROVENANCE.json`. No Ultralytics package source or model weights are vendored.

The optional inference and training workflows depend on [Ultralytics](https://github.com/ultralytics/ultralytics), pinned to 8.4.138. Ultralytics offers AGPL-3.0 and commercial licensing; see its [official licensing page](https://www.ultralytics.com/license). Original project code in this release is licensed under AGPL-3.0-only. The complete license text is in `LICENSE`; it is copied verbatim from the dependency version's [license file](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.138/LICENSE). Existing third-party notices and dependency licenses continue to apply.

The CPU API uses NumPy. Analysis scripts additionally use pandas, SciPy, Pillow, PyYAML and scikit-learn; tests use pytest. PyTorch and related packages are needed for detector execution. These dependencies are installed separately and retain their own licenses. Version ranges in `pyproject.toml` describe the portable tools, not a reconstruction of every experiment environment.

Dataset rights and model-weight rights are separate from the code license. Installing a dependency or selecting a code license does not authorize redistribution of dataset material or third-party weights.
