// CI/CD pipeline for the vial defect classifier.
//
// Functions here are called from GitHub Actions to test, train, build and
// deploy the model.

package main

import (
	"context"
	"dagger/pipeline/internal/dagger"
)

// Python version used across the pipeline; matches .python-version.
const pythonImage = "python:3.13-slim"

// CPU-only PyTorch wheels: the CUDA build from PyPI is several GB and CI has
// no GPU.
const torchCPUIndex = "https://download.pytorch.org/whl/cpu"

type Pipeline struct{}

// PythonVersion prints the Python version available in the pipeline container.
func (m *Pipeline) PythonVersion(ctx context.Context) (string, error) {
	return dag.Container().
		From(pythonImage).
		WithExec([]string{"python", "--version"}).
		Stdout(ctx)
}

// Test installs the Python dependencies and runs pytest.
//
// Passes when no tests are collected yet (pytest exit code 5), so the
// pipeline stays green until the first tests are merged.
func (m *Pipeline) Test(
	ctx context.Context,
	// Project root
	// +defaultPath="/"
	// +ignore=[".git", ".venv", "dagger", "notebooks", "data/images/raw", "artifacts", "mlflow-state", "models", "**/__pycache__", "**/.pytest_cache"]
	source *dagger.Directory,
) (string, error) {
	return pythonEnv(source).
		WithExec([]string{"sh", "-c", `
python -m pytest -v
status=$?
if [ "$status" -eq 5 ]; then
  echo "No tests collected yet"
  exit 0
fi
exit "$status"
`}).
		Stdout(ctx)
}

// pythonEnv returns a Python container with requirements.txt installed and
// the project source at /app.
//
// requirements.txt is copied and installed before the rest of the source, so
// the install layer is reused when only code changes; pip downloads are kept
// in a cache volume across runs.
func pythonEnv(source *dagger.Directory) *dagger.Container {
	return dag.Container().
		From(pythonImage).
		WithMountedCache("/root/.cache/pip", dag.CacheVolume("pip")).
		WithWorkdir("/app").
		WithFile("requirements.txt", source.File("requirements.txt")).
		WithExec([]string{
			"pip", "install",
			"--extra-index-url", torchCPUIndex,
			"-r", "requirements.txt",
		}).
		WithDirectory(".", source)
}
