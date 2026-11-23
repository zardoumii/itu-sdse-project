// CI/CD pipeline for the vial defect classifier.
//
// Functions here are called from GitHub Actions to test, train, build and
// deploy the model.

package main

import (
	"context"
)

// Python version used across the pipeline; matches .python-version.
const pythonImage = "python:3.13-slim"

type Pipeline struct{}

// PythonVersion prints the Python version available in the pipeline container.
func (m *Pipeline) PythonVersion(ctx context.Context) (string, error) {
	return dag.Container().
		From(pythonImage).
		WithExec([]string{"python", "--version"}).
		Stdout(ctx)
}
