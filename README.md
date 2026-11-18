# Vial defect classification – MLOps pipeline

Group project for ITU BDS MLOps (Data Science in Production), 2026.

A CNN classifies images of glass vials from a production line as clean or
defective. We are turning the original monolithic notebooks
(`notebooks/01_training.ipynb` and `notebooks/02_deployment.ipynb`) into a
modular Python project, with a Dagger workflow (Go) run by GitHub Actions that
trains the model and deploys it to DigitalOcean.

The original task description is in the course repository:
[lasselundstenjensen/itu-sdse-project](https://github.com/lasselundstenjensen/itu-sdse-project).

## Group members

- Erik Aymerich
- Benjamin
- Youssef

## Project structure

```text
.
├── data/                  # metadata csvs tracked with DVC; images are downloaded, not in git
│   ├── images/
│   └── test/
├── docker/                # Dockerfiles
├── docs/                  # architecture diagrams and dataset description
├── infra/digitalocean/    # DigitalOcean App Platform spec
├── notebooks/             # original notebooks, being split into src/
├── src/
│   ├── common/            # code shared by training and deployment (MLflow setup)
│   ├── training/          # training pipeline
│   ├── deployment/        # model selection, promotion and export
│   └── serving/           # inference service (/health, /invocations, /drift)
├── tests/
└── requirements.txt
```
