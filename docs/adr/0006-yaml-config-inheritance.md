# ADR-0006: YAML Config Inheritance System

All model configs extend `configs/base.yaml` via an `extends:` key. Only differences are overridden. The config loader supports deep merge, path resolution, validation (type + range), and CLI overrides (`--section.key value`).

Without inheritance, `nano.yaml` and `lite.yaml` would each need to copy every shared default, creating drift as defaults evolve.

Consequence: Adding a new model variant means writing a ~35-line YAML that declares only what differs from the base.
