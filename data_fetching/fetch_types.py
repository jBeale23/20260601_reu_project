"""Shared type aliases for InterPro data-fetching scripts."""

# Raw JSON object returned by the InterPro API — keys are strings, values are
# JSON-native scalars or nested containers.
type ApiResponse = dict[str, str | int | float | bool | list | dict | None]

# Tracks how many architectures each protein accession appears in.
type ProteinCounts = dict[str, int]

# Per-architecture result assembled by fetch_architectures_dnaj.
type ArchResult = dict[str, str | int | bool | list[ApiResponse]]

# Single-entry protein fetch result from fetch_proteins_dnak.
type ProteinResult = dict[str, str | int | bool | list[ApiResponse] | None]

# Checkpoint payload saved when cursor pagination is interrupted.
type CheckpointData = dict[str, str | int | list[ApiResponse] | None]
