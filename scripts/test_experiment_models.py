from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.experiments.models import (
    ExperimentArtifact,
    ExperimentInfo,
    ExperimentModelError,
    ExperimentResult,
    ExperimentStatus,
    ExperimentType,
)


def print_separator() -> None:
    print("=" * 70)


def test_experiment_creation() -> None:
    print_separator()
    print("TEST DE CREATION D'UNE EXPERIENCE")
    print_separator()

    experiment = ExperimentInfo.create(
        name="Fixed Size 256 32",
        experiment_type=ExperimentType.CHUNKING,
        dataset_name="azure",
        description=(
            "Benchmark du chunking fixe avec une taille "
            "de 256 tokens et un overlap de 32 tokens."
        ),
        tags=[
            "Fixed Size",
            "Chunking",
            "fixed size",
        ],
        metadata={
            "corpus_version": "1.0",
        },
    )

    print("Experiment ID :", experiment.experiment_id)
    print("Nom :", experiment.name)
    print("Type :", experiment.experiment_type.value)
    print("Dataset :", experiment.dataset_name)
    print("Statut :", experiment.status.value)
    print("Tags :", experiment.tags)

    assert experiment.experiment_id.startswith(
        "chunking_fixed_size_256_32_"
    )
    assert experiment.name == "Fixed Size 256 32"
    assert experiment.experiment_type == (
        ExperimentType.CHUNKING
    )
    assert experiment.status == ExperimentStatus.CREATED
    assert experiment.dataset_name == "azure"

    assert experiment.tags == [
        "fixed size",
        "chunking",
    ]

    assert experiment.duration_seconds is None
    assert experiment.is_finished is False

    print("Création de l'expérience : OK")


def test_experiment_lifecycle() -> None:
    print()
    print_separator()
    print("TEST DU CYCLE DE VIE")
    print_separator()

    experiment = ExperimentInfo.create(
        name="Fixed 512 64",
        experiment_type="chunking",
        dataset_name="azure",
    )

    start_time = datetime(
        2026,
        7,
        22,
        10,
        0,
        0,
        tzinfo=timezone.utc,
    )

    end_time = start_time + timedelta(
        seconds=12.5
    )

    experiment.mark_running(
        started_at=start_time
    )

    assert experiment.status == ExperimentStatus.RUNNING
    assert experiment.started_at == start_time

    experiment.mark_completed(
        completed_at=end_time
    )

    assert experiment.status == (
        ExperimentStatus.COMPLETED
    )
    assert experiment.completed_at == end_time
    assert experiment.duration_seconds == 12.5
    assert experiment.is_finished is True

    print("Statut final :", experiment.status.value)
    print(
        "Durée :",
        experiment.duration_seconds,
        "secondes",
    )
    print("Cycle de vie : OK")


def test_failed_experiment() -> None:
    print()
    print_separator()
    print("TEST D'UNE EXPERIENCE EN ECHEC")
    print_separator()

    experiment = ExperimentInfo.create(
        name="Failed experiment",
        experiment_type="retrieval",
        dataset_name="test-dataset",
    )

    experiment.mark_running()
    experiment.mark_failed(
        "Impossible de charger l'index."
    )

    assert experiment.status == ExperimentStatus.FAILED
    assert experiment.error_message == (
        "Impossible de charger l'index."
    )
    assert experiment.is_finished is True
    assert experiment.duration_seconds is not None

    print("Statut :", experiment.status.value)
    print("Erreur :", experiment.error_message)
    print("Gestion de l'échec : OK")


def test_artifact() -> None:
    print()
    print_separator()
    print("TEST D'UN ARTIFACT")
    print_separator()

    artifact = ExperimentArtifact(
        name="Chunks JSONL",
        path=(
            "data/experiments/chunking/"
            "fixed_256_32/chunks.jsonl"
        ),
        artifact_type="jsonl",
        description="Chunks produits par l'expérience.",
    )

    assert artifact.filename == "chunks.jsonl"
    assert artifact.artifact_type == "jsonl"

    print("Nom :", artifact.name)
    print("Fichier :", artifact.filename)
    print("Type :", artifact.artifact_type)
    print("Artifact : OK")


def test_experiment_result() -> None:
    print()
    print_separator()
    print("TEST DU RESULTAT D'EXPERIENCE")
    print_separator()

    experiment = ExperimentInfo.create(
        name="Fixed 256 32",
        experiment_type=ExperimentType.CHUNKING,
        dataset_name="azure",
    )

    experiment.mark_running()
    experiment.mark_completed()

    result = ExperimentResult(
        experiment=experiment,
        configuration={
            "strategy": "fixed_size",
            "chunk_size": 256,
            "chunk_overlap": 32,
            "tokenizer_name": (
                "BAAI/bge-small-en-v1.5"
            ),
        },
    )

    result.add_metric(
        "document_count",
        203,
    )
    result.add_metric(
        "chunk_count",
        4286,
    )
    result.add_metric(
        "average_tokens",
        247.3,
    )

    result.add_artifact(
        ExperimentArtifact(
            name="Chunks",
            path="chunks.jsonl",
            artifact_type="jsonl",
        )
    )

    result.add_note(
        "Le dernier chunk de chaque document peut être plus court."
    )

    serialized = result.to_dict()

    print(
        json.dumps(
            serialized,
            ensure_ascii=False,
            indent=2,
        )
    )

    assert serialized["metrics"]["document_count"] == 203
    assert serialized["metrics"]["chunk_count"] == 4286

    assert serialized["configuration"]["chunk_size"] == 256

    assert len(serialized["artifacts"]) == 1
    assert serialized["artifacts"][0]["path"] == (
        "chunks.jsonl"
    )

    assert len(serialized["notes"]) == 1

    print("Résultat d'expérience : OK")


def test_json_serialization() -> None:
    print()
    print_separator()
    print("TEST DE SERIALISATION JSON")
    print_separator()

    experiment = ExperimentInfo.create(
        name="Semantic 512",
        experiment_type="chunking",
        dataset_name="azure",
    )

    serialized = experiment.to_dict()

    json_content = json.dumps(
        serialized,
        ensure_ascii=False,
        indent=2,
    )

    assert isinstance(json_content, str)
    assert '"experiment_type": "chunking"' in json_content
    assert '"status": "created"' in json_content

    print(json_content)
    print("Sérialisation JSON : OK")


def test_invalid_experiment() -> None:
    print()
    print_separator()
    print("TEST DE VALIDATION")
    print_separator()

    try:
        ExperimentInfo(
            experiment_id="",
            name="Test",
            experiment_type=ExperimentType.CHUNKING,
            dataset_name="azure",
        )
    except ExperimentModelError as exc:
        print(
            "Identifiant vide détecté :",
            exc,
        )
    else:
        raise AssertionError(
            "Une erreur était attendue pour "
            "experiment_id vide."
        )

    try:
        ExperimentInfo.create(
            name="",
            experiment_type="chunking",
            dataset_name="azure",
        )
    except ExperimentModelError as exc:
        print(
            "Nom vide détecté :",
            exc,
        )
    else:
        raise AssertionError(
            "Une erreur était attendue pour un nom vide."
        )

    try:
        ExperimentInfo.create(
            name="Invalid",
            experiment_type="invalid_type",
            dataset_name="azure",
        )
    except ExperimentModelError as exc:
        print(
            "Type invalide détecté :",
            exc,
        )
    else:
        raise AssertionError(
            "Une erreur était attendue pour "
            "un type invalide."
        )

    print("Validations : OK")


def test_invalid_transition() -> None:
    print()
    print_separator()
    print("TEST D'UNE TRANSITION INVALIDE")
    print_separator()

    experiment = ExperimentInfo.create(
        name="Transition test",
        experiment_type="embedding",
        dataset_name="azure",
    )

    try:
        experiment.mark_completed()
    except ExperimentModelError as exc:
        print(
            "Transition invalide détectée :",
            exc,
        )
    else:
        raise AssertionError(
            "Une expérience CREATED ne doit pas "
            "passer directement à COMPLETED."
        )

    experiment.mark_running()
    experiment.mark_completed()

    try:
        experiment.mark_running()
    except ExperimentModelError as exc:
        print(
            "Redémarrage interdit détecté :",
            exc,
        )
    else:
        raise AssertionError(
            "Une expérience terminée ne doit pas "
            "être redémarrée."
        )

    print("Transitions contrôlées : OK")


def main() -> None:
    test_experiment_creation()
    test_experiment_lifecycle()
    test_failed_experiment()
    test_artifact()
    test_experiment_result()
    test_json_serialization()
    test_invalid_experiment()
    test_invalid_transition()

    print()
    print_separator()
    print("MODELES D'EXPERIMENTATION FONCTIONNELS.")
    print_separator()


if __name__ == "__main__":
    main()