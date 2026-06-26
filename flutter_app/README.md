# MindGraph++ Flutter Client

Production mobile application for MindGraph++ (SecureGraph-MultiDep).

## Prerequisites

- Flutter stable SDK
- JDK 21 for Android

## First-Time Setup

If platform folders (`android/`, `ios/`) are missing:

```bash
cd flutter_app
flutter create --org com.mindgraph --project-name mindgraph_plus_plus .
flutter pub get
```

## Run

```bash
flutter run
```

Configure API URL in `../configs/flutter.yaml` or via `--dart-define=API_BASE_URL=...`

## Architecture

Feature-first clean architecture with Riverpod, GoRouter, and Freezed.

See `lib/core/theme/` for the Healthcare Design Language (HDL) tokens.
