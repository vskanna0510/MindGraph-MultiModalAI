# MindGraph++ Flutter Application

Enterprise UI/UX foundation for the MindGraph++ mobile and desktop client.

## Setup

```bash
cd flutter_app
flutter create . --platforms=android,ios,web,windows,macos,linux  # first time only
flutter pub get
flutter gen-l10n
flutter run
```

API base URL: `--dart-define=API_BASE_URL=http://localhost:8000`

## Architecture (MP5 Parts 1–7)

```
lib/
├── core/
│   ├── design_system/   # HDL tokens + 30+ reusable widgets
│   ├── theme/           # ThemeData assembly
│   ├── animations/      # Page transitions, stagger, reduced motion
│   ├── providers/       # Riverpod (startup, theme, locale, router)
│   ├── routing/         # GoRouter exports
│   ├── services/        # Config + secure storage
│   └── constants/       # Env, breakpoints
├── features/
│   ├── record/          # Multimodal check-in (MP5 Part 5)
│   ├── results/         # AI processing + explainable results (MP5 Part 6)
│   ├── graph/           # Interactive knowledge graph (MP5 Part 7)
│   ├── settings/        # Settings, export, offline (MP5 Part 7)
│   ├── history/         # Session history (MP5 Part 7)
│   ├── home/            # Personalized dashboard (MP5 Part 4)
│   ├── onboarding/      # 11-step first-launch flow (MP5 Part 3)
│   ├── consent/         # Granular consent management
│   ├── authentication/  # Email, social, guest auth (stub)
│   └── …                # home, record, history, insights, profile
└── shared/widgets/      # Legacy re-exports
```

## Startup flow (MP5 Part 3)

```
splash → security → language → welcome → overview → privacy-intro
→ consent → permissions → auth → profile-setup → tutorial → home
```

State persisted via `flutter_secure_storage` (JWT, consent, profile, progress). Reset from Profile → "Reset onboarding" clears all onboarding keys.

See [architecture.md](architecture.md) for the full onboarding spec.

## Design system

Import everything from:

```dart
import 'package:mindgraph_plus_plus/core/design_system/design_system.dart';
```

See [architecture.md](architecture.md) for token reference and component catalog.

## Tests

```bash
flutter test
flutter test test/history/
flutter test test/insights/
flutter test test/graph/
flutter test test/settings/
flutter test test/results/
flutter test test/record/
flutter test --coverage
```

Target: ≥90% coverage on `core/design_system/`, `core/theme/`, `test/onboarding/`, and `test/results/`.

## Localization

Supported: English (`en`), Tamil (`ta`), Hindi (`hi`). ARB files in `l10n/`.
