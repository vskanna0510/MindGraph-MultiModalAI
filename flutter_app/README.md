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

## Architecture (MP5 Part 1)

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
├── features/            # Feature-owned screens (UI only in Part 1)
└── shared/widgets/      # Legacy re-exports
```

## Startup flow

`splash` → `security` → `permissions` → `language` → `consent` → `auth` → main shell (`/home`)

State persisted via `flutter_secure_storage`. Reset from Profile → "Reset onboarding".

## Design system

Import everything from:

```dart
import 'package:mindgraph_plus_plus/core/design_system/design_system.dart';
```

See [architecture.md](architecture.md) for token reference and component catalog.

## Tests

```bash
flutter test
flutter test --coverage
```

Target: ≥90% coverage on `core/design_system/` and `core/theme/`.

## Localization

Supported: English (`en`), Tamil (`ta`), Hindi (`hi`). ARB files in `l10n/`.
