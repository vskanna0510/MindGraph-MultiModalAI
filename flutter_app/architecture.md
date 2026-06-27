# MindGraph++ Flutter Architecture (MP5 Part 1)

## Healthcare Design Language (HDL v3)

### Colors (light)

| Token | Hex |
|-------|-----|
| Primary | `#5E9ED6` |
| Primary Dark | `#4A89C5` |
| Secondary | `#4CAF9C` |
| Accent | `#7C8CF8` |
| Support | `#A78BFA` |
| Success | `#34D399` |
| Warning | `#FBBF24` |
| Danger | `#EF4444` |
| Background | `#F8FAFC` |

### Colors (dark)

| Token | Hex |
|-------|-----|
| Background | `#09090B` |
| Surface | `#18181B` |
| Card | `#202024` |
| Primary | `#7CB5F0` |
| Secondary | `#5AD4B8` |
| Text | `#FAFAFA` |

### Typography scale

Display XL 48 · Display L 40 · Headline 32 · Title 24 · Subtitle 18 · Body 16 · Caption 14 · Label 12

Font: **Inter** with fallbacks SF Pro, Roboto, Noto Sans.

### Spacing (8dp grid)

4 · 8 · 12 · 16 · 20 · 24 · 32 · 40 · 48 · 64 · 80

### Radius

8 · 16 · 24 · 32 · pill 999

## Component catalog

| Widget | Purpose |
|--------|---------|
| `PrimaryButton` / `SecondaryButton` / `DangerButton` / `AppOutlinedButton` | Actions |
| `SurfaceCard` / `GradientCard` / `GlassCard` | Containers |
| `MoodCard` / `InsightCard` / `StatisticCard` | Home & insights |
| `RiskGauge` / `GraphCard` | Analytics |
| `RecommendationCard` / `SessionCard` / `TimelineCard` | History & guidance |
| `PrivacyBanner` / `ConsentTile` / `LanguageSelector` | Trust & settings |
| `FloatingRecordButton` / `AudioWaveCard` / `VideoPreviewCard` | Recording |
| `AdaptiveScaffold` | Responsive navigation |

## Riverpod providers

| Provider | Type | Role |
|----------|------|------|
| `startupProvider` | `Notifier<StartupState>` | Onboarding progress |
| `themeModeProvider` | `Notifier<ThemeMode>` | Light/dark/system |
| `localeProvider` | `Notifier<Locale>` | en/ta/hi |
| `connectivityProvider` | `StreamProvider` | Online/offline |
| `appRouterProvider` | `Provider<GoRouter>` | Navigation |

## Routes

| Path | Screen |
|------|--------|
| `/splash` | Branded splash |
| `/security` | Biometric opt-in |
| `/permissions` | Mic/camera/speech |
| `/language` | Locale picker |
| `/consent` | Privacy consent |
| `/auth` | Sign-in stub |
| `/home` … `/profile` | Main shell tabs |
| `/offline` `/error` | Utility |

## Accessibility checklist

- [x] 48dp minimum touch targets on buttons
- [x] Semantic labels on interactive widgets
- [x] Text scale clamped to 2.0
- [x] Reduced motion support
- [x] High contrast theme
- [x] Color + icon + label for risk/mood (not color alone)

## Deferred to MP5 Part 2

- Dio API client + graph/twin repositories
- Backend JWT authentication
- Camera/mic recording pipeline
- On-device ML inference
- Hive offline sync
