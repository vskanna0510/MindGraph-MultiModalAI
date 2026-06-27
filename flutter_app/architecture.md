# MindGraph++ Design System, Onboarding & Dashboard

Enterprise UI foundation for MindGraph++ (MP5 Parts 2–4).

## Principles

1. **Never hardcode** colors, spacing, or typography in widgets
2. Use `context.colors` for semantic colors
3. Use `AppSpacingTokens`, `AppRadiusTokens`, etc. for layout
4. Import from `core/design_system/design_system.dart`
5. **All user-facing strings** in ARB files (`l10n/`)

## Token architecture

See [`tokens/README.md`](lib/core/design_system/tokens/README.md).

## Onboarding flow (MP5 Part 3)

```
/splash → /security → /language → /welcome → /overview → /privacy-intro
→ /consent → /permissions → /auth → /profile-setup → /tutorial → /home
```

State persisted in secure storage. Reset from Profile clears all onboarding keys.

## Home dashboard (MP5 Part 4)

Single-scroll personalized home at `/home`:

```
App Bar (greeting, date, privacy, notifications, avatar)
→ Greeting + streak
→ Today's Mood Card
→ Risk Overview + AnimatedRiskGauge
→ Trend Summary (7d / 30d / 90d)
→ Quick Actions grid
→ Recent Check-ins (horizontal)
→ AI Insight
→ Recommendation
→ Daily Motivation
```

### Providers

| Provider | Purpose |
|----------|---------|
| `dashboardProvider` | Aggregated dashboard state |
| `riskProvider` | Current risk overview |
| `moodProvider` | Today's mood snapshot |
| `recommendationProvider` | Daily recommendation |
| `recentSessionsProvider` | Recent check-in sessions |
| `offlineCacheProvider` | Cached dashboard for offline |

Data loads from secure storage cache; Neo4j/backend integration is stubbed. Offline banner when connectivity is unavailable.

## Multimodal recording (MP5 Part 5)

Flagship check-in at `/record`:

```
Mode Selection → Privacy Reminder → Preparation → Live Recording → AI Processing → Result
```

### Modes

Voice · Video · Image · Text · Combined (video + audio + journal)

### Providers

| Provider | Purpose |
|----------|---------|
| `recordingProvider` | Flow state machine + session lifecycle |
| `permissionProvider` | Camera / mic / storage status |
| `cameraProvider` / `microphoneProvider` | Permission-derived booleans |
| `journalProvider` | Auto-saved journal draft |
| `processingProvider` | Upload → features → inference → graph progress |

Temporary session metadata stored encrypted; media cleanup after successful processing unless user retains.

## AI processing & results (MP5 Part 6)

After recording completes:

```
8-step pipeline → SessionAnalysisResult → /session/:id/result
```

### Processing screen

Animated particles, 8-step checklist, gradient progress ring, ETA, privacy bar (device/edge/cloud, encrypted, consent).

### Results screen

Explainable, non-diagnostic presentation:

- Risk gauge with supportive narrative + confidence interval
- Confidence card (model/graph version, processing mode)
- Modality + feature contributions (SHAP-style summary)
- Temporal context + graph preview
- Three AI insights, recommendations, conditional support resources
- Export/share sheet (PDF, JSON, CSV, encrypted — consent-gated)

### Providers

| Provider | Purpose |
|----------|---------|
| `pipelineProvider` | 8-step AI processing progress |
| `resultProvider` | Session analysis payload |
| `insightProvider` | AI insight list |
| `recommendationProvider` | Daily recommendations |
| `exportProvider` | Consent-gated export |

## Component catalog
| Category | Key widgets |
|----------|-------------|
| Layout | `AppScaffold`, `AppHeader`, `PermissionCard`, `PrivacyBanner`, `AppIllustration` |
| Surfaces | `GlassSurface`, `PrimaryCard`, `SecondaryCard` |
| Metrics | `MetricCard`, `RiskCard`, `TrendCard`, `AnalyticsCard` |
| Buttons | `PrimaryButton`, `FilledTonalButton`, `AsyncButton`, `GlassButton` |
| Inputs | `AppTextField`, `AppSearchField`, `JournalField`, `OtpField` |
| Modals | `AppConfirmDialog`, `PrivacyDialog`, `RecordingDialog` |
| Loading | `SkeletonCard`, `ShimmerLoader`, `ProgressRing`, `WaveAnimation` |
| Feedback | `StatusBadge`, `PrivacyBadge`, `InlineAlert`, `OnboardingEmptyState` |
| Charts | `RiskGauge`, `CircularGauge`, `TrendChart`, `KnowledgeGraphPreview` |
| Motion | `PulseAnimation`, `AppMotionPresets`, shared-axis page transitions |
| Haptics | `AppHaptics` |

## Accessibility

- Semantic labels on all onboarding controls
- VoiceOver / TalkBack compatible progress indicator
- Reduced motion respected in page transitions
- 48dp touch targets on primary actions

## Tests

```bash
flutter test test/design_system/
flutter test test/history/
flutter test test/insights/
flutter test test/graph/
flutter test test/settings/
flutter test test/home/
flutter test test/onboarding/
flutter test test/routing/
flutter test --coverage
```

Target: ≥90% coverage on design system, onboarding, home, routing, and Part 7 features (history, insights, graph, settings).

## History, insights, graph, profile & settings (MP5 Part 7)

| Feature | Route | Key providers |
|---------|-------|---------------|
| History | `/history` | `historyProvider`, `filteredHistoryProvider` |
| Session detail | `/session/:id` | `historyProvider` |
| Insights | `/insights` | `insightsProvider` |
| Knowledge graph | `/graph` | `graphProvider`, `graphTimelineProvider` |
| Profile | `/profile` | `profileSetupProvider`, `consentProvider` |
| Settings | `/settings` | `settingsProvider`, `notificationProvider` |

History: search, filters, timeline/calendar, session cards, analytics summary.
Insights: trend charts, behavioural/emotion/language analytics, forecast (non-certain).
Graph: interactive pan/zoom canvas, timeline replay, node details.
Profile/Settings: privacy, security, notifications, export & offline centers.

## Localization

Supported: English (`en`), Tamil (`ta`), Hindi (`hi`). ARB files in `l10n/`.
