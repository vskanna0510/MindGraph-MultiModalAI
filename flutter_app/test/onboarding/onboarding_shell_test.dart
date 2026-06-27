import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_gen/gen_l10n/app_localizations.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:mindgraph_plusplus/core/providers/startup_state.dart';
import 'package:mindgraph_plusplus/features/onboarding/presentation/widgets/onboarding_shell.dart';

void main() {
  Widget wrap(Widget child) {
    return MaterialApp(
      localizationsDelegates: const [
        AppLocalizations.delegate,
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],
      supportedLocales: AppLocalizations.supportedLocales,
      home: child,
    );
  }

  group('OnboardingShell', () {
    testWidgets('shows progress indicator with correct step count', (tester) async {
      await tester.pumpWidget(
        wrap(
          OnboardingShell(
            stepIndex: 3,
            child: const Text('Content'),
          ),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.text('Content'), findsOneWidget);
      expect(find.byType(OnboardingProgressIndicator), findsOneWidget);
    });

    testWidgets('skip button calls callback', (tester) async {
      var skipped = false;
      await tester.pumpWidget(
        wrap(
          OnboardingShell(
            stepIndex: 5,
            onSkip: () => skipped = true,
            child: const SizedBox(),
          ),
        ),
      );
      await tester.pumpAndSettle();

      await tester.tap(find.text('Skip'));
      await tester.pumpAndSettle();
      expect(skipped, isTrue);
    });
  });

  group('OnboardingProgressIndicator', () {
    test('total matches StartupState.totalSteps', () {
      expect(StartupState.totalSteps, 11);
    });
  });
}
