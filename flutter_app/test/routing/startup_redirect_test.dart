import 'package:flutter_test/flutter_test.dart';
import 'package:mindgraph_plus_plus/core/providers/startup_state.dart';
import 'package:mindgraph_plus_plus/core/providers/router_provider.dart';
import 'package:mindgraph_plus_plus/features/onboarding/domain/onboarding_models.dart';

void main() {
  group('startupRedirectForPath', () {
    test('redirects to splash when incomplete', () {
      const startup = StartupState();
      expect(startupRedirectForPath('/home', startup), '/splash');
    });

    test('follows full onboarding sequence', () {
      expect(const StartupState(splashComplete: true).nextRoute, '/security');
      expect(
        const StartupState(splashComplete: true, securityComplete: true).nextRoute,
        '/language',
      );
      expect(
        const StartupState(
          splashComplete: true,
          securityComplete: true,
          languageComplete: true,
        ).nextRoute,
        '/welcome',
      );
      expect(
        const StartupState(
          splashComplete: true,
          securityComplete: true,
          languageComplete: true,
          welcomeComplete: true,
          overviewComplete: true,
          privacyIntroComplete: true,
          consentComplete: true,
          permissionsComplete: true,
          authenticated: true,
          profileSetupComplete: true,
        ).nextRoute,
        '/tutorial',
      );
    });

    test('allows home when onboarding complete', () {
      const startup = StartupState(
        splashComplete: true,
        securityComplete: true,
        languageComplete: true,
        welcomeComplete: true,
        overviewComplete: true,
        privacyIntroComplete: true,
        consentComplete: true,
        permissionsComplete: true,
        authenticated: true,
        profileSetupComplete: true,
        tutorialComplete: true,
      );
      expect(startupRedirectForPath('/home', startup), isNull);
      expect(startup.isOnboardingComplete, isTrue);
    });

    test('offline route is always allowed', () {
      const startup = StartupState();
      expect(startupRedirectForPath('/offline', startup), isNull);
    });

    test('session result route is always allowed', () {
      const startup = StartupState();
      expect(startupRedirectForPath('/session/abc123/result', startup), isNull);
    });
  });

  group('ConsentPreferences', () {
    test('requires at least one modality', () {
      expect(const ConsentPreferences().hasRequiredConsent, isFalse);
      expect(
        const ConsentPreferences(textJournal: true).hasRequiredConsent,
        isTrue,
      );
    });

    test('json roundtrip', () {
      const prefs = ConsentPreferences(audioAnalysis: true, edgeProcessing: true);
      expect(ConsentPreferences.fromJson(prefs.toJson()).audioAnalysis, isTrue);
    });
  });

  group('StartupState', () {
    test('toJson roundtrip includes new fields', () {
      const s = StartupState(welcomeComplete: true, tutorialComplete: false);
      final restored = StartupState.fromJson(s.toJson());
      expect(restored.welcomeComplete, isTrue);
      expect(restored.tutorialComplete, isFalse);
    });

    test('totalSteps is 11', () {
      expect(StartupState.totalSteps, 11);
    });
  });
}
