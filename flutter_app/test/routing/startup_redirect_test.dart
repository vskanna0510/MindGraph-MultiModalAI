import 'package:flutter_test/flutter_test.dart';
import 'package:mindgraph_plus_plus/core/providers/startup_state.dart';
import 'package:mindgraph_plus_plus/core/providers/router_provider.dart';

void main() {
  group('startupRedirectForPath', () {
    test('redirects to splash when incomplete', () {
      const startup = StartupState();
      expect(startupRedirectForPath('/home', startup), '/splash');
    });

    test('allows home when onboarding complete', () {
      const startup = StartupState(
        splashComplete: true,
        securityComplete: true,
        permissionsComplete: true,
        languageComplete: true,
        consentComplete: true,
        authenticated: true,
      );
      expect(startupRedirectForPath('/home', startup), isNull);
    });

    test('redirects away from splash when complete', () {
      const startup = StartupState(
        splashComplete: true,
        securityComplete: true,
        permissionsComplete: true,
        languageComplete: true,
        consentComplete: true,
        authenticated: true,
      );
      expect(startupRedirectForPath('/splash', startup), '/home');
    });

    test('offline route is always allowed', () {
      const startup = StartupState();
      expect(startupRedirectForPath('/offline', startup), isNull);
    });
  });

  group('StartupState', () {
    test('nextRoute walks onboarding sequence', () {
      const s = StartupState(splashComplete: true);
      expect(s.nextRoute, '/security');
    });

    test('toJson roundtrip', () {
      const s = StartupState(languageComplete: true, consentComplete: true);
      expect(StartupState.fromJson(s.toJson()).languageComplete, isTrue);
    });
  });
}
