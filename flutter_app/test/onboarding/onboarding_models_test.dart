import 'package:flutter_test/flutter_test.dart';
import 'package:mindgraph_plus_plus/features/onboarding/domain/onboarding_models.dart';

void main() {
  group('ProfileSetupData', () {
    test('json roundtrip', () {
      const data = ProfileSetupData(
        preferredName: 'Alex',
        notificationsEnabled: false,
        privacyLevel: PrivacyLevel.enhanced,
      );
      final restored = ProfileSetupData.fromJson(data.toJson());
      expect(restored.preferredName, 'Alex');
      expect(restored.privacyLevel, PrivacyLevel.enhanced);
    });
  });

  group('AuthSession', () {
    test('guest session is authenticated', () {
      const session = AuthSession(mode: AuthMode.guest, isGuest: true);
      expect(session.isAuthenticated, isTrue);
    });
  });
}
