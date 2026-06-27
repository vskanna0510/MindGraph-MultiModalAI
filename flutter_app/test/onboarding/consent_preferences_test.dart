import 'package:flutter_test/flutter_test.dart';
import 'package:mindgraph_plusplus/features/onboarding/domain/onboarding_models.dart';

void main() {
  group('ProfileSetupData', () {
    test('json roundtrip preserves fields', () {
      const data = ProfileSetupData(
        preferredName: 'Alex',
        notificationsEnabled: false,
        dailyReminderHour: 18,
        dailyReminderMinute: 30,
        privacyLevel: PrivacyLevel.enhanced,
      );
      final restored = ProfileSetupData.fromJson(data.toJson());
      expect(restored.preferredName, 'Alex');
      expect(restored.notificationsEnabled, isFalse);
      expect(restored.dailyReminderHour, 18);
      expect(restored.privacyLevel, PrivacyLevel.enhanced);
    });
  });

  group('AuthSession', () {
    test('guest session is authenticated', () {
      const session = AuthSession(mode: AuthMode.guest, isGuest: true);
      expect(session.isAuthenticated, isTrue);
    });

    test('none session is not authenticated', () {
      expect(const AuthSession().isAuthenticated, isFalse);
    });
  });

  group('DeviceSecurityStatus', () {
    test('critical failure when secure storage unavailable', () {
      expect(const DeviceSecurityStatus(secureStorageAvailable: false).hasCriticalFailure, isTrue);
      expect(const DeviceSecurityStatus().hasCriticalFailure, isFalse);
    });
  });
}
