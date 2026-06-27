import 'package:flutter_test/flutter_test.dart';
import 'package:mindgraph_plusplus/features/settings/domain/settings_models.dart';

void main() {
  group('AppSettings', () {
    test('json roundtrip', () {
      const settings = AppSettings(
        security: SecuritySettings(biometricLogin: true),
        notifications: NotificationSettings(silentMode: true),
      );
      final restored = AppSettings.fromJson(settings.toJson());
      expect(restored.security.biometricLogin, isTrue);
      expect(restored.notifications.silentMode, isTrue);
    });
  });

  group('SettingsSearchItem', () {
    test('matches query', () {
      const item = SettingsSearchItem(
        id: 'privacy',
        title: 'Privacy & consent',
        section: 'Privacy',
        keywords: ['audio', 'consent'],
      );
      expect(item.matches('audio'), isTrue);
      expect(item.matches('xyz'), isFalse);
    });
  });
}
