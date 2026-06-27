import 'package:flutter_test/flutter_test.dart';
import 'package:mindgraph_plusplus/features/home/domain/dashboard_models.dart';

void main() {
  group('DashboardData', () {
    test('json roundtrip', () {
      final data = DashboardData(
        mood: const MoodSnapshot(label: 'Calm', energy: 0.7),
        risk: const RiskOverview(level: RiskLevel.moderate, score: 0.5),
        streakDays: 5,
        hasCheckIns: true,
      );
      final restored = DashboardData.fromJson(data.toJson());
      expect(restored.streakDays, 5);
      expect(restored.mood?.label, 'Calm');
      expect(restored.risk?.level, RiskLevel.moderate);
    });

    test('isEmpty when no check-ins', () {
      expect(const DashboardData(hasCheckIns: false).isEmpty, isTrue);
      expect(const DashboardData(hasCheckIns: true).isEmpty, isFalse);
    });
  });

  group('MoodSnapshot', () {
    test('copyWith updates energy', () {
      const mood = MoodSnapshot(energy: 0.5);
      expect(mood.copyWith(energy: 0.8).energy, 0.8);
    });
  });

  group('CheckInSession', () {
    test('json roundtrip', () {
      final session = CheckInSession(
        id: 'a',
        date: DateTime(2025, 6, 1),
        moodLabel: 'Good',
        modalities: ['voice'],
      );
      final restored = CheckInSession.fromJson(session.toJson());
      expect(restored.moodLabel, 'Good');
      expect(restored.modalities, ['voice']);
    });
  });
}
