import 'package:flutter_test/flutter_test.dart';
import 'package:mindgraph_plusplus/features/history/domain/history_models.dart';
import 'package:mindgraph_plusplus/features/home/domain/dashboard_models.dart';

void main() {
  group('HistoryFilterState', () {
    test('copyWith updates query', () {
      const f = HistoryFilterState(query: 'calm');
      expect(f.copyWith(risk: HistoryRiskFilter.low).risk, HistoryRiskFilter.low);
    });
  });

  group('HistorySession', () {
    test('risk pattern labels are non-diagnostic', () {
      final session = HistorySession(
        id: '1',
        date: DateTime(2026, 1, 1),
        moodLabel: 'Calm',
        riskLevel: RiskLevel.elevated,
      );
      expect(session.riskPatternLabel, 'Elevated pattern');
      expect(session.riskPatternLabel, isNot(contains('depressed')));
    });
  });
}
