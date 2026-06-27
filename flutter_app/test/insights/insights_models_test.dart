import 'package:flutter_test/flutter_test.dart';
import 'package:mindgraph_plusplus/features/insights/domain/insights_models.dart';
import 'package:mindgraph_plusplus/features/home/domain/dashboard_models.dart';

void main() {
  group('InsightsData', () {
    test('defaults are safe empty state', () {
      const data = InsightsData();
      expect(data.totalSessions, 0);
      expect(data.forecasts, isEmpty);
    });
  });

  group('ForecastPanel', () {
    test('holds projection without certainty claim', () {
      const panel = ForecastPanel(
        horizon: ForecastHorizon.days7,
        projectedScore: 0.4,
        direction: TrendDirection.stable,
        confidence: 0.6,
      );
      expect(panel.confidence, lessThan(1.0));
    });
  });
}
