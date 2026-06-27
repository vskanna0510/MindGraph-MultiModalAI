import 'package:flutter_test/flutter_test.dart';
import 'package:mindgraph_plusplus/features/home/presentation/widgets/animated_risk_gauge.dart';
import 'package:mindgraph_plusplus/features/home/domain/dashboard_models.dart';

void main() {
  group('riskLevelLabel', () {
    test('maps levels to supportive labels', () {
      expect(
        riskLevelLabel(RiskLevel.elevated, 'Low', 'Moderate', 'Elevated'),
        'Elevated',
      );
    });
  });

  group('trendLabel', () {
    test('maps directions', () {
      expect(
        trendLabel(TrendDirection.improving, 'Up', 'Down', 'Flat'),
        'Up',
      );
    });
  });

  group('periodLabel', () {
    test('maps periods', () {
      expect(
        periodLabel(TrendPeriod.days7, '7d', '30d', '90d'),
        '7d',
      );
    });
  });
}
