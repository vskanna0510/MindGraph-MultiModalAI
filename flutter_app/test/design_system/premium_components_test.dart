import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mindgraph_plus_plus/core/design_system/design_system.dart';
import 'package:mindgraph_plus_plus/core/theme/app_theme.dart';

void main() {
  testWidgets('theme includes AppSemanticColors extension', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: AppTheme.light,
        home: Builder(
          builder: (context) {
            final c = context.colors;
            return Text('${c.primary.value}');
          },
        ),
      ),
    );
    expect(find.byType(Text), findsOneWidget);
  });

  testWidgets('MetricCard renders with tokens', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: AppTheme.light,
        home: const Scaffold(
          body: MetricCard(label: 'Sessions', value: '12', trend: '+2', trendUp: true),
        ),
      ),
    );
    expect(find.text('Sessions'), findsOneWidget);
    expect(find.text('12'), findsOneWidget);
  });

  testWidgets('StatusBadge renders risk tiers', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: AppTheme.light,
        home: const Scaffold(body: RiskBadge(tier: 'high')),
      ),
    );
    expect(find.text('high'), findsOneWidget);
  });

  testWidgets('dark theme semantic colors differ from light', (tester) async {
    late AppSemanticColors lightColors;
    late AppSemanticColors darkColors;

    await tester.pumpWidget(
      MaterialApp(
        theme: AppTheme.light,
        home: Builder(builder: (c) {
          lightColors = c.colors;
          return const SizedBox();
        }),
      ),
    );

    await tester.pumpWidget(
      MaterialApp(
        theme: AppTheme.dark,
        home: Builder(builder: (c) {
          darkColors = c.colors;
          return const SizedBox();
        }),
      ),
    );

    expect(lightColors.background, isNot(darkColors.background));
  });
}
