import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mindgraph_plus_plus/core/design_system/cards/content_cards.dart';
import 'package:mindgraph_plus_plus/core/design_system/cards/surface_card.dart';
import 'package:mindgraph_plus_plus/core/design_system/charts/risk_gauge.dart';

void main() {
  testWidgets('SurfaceCard renders child', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(home: Scaffold(body: SurfaceCard(child: Text('Card')))),
    );
    expect(find.text('Card'), findsOneWidget);
  });

  testWidgets('MoodCard exposes mood semantics', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: MoodCard(selectedIndex: null, onSelected: (_) {}),
        ),
      ),
    );
    expect(find.bySemanticsLabel('Mood selector'), findsOneWidget);
  });

  testWidgets('RiskGauge shows non-diagnostic disclaimer', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(home: Scaffold(body: RiskGauge(risk: 0.4))),
    );
    expect(find.textContaining('not a diagnosis'), findsOneWidget);
  });

  testWidgets('RecommendationCard renders tier content', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: RecommendationCard(
            title: 'Breathing',
            description: 'Calm guidance',
            tier: 'medium',
          ),
        ),
      ),
    );
    expect(find.text('Breathing'), findsOneWidget);
  });
}
