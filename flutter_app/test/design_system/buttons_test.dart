import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mindgraph_plus_plus/core/design_system/buttons/app_buttons.dart';

void main() {
  testWidgets('PrimaryButton has semantics and 48dp height', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: PrimaryButton(label: 'Continue', onPressed: () {}),
        ),
      ),
    );
    expect(find.bySemanticsLabel('Continue'), findsOneWidget);
    expect(tester.getSize(find.byType(ElevatedButton)).height, greaterThanOrEqualTo(48));
  });

  testWidgets('DangerButton renders label', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(body: DangerButton(label: 'Delete', onPressed: () {})),
      ),
    );
    expect(find.text('Delete'), findsOneWidget);
  });
}
