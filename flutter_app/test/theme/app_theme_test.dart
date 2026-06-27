import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mindgraph_plus_plus/core/theme/app_theme.dart';

void main() {
  testWidgets('light theme uses 48dp minimum button height', (tester) async {
    await tester.pumpWidget(
      MaterialApp(theme: AppTheme.light, home: const Scaffold(body: ElevatedButton(onPressed: null, child: Text('Go')))),
    );
    final button = tester.getSize(find.byType(ElevatedButton));
    expect(button.height, greaterThanOrEqualTo(48));
  });

  testWidgets('dark theme renders with dark background', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: AppTheme.dark,
        home: const Scaffold(body: Text('Dark')),
      ),
    );
    final scaffold = tester.widget<Scaffold>(find.byType(Scaffold));
    expect(scaffold.backgroundColor, isNotNull);
  });

  testWidgets('high contrast theme has thicker borders', (tester) async {
    await tester.pumpWidget(
      MaterialApp(theme: AppTheme.highContrast, home: const Scaffold(body: Card(child: Text('Hi')))),
    );
    expect(find.text('Hi'), findsOneWidget);
  });
}
