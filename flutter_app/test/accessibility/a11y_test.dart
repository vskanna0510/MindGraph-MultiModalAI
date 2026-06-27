import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mindgraph_plus_plus/core/animations/reduced_motion.dart';
import 'package:mindgraph_plus_plus/core/design_system/forms/app_forms.dart';

void main() {
  testWidgets('ConsentTile meets minimum touch target', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: ConsentTile(
            title: 'Data',
            description: 'Process locally',
            value: true,
            onChanged: (_) {},
          ),
        ),
      ),
    );
    final switchFinder = find.byType(SwitchListTile);
    expect(switchFinder, findsOneWidget);
    final size = tester.getSize(switchFinder);
    expect(size.height, greaterThanOrEqualTo(48));
  });

  testWidgets('reduced motion reads MediaQuery flag', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: MediaQuery(
          data: MediaQueryData(disableAnimations: true),
          child: _MotionProbe(),
        ),
      ),
    );
    expect(find.text('reduced'), findsOneWidget);
  });
}

class _MotionProbe extends StatelessWidget {
  const _MotionProbe();

  @override
  Widget build(BuildContext context) {
    return Text(ReducedMotion.of(context) ? 'reduced' : 'full');
  }
}
