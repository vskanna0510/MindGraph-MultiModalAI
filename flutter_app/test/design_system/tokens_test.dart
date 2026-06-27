import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mindgraph_plus_plus/core/design_system/tokens/app_spacing.dart';
import 'package:mindgraph_plus_plus/core/design_system/tokens/app_typography.dart';

void main() {
  test('spacing follows 8dp grid', () {
    expect(AppSpacing.s8 % 4, 0);
    expect(AppSpacing.s16, AppSpacing.s8 * 2);
    expect(AppSpacing.s48, AppSpacing.s8 * 6);
  });

  test('typography scale matches spec', () {
    expect(AppTypography.displayXl, 48);
    expect(AppTypography.headline, 32);
    expect(AppTypography.body, 16);
    expect(AppTypography.label, 12);
  });
}
